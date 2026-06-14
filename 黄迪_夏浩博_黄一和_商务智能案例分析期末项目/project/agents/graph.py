"""LangGraph StateGraph — multi-agent orchestration with MemorySaver."""
import json
import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared state across all agents."""
    question: str
    resolved_question: str
    memory_context: str
    memory_snapshot: dict
    question_summary: str
    analysis_type: str
    task_plan: list

    # Data analyst outputs
    data_results: list
    data_summary: str
    query_strategy: str
    query_time_seconds: float
    sql_time_seconds: float

    # NLP outputs
    nlp_results: dict
    nlp_summary: str

    # Forecast outputs
    forecast_results: dict
    forecast_summary: str

    # Scenario Agent outputs
    scenario_results: dict
    scenario_summary: str

    # Visualization paths
    charts: list

    # Decision outputs
    recommendations: str

    # Final response
    final_response: str
    error: str


import re, json

# ── Chart selection logic (LLM-PRIMARY with keyword fallback) ──

CHART_TYPES_INFO = {
    "line": "time series trend line chart",
    "geo_map": "Brazil state-level geographic choropleth map",
    "state_bar": "bar chart: GMV/orders by customer state",
    "payment_bar": "bar chart: transactions by payment method",
    "payment_heatmap": "matrix heatmap: payment type × installments",
    "wordcloud": "positive/negative review keyword word cloud",
    "category_bar": "horizontal bar: top product categories by sales",
    "scatter": "scatter/bubble: product weight vs freight cost",
    "delivery_bar": "bar chart: on-time delivery rate by state",
    "basket_bar": "bar chart: average basket size by state",
}


NLP_TRIGGER_TERMS = [
    "review", "reviews", "comment", "comments", "complaint", "complaints",
    "sentiment", "theme", "themes", "topic", "topics",
    "feedback", "refund", "return",
    "评论", "评价", "差评", "好评", "情感", "主题", "投诉", "抱怨",
    "反馈", "满意", "不满意", "退货", "客服", "口碑",
]


def _select_charts_llm(question: str, analysis_type: str, data_results: list) -> set | None:
    """LLM-based chart selection (PRIMARY). Returns None if LLM fails."""
    from config.prompts import VISUALIZER_SYSTEM
    from utils.llm import chat
    from models.llm_outputs import ChartSelectionOutput, safe_parse_pydantic

    data_ctx = ""
    for i, r in enumerate(data_results[:3]):
        if r.get("data") and len(r.get("data", [])) > 0:
            cols = list(r["data"][0].keys())[:8]
            data_ctx += f"\nResult {i+1}: cols={cols}, rows={len(r['data'])}"
        if r.get("summary", ""):
            data_ctx += f"\n  Summary: {r['summary'][:150]}"

    chart_desc = "\n".join(f"- {k}: {v}" for k, v in CHART_TYPES_INFO.items())
    prompt = f"""Question: {question}
Analysis type: {analysis_type}{data_ctx}

Available chart types:
{chart_desc}

Select ALL chart types relevant to the question. Return ONLY a JSON array: ["type1", "type2", ...]"""

    try:
        response = chat(
            [{"role": "system", "content": VISUALIZER_SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200, timeout=25, json_mode=True,
        )
        parsed = safe_parse_pydantic(response, ChartSelectionOutput)
        if parsed and parsed.charts:
            logger.info("LLM charts: %s", parsed.charts)
            return set(parsed.charts)
    except Exception as e:
        logger.warning("LLM chart selection failed: %s", e)
    return None


def _select_charts(question: str, analysis_type: str, data_results: list) -> set:
    """Select chart types: LLM-first with keyword fallback."""
    clean_q = question.strip()
    charts = _select_charts_llm(clean_q, analysis_type, data_results)

    if charts is None:
        # KEYWORD FALLBACK
        q = clean_q.lower()
        charts = set()
        signals = {
            "line": ["trend", "趋势", "monthly", "月度", "sales trend", "销售趋势",
                      "overview", "概览", "预测", "predict", "forecast", "未来", "时间序列"],
            "geo_map": ["state", "州", "geo", "地理", "region", "区域", "map", "地图", "巴西", "brazil", "分布"],
            "payment_bar": ["payment", "支付", "installment", "分期"],
            "payment_heatmap": ["payment", "支付", "installment", "分期"],
            "wordcloud": ["review", "评论", "评价", "差评", "好评", "sentiment", "情感", "评分", "反馈", "退货"],
            "category_bar": ["category", "品类", "类别", "product category"],
            "scatter": ["weight", "重量", "freight", "运费", "shipping", "尺寸", "size", "关系", "关联", "correlation", "相关性"],
            "delivery_bar": ["delivery", "配送", "delay", "延迟", "on.time", "准时", "物流", "deliver", "ontime", "送达", "时效"],
            "basket_bar": ["basket", "客单价", "avg order", "average basket"],
        }
        for ck, kws in signals.items():
            if any(kw in q for kw in kws):
                charts.add(ck)
        if not charts:
            charts = {"line", "geo_map", "state_bar", "payment_bar", "payment_heatmap", "category_bar", "scatter"}

    # Apply analysis_type overrides
    if analysis_type in ("diagnostic", "prescriptive"):
        charts.add("wordcloud")
        charts.add("delivery_bar")
    if analysis_type == "predictive":
        charts.add("line")
    if "geo_map" in charts:
        charts.add("state_bar")
    if "wordcloud" in charts:
        charts.add("category_bar")

    logger.info("Selected charts: %s", charts)
    return charts


# ── Node functions ──────────────────────────────────────

def coordinator_node(state: AgentState) -> dict:
    """Parse user question and create task plan."""
    from agents.coordinator import parse_question

    question = state.get("resolved_question") or state.get("question", "")
    # Strip prior context for coordinator parsing (Bug 22/23)
    clean_question = question.strip()
    plan = parse_question(clean_question)

    return {
        "question_summary": plan.get("question_summary", clean_question),
        "analysis_type": plan.get("analysis_type", "descriptive"),
        "task_plan": plan.get("tasks", []),
    }


def data_analyst_node(state: AgentState) -> dict:
    """Execute data analyst tasks — iterate through task_plan if available."""
    from agents.data_analyst import analyze
    import time

    question = state.get("resolved_question") or state.get("question", "")
    # Strip prior context for clean query (Bug 22/23)
    clean_question = question.strip()
    task_plan = state.get("task_plan", [])
    analysis_type = state.get("analysis_type", "")
    all_results = []
    summaries = []
    strategy = "unknown"
    query_time = 0.0

    # Determine which sub-questions to send
    data_tasks = [t for t in task_plan if t.get("agent") == "data_analyst"]
    if not data_tasks:
        data_tasks = [{"task": clean_question}]

    # Auto-split multi-part questions when coordinator only produced 1 task
    if len(data_tasks) == 1:
        q = clean_question.lower()
        parts = []
        if any(w in q for w in ["哪个州", "销售额最高", "各州", "州排名"]):
            parts.append("查询各州销售额排名，使用 mv_state_sales 视图")
        if any(w in q for w in ["准时", "配送", "交付", "延迟"]):
            parts.append("查询平台整体交付准时率及各州延迟情况，使用 mv_delivery_perf 视图")
        if any(w in q for w in ["支付", "分期"]):
            parts.append("查询哪种支付方式最受欢迎，使用 mv_payment_dist 视图")
        if len(parts) >= 2:
            data_tasks = [{"task": p} for p in parts]
            logger.info("Auto-split multi-part query into %d sub-tasks", len(parts))

    logger.info("Data analyst processing %d sub-task(s)", len(data_tasks))

    for dt in data_tasks:
        sub_q = dt.get("task", clean_question)
        # Only inject original question context when there's exactly 1 sub-task
        # For multi-part auto-split, each sub-task is self-contained
        if len(data_tasks) == 1 and sub_q != clean_question:
            sub_q = f"[Original question]: {clean_question}\n[Sub-task]: {sub_q}"
        try:
            t0 = time.time()
            result = analyze(sub_q)
            dt_qtime = round(time.time() - t0, 2)
            query_time += dt_qtime
            if result.get("error"):
                logger.error("Data analyst error for '%s': %s",
                             sub_q[:80], result["error"])
                all_results.append(result)
                if strategy == "unknown":
                    strategy = "error"
                continue
            all_results.append(result)
            summaries.append(result.get("summary", ""))
            if strategy == "unknown" or result.get("strategy") == "base_table":
                strategy = result.get("strategy", strategy)
        except Exception as e:
            logger.error("Data analyst error: %s", e, exc_info=True)
            all_results.append({"error": str(e), "strategy": "error"})

    if not all_results and strategy == "unknown":
        return {"error": "No data analysis tasks completed",
                "query_strategy": "error", "query_time_seconds": 0.0}

    # Extract SQL execution times from results (Bug 32)
    sql_time = sum(r.get("sql_time", 0) for r in all_results if isinstance(r, dict))

    return {
        "data_results": all_results,
        "data_summary": " | ".join(s for s in summaries if s),
        "query_strategy": strategy,
        "query_time_seconds": round(query_time, 2),
        "sql_time_seconds": round(sql_time, 2) if sql_time > 0 else 0.0,
    }


def nlp_node(state: AgentState) -> dict:
    """Run NLP review analysis — uses LLM-based Portuguese sentiment (Bug 25.1).
    Uses clean question without MemorySaver context for keyword detection.
    """
    raw_question = state.get("resolved_question") or state.get("question", "")
    question = raw_question.strip().lower()
    task_plan = state.get("task_plan", [])

    nlp_needed = any(w in question for w in NLP_TRIGGER_TERMS)
    if not nlp_needed:
        nlp_tasks = [t for t in task_plan if t.get("agent") == "nlp_insight"]
        nlp_needed = any(
            any(term in str(t.get("task", "")).lower() for term in NLP_TRIGGER_TERMS)
            for t in nlp_tasks
        )

    if not nlp_needed:
        logger.info("NLP not needed for this query, skipping.")
        return {"nlp_results": {}, "nlp_summary": ""}

    from agents.nlp_insight import analyze_reviews

    try:
        nlp_results = analyze_reviews()
        if nlp_results.get("error"):
            logger.warning("NLP error: %s", nlp_results["error"])
            return {}
        return {
            "nlp_results": nlp_results,
            "nlp_summary": nlp_results.get("sentiment_summary", ""),
        }
    except Exception as e:
        logger.error("NLP node error: %s", e, exc_info=True)
        return {}


def predictor_node(state: AgentState) -> dict:
    """Run time series forecasting — skips if not predictive.
    Uses clean question without MemorySaver context.
    """
    atype = state.get("analysis_type", "")
    raw_question = state.get("resolved_question") or state.get("question", "")
    question = raw_question.strip().lower()
    task_plan = state.get("task_plan", [])

    pred_needed = atype == "predictive"
    pred_needed = pred_needed or any(w in question for w in
        ["预测", "predict", "forecast", "未来", "趋势预测"])
    if not pred_needed:
        pred_tasks = [t for t in task_plan if t.get("agent") == "predictor"]
        pred_needed = len(pred_tasks) > 0

    if not pred_needed:
        logger.info("Prediction not needed for this query, skipping.")
        return {"forecast_results": {}, "forecast_summary": ""}

    from models.predictor import forecast_monthly

    try:
        # Determine forecast horizon from question
        import re as _re
        weeks = 6  # default
        week_match = _re.search(r'(\d+)\s*周', question)
        day_match = _re.search(r'(\d+)\s*天', question)
        month_match = _re.search(r'(\d+)\s*个?月', question)
        if week_match:
            weeks = int(week_match.group(1))
        elif day_match:
            weeks = max(1, int(day_match.group(1)) // 7)
        elif month_match:
            weeks = int(month_match.group(1)) * 4

        forecast = forecast_monthly(weeks=weeks)
        if forecast.get("error"):
            logger.warning("Forecast error: %s", forecast["error"])
            return {}
        return {
            "forecast_results": forecast,
            "forecast_summary": forecast.get("model_summary", ""),
        }
    except Exception as e:
        logger.error("Predictor node error: %s", e, exc_info=True)
        return {}


def scenario_node(state: AgentState) -> dict:
    """Run What-if and anomaly checks when the question asks for scenario analysis."""
    from agents.scenario import run_scenario_analysis

    question = state.get("resolved_question") or state.get("question", "")
    analysis_type = state.get("analysis_type", "")

    try:
        scenario = run_scenario_analysis(question, analysis_type)
        return {
            "scenario_results": scenario,
            "scenario_summary": scenario.get("summary", ""),
        }
    except Exception as e:
        logger.error("Scenario node error: %s", e, exc_info=True)
        return {
            "scenario_results": {"error": str(e)},
            "scenario_summary": f"Scenario Agent failed: {e}",
        }


def visualizer_node(state: AgentState) -> dict:
    """Generate visualizations based on question content and data results."""
    from agents.visualizer import (
        line_chart, bar_chart, geo_heatmap, matrix_heatmap,
        scatter_bubble, wordcloud_image, forecast_chart,
        confidence_interval_summary,
    )
    from agents.data_analyst import (
        get_monthly_sales, get_state_sales, get_payment_distribution,
    )

    question = state.get("resolved_question") or state.get("question", "")
    analysis_type = state.get("analysis_type", "")
    data_results = state.get("data_results", [])

    selected = _select_charts(question, analysis_type, data_results)
    charts = []

    try:
        # 1. Line chart — monthly sales trend
        if "line" in selected:
            monthly = get_monthly_sales()
            if monthly:
                chart = line_chart(
                    monthly, x_col="ym",
                    y_cols=["total_gmv", "total_orders"],
                    title="Monthly Sales Trend",
                )
                charts.append({"type": "line", "title": "Monthly Sales Trend", **chart})

        # Separate forecast chart + CI summary
        forecast = state.get("forecast_results", {})
        fc = forecast.get("forecast")
        if fc and len(fc) > 0:
            granularity = forecast.get("forecast_granularity", "weekly")
            fc_chart = forecast_chart(
                fc,
                forecast_granularity=granularity,
                title="Sales Forecast with Confidence Interval",
            )
            charts.append({"type": "line", "title": "Sales Forecast", **fc_chart})

            ci_text = confidence_interval_summary(fc)
            charts.append({"type": "text", "title": "Confidence Intervals",
                           "png": None, "html": None, "text": ci_text})

        scenario_summary = state.get("scenario_summary", "")
        if scenario_summary:
            charts.append({
                "type": "text",
                "title": "Scenario Agent Summary",
                "png": None,
                "html": None,
                "text": f"### Scenario Agent Summary\n{scenario_summary}",
            })

        # 2. State sales bar chart
        if "state_bar" in selected:
            state_sales = get_state_sales()
            if state_sales:
                chart = bar_chart(
                    state_sales, x_col="customer_state", y_col="total_gmv",
                    title="GMV by State",
                )
                charts.append({"type": "bar", "title": "GMV by State", **chart})

        # 3. Geo heatmap
        if "geo_map" in selected:
            try:
                from utils.db import execute_query
                geo_data = execute_query("""
                    SELECT customer_state, SUM(total_gmv) AS total_gmv
                    FROM mv_state_sales
                    GROUP BY customer_state
                """)
                if geo_data:
                    chart = geo_heatmap(
                        geo_data, state_col="customer_state", value_col="total_gmv",
                        title="Brazil State Sales Distribution",
                    )
                    charts.append({"type": "geo_map", "title": "State Sales Geo Map", **chart})
            except Exception:
                pass

        # 4. Payment distribution bar
        if "payment_bar" in selected:
            payment = get_payment_distribution()
            if payment:
                chart = bar_chart(
                    payment, x_col="payment_type", y_col="total_transactions",
                    title="Payment Methods Distribution",
                )
                charts.append({"type": "bar", "title": "Payment Distribution", **chart})

        # 5. Avg basket by state
        if "basket_bar" in selected:
            try:
                from utils.db import execute_query
                basket_data = execute_query("""
                    SELECT customer_state,
                           ROUND(SUM(total_gmv) / SUM(total_orders), 0) AS avg_basket
                    FROM mv_state_sales
                    GROUP BY customer_state
                    ORDER BY avg_basket DESC
                """)
                if basket_data:
                    chart = bar_chart(
                        basket_data, x_col="customer_state", y_col="avg_basket",
                        title="Average Basket Size by State",
                    )
                    charts.append({"type": "bar", "title": "Avg Basket by State", **chart})
            except Exception:
                pass

        # 6. Top category sales
        if "category_bar" in selected:
            try:
                from utils.db import execute_query
                cat_data = execute_query("""
                    SELECT product_category_english,
                           SUM(total_gmv) AS total_gmv,
                           SUM(total_orders) AS total_orders
                    FROM mv_category_sales
                    GROUP BY product_category_english
                    ORDER BY total_gmv DESC
                    LIMIT 15
                """)
                if cat_data:
                    chart = bar_chart(
                        cat_data, x_col="product_category_english", y_col="total_gmv",
                        title="Top Categories by Sales",
                        orientation="h",
                    )
                    charts.append({"type": "bar", "title": "Top Category Sales", **chart})
            except Exception:
                pass

        # 7. Delivery performance — show ALL states (Bug 21)
        if "delivery_bar" in selected:
            try:
                from utils.db import execute_query
                delivery_data = execute_query("""
                    SELECT customer_state,
                           ROUND(AVG(on_time_rate), 1) AS avg_on_time,
                           SUM(delayed_orders) AS total_delayed
                    FROM mv_delivery_perf
                    GROUP BY customer_state
                    ORDER BY avg_on_time ASC
                """)
                if delivery_data:
                    chart = bar_chart(
                        delivery_data, x_col="customer_state", y_col="avg_on_time",
                        title="On-Time Delivery Rate by State (All States)",
                    )
                    charts.append({"type": "bar", "title": "Delivery Performance", **chart})
            except Exception:
                pass

        # 8. Scatter: weight vs freight
        if "scatter" in selected:
            try:
                from utils.db import execute_query
                wf = execute_query("""
                    SELECT ROUND(p.product_weight_g, -2) AS product_weight_g,
                           ROUND(oi.freight_value, 0) AS freight_value,
                           COUNT(*) AS order_count,
                           AVG(CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date
                                    THEN 1.0 ELSE 0.0 END) AS on_time_rate
                    FROM products p
                    JOIN order_items oi ON p.product_id = oi.product_id
                    JOIN orders o ON oi.order_id = o.order_id
                    WHERE p.product_weight_g < 50000 AND oi.freight_value < 200
                    GROUP BY 1, 2
                    HAVING COUNT(*) >= 2
                    LIMIT 2000
                """)
                if wf:
                    chart = scatter_bubble(
                        wf, x_col="product_weight_g", y_col="freight_value",
                        title="Product Weight vs Freight Cost",
                        size_col="order_count",
                        color_col="on_time_rate",
                    )
                    charts.append({"type": "scatter", "title": "Weight vs Freight", **chart})
            except Exception:
                pass

        # 9. Payment matrix heatmap
        if "payment_heatmap" in selected:
            try:
                from utils.db import execute_query
                ph = execute_query("""
                    SELECT payment_type,
                           payment_installments,
                           ROUND(
                               COUNT(*) * 100.0
                               / SUM(COUNT(*)) OVER (PARTITION BY payment_type),
                               1
                           ) AS share_pct
                    FROM payments
                    WHERE payment_installments BETWEEN 1 AND 12
                    GROUP BY payment_type, payment_installments
                """)
                if ph:
                    chart = matrix_heatmap(
                        ph, x_col="payment_installments", y_col="payment_type",
                        value_col="share_pct", title="Installment Distribution within Each Payment Type (%)",
                    )
                    charts.append({"type": "heatmap", "title": "Payment Matrix", **chart})
            except Exception:
                pass

        # 10. NLP word cloud (only if NLP data available and relevant)
        if "wordcloud" in selected:
            nlp = state.get("nlp_results", {})
            if nlp:
                pos_words = nlp.get("top_positive_keywords", [])
                neg_words = nlp.get("top_negative_keywords", [])
                if pos_words or neg_words:
                    chart = wordcloud_image(pos_words, neg_words)
                    charts.append({"type": "wordcloud", "title": "Review Word Cloud", **chart})

    except Exception as e:
        logger.error("Visualizer node error: %s", e, exc_info=True)

    return {"charts": charts}


def decision_node(state: AgentState) -> dict:
    """Generate business recommendations — injects actual data into prompt (Bug 25.3 + Bug 21)."""
    atype = state.get("analysis_type", "")
    question = (state.get("resolved_question") or state.get("question", "")).lower()

    presc_needed = atype in ("prescriptive", "diagnostic")
    presc_needed = presc_needed or any(w in question for w in
        ["建议", "优化", "改进", "策略", "recommend", "improve", "how to", "策略",
         "如何", "方案", "降低", "提升"])

    from agents.decision import generate_recommendations

    try:
        # Build data context with ACTUAL data rows (Bug 25.3)
        data_tables_text = ""
        data_results = state.get("data_results", [])
        for i, dr in enumerate(data_results):
            if dr.get("data") and len(dr.get("data", [])) > 0:
                rows = dr["data"]
                # Serialize first 20 rows as markdown table
                if rows:
                    cols = list(rows[0].keys())
                    data_tables_text += f"\n### Data Table {i+1}: {dr.get('summary', '')[:100]}\n"
                    data_tables_text += "| " + " | ".join(cols) + " |\n"
                    data_tables_text += "|" + "|".join(["---"] * len(cols)) + "|\n"
                    for row in rows[:20]:
                        vals = [str(v)[:40] if v is not None else "NULL" for v in row.values()]
                        data_tables_text += "| " + " | ".join(vals) + " |\n"
                    if len(rows) > 20:
                        data_tables_text += f"\n*(Showing 20 of {len(rows)} rows)*\n"

        recs = generate_recommendations(
            analysis_summary=state.get("data_summary", "No analysis available"),
            nlp_results=state.get("nlp_results"),
            forecast_summary=state.get("forecast_summary"),
            data_tables=data_tables_text,
            scenario_results=state.get("scenario_results"),
            memory_context=state.get("memory_context", ""),
        )
        return {"recommendations": recs}
    except Exception as e:
        logger.error("Decision node error: %s", e, exc_info=True)
        return {"recommendations": f"Unable to generate recommendations: {e}"}


def synthesis_node(state: AgentState) -> dict:
    """Synthesize all agent outputs into a final response."""
    parts = []

    question = state.get("question", "")
    parts.append("## Analysis Results\n")

    if state.get("error"):
        parts.append(f"**Error:** {state['error']}")

    if state.get("data_summary"):
        parts.append(f"### Data Insights\n{state['data_summary']}")

    if state.get("nlp_summary"):
        parts.append(f"\n### Review Sentiment\n{state['nlp_summary']}")

    if state.get("forecast_summary"):
        parts.append(f"\n### Forecast\n{state['forecast_summary']}")

    if state.get("scenario_summary"):
        parts.append(f"\n### Scenario Agent\n{state['scenario_summary']}")

    if state.get("recommendations"):
        parts.append(f"\n{state['recommendations']}")

    charts = state.get("charts", [])
    if charts:
        chart_list = "\n".join([f"- {c.get('title', 'Chart')} ({c.get('type', '?')})" for c in charts])
        parts.append(f"\n### Generated Charts\n{chart_list}")

    return {"final_response": "\n\n".join(parts)}


# ── Routing functions ───────────────────────────────────

def route_after_coordinator(state: AgentState) -> str:
    """Always go to data_analyst."""
    return "data_analyst"


def route_after_analyst(state: AgentState) -> str:
    """Route based on analysis_type (from coordinator) + plan_agents.

    Uses clean question WITHOUT MemorySaver context to avoid keyword contamination.
    """
    task_plan = state.get("task_plan", [])
    atype = state.get("analysis_type", "")
    # Clean question of MemorySaver context before keyword matching
    raw_question = state.get("resolved_question") or state.get("question", "")
    question = raw_question.strip().lower()

    plan_agents = {t.get("agent") for t in task_plan} if task_plan else set()

    # Trigger NLP only when the current question/task is actually about reviews.
    nlp_needed = any(w in question for w in NLP_TRIGGER_TERMS)
    if not nlp_needed:
        nlp_needed = any(
            t.get("agent") == "nlp_insight"
            and any(term in str(t.get("task", "")).lower() for term in NLP_TRIGGER_TERMS)
            for t in task_plan
        )

    pred_needed = "predictor" in plan_agents
    if not pred_needed:
        pred_needed = atype == "predictive"
        pred_needed = pred_needed or any(w in question for w in
            ["预测", "predict", "forecast", "未来"])

    logger.info("Routing after analyst — NLP: %s, Predictor: %s (type: %s, plan: %s)",
                nlp_needed, pred_needed, atype, plan_agents)

    if nlp_needed:
        return "nlp"
    if pred_needed:
        return "predictor"
    return "visualizer"


def route_after_nlp(state: AgentState) -> str:
    """After NLP, go to predictor or visualizer.
    Uses clean question without MemorySaver context.
    """
    task_plan = state.get("task_plan", [])
    plan_agents = {t.get("agent") for t in task_plan} if task_plan else set()

    pred_needed = "predictor" in plan_agents
    if not pred_needed:
        atype = state.get("analysis_type", "")
        raw_question = state.get("resolved_question") or state.get("question", "")
        question = raw_question.strip().lower()
        pred_needed = atype == "predictive"
        pred_needed = pred_needed or any(w in question for w in
            ["预测", "predict", "forecast", "未来"])
    return "predictor" if pred_needed else "visualizer"


def route_after_visualizer(state: AgentState) -> str:
    """After visualizer: go to decision then synthesis."""
    return "decision"


# ── Build graph ────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the StateGraph with MemorySaver."""
    graph = StateGraph(AgentState)

    graph.add_node("coordinator", coordinator_node)
    graph.add_node("data_analyst", data_analyst_node)
    graph.add_node("nlp", nlp_node)
    graph.add_node("predictor", predictor_node)
    graph.add_node("scenario", scenario_node)
    graph.add_node("visualizer", visualizer_node)
    graph.add_node("decision", decision_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("coordinator")
    graph.add_edge("coordinator", "data_analyst")

    graph.add_conditional_edges("data_analyst", route_after_analyst, {
        "nlp": "nlp",
        "predictor": "predictor",
        "visualizer": "scenario",
    })

    graph.add_conditional_edges("nlp", route_after_nlp, {
        "predictor": "predictor",
        "visualizer": "scenario",
    })

    graph.add_edge("predictor", "scenario")
    graph.add_edge("scenario", "visualizer")
    graph.add_edge("visualizer", "decision")
    graph.add_edge("decision", "synthesis")
    graph.add_edge("synthesis", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Singleton
_graph = None


def get_graph():
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(question: str, thread_id: str = "default") -> AgentState:
    """Run a user question through the agent graph.

    Uses LangGraph persistence for conversation continuity.
    Prior context stored in data_summary field, NOT appended to question string.
    This prevents MemorySaver context from contaminating keyword matching.
    """
    import time
    from agents.memory import (
        build_memory_snapshot,
        get_thread_memory,
        resolve_follow_up_question,
        save_thread_memory,
    )

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Retrieve prior conversation context from checkpointer — store as separate field
    prior_context = ""
    memory_snapshot = {}
    resolved_question = question
    memory_context = ""
    stored_values = get_thread_memory(thread_id)
    if stored_values:
        memory_snapshot = build_memory_snapshot(stored_values)
        resolved_question, memory_context = resolve_follow_up_question(question, memory_snapshot)
        prev_data = stored_values.get("data_summary", "")
        prev_q = stored_values.get("question", "")
        prev_recs = stored_values.get("recommendations", "")
        if prev_q:
            clean_prev_q = prev_q.strip()
            prior_context = (
                f"Previous question: {clean_prev_q}\n"
                f"Previous findings: {prev_data[:200]}\n"
                f"Previous recommendations: {prev_recs[:200]}"
            )
    else:
        pass
    try:
        prior_state = None if stored_values else graph.get_state(config)
        if prior_state and prior_state.values:
            memory_snapshot = build_memory_snapshot(prior_state.values)
            resolved_question, memory_context = resolve_follow_up_question(question, memory_snapshot)
            prev_data = prior_state.values.get("data_summary", "")
            prev_q = prior_state.values.get("question", "")
            prev_recs = prior_state.values.get("recommendations", "")
            if prev_q:
                clean_prev_q = prev_q.strip()
                prior_context = (
                    f"Previous question: {clean_prev_q}\n"
                    f"Previous findings: {prev_data[:200]}\n"
                    f"Previous recommendations: {prev_recs[:200]}"
                )
    except Exception:
        pass

    graph_question = resolved_question if memory_context else question

    initial_state = {
        "question": graph_question,
        "resolved_question": resolved_question,
        "memory_context": memory_context,
        "memory_snapshot": memory_snapshot,
        "question_summary": "",
        "analysis_type": "",
        "task_plan": [],
        "data_results": [],
        "data_summary": prior_context,  # Prior context here — used by decision, NOT by routing
        "query_strategy": "unknown",
        "query_time_seconds": 0.0,
        "sql_time_seconds": 0.0,
        "nlp_results": {},
        "nlp_summary": "",
        "forecast_results": {},
        "forecast_summary": "",
        "scenario_results": {},
        "scenario_summary": "",
        "charts": [],
        "recommendations": "",
        "final_response": "",
        "error": "",
    }

    t0 = time.time()
    result = graph.invoke(initial_state, config)
    result["query_time_seconds"] = round(time.time() - t0, 1)
    result["question"] = question
    result["resolved_question"] = resolved_question
    result["memory_context"] = memory_context
    result["memory_snapshot"] = memory_snapshot
    save_thread_memory(thread_id, result)
    return result
