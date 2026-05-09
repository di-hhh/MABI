"""LangGraph StateGraph — multi-agent orchestration with MemorySaver."""
import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared state across all agents."""
    question: str
    question_summary: str
    analysis_type: str
    task_plan: list

    # Data analyst outputs
    data_results: list
    data_summary: str
    query_strategy: str
    query_time_seconds: float

    # NLP outputs
    nlp_results: dict
    nlp_summary: str

    # Forecast outputs
    forecast_results: dict
    forecast_summary: str

    # Visualization paths
    charts: list

    # Decision outputs
    recommendations: str

    # Final response
    final_response: str
    error: str


# ── Chart selection logic (Bug 16.1) ─────────────────────

def _select_charts(question: str, analysis_type: str, data_results: list) -> set:
    """Determine which chart types to generate based on question content.
    Returns a set of chart keys: line, geo_map, state_bar, payment_bar,
    payment_heatmap, wordcloud, category_bar, scatter, delivery_bar, basket_bar.
    """
    q = question.lower()
    charts = set()

    # Signal-based selection
    signals = {
        "line": ["trend", "趋势", "monthly", "月度", "sales trend", "销售趋势",
                  "overview", "概览", "预测", "predict", "forecast", "未来"],
        "geo_map": ["state", "州", "geo", "地理", "region", "区域", "map",
                     "地图", "巴西", "brazil"],
        "payment_bar": ["payment", "支付", "installment", "分期"],
        "payment_heatmap": ["payment", "支付", "installment", "分期"],
        "wordcloud": ["review", "评论", "评价", "差评", "好评", "sentiment",
                       "情感", "评分"],
        "category_bar": ["category", "品类", "类别", "product category"],
        "scatter": ["weight", "重量", "freight", "运费", "shipping",
                     "产品重量", "尺寸", "size"],
        "delivery_bar": ["delivery", "配送", "delay", "延迟", "on.time",
                          "准时", "物流", "deliver", "delivery time"],
        "basket_bar": ["basket", "客单价", "avg order", "average basket"],
    }

    for chart_key, keywords in signals.items():
        if any(kw in q for kw in keywords):
            charts.add(chart_key)

    # analysis_type overrides
    if analysis_type in ("diagnostic", "prescriptive"):
        charts.add("wordcloud")
    if analysis_type == "predictive":
        charts.add("line")

    # Default: if question is very general or empty signals, show core charts
    if not charts:
        charts = {"line", "geo_map", "state_bar", "payment_bar", "payment_heatmap",
                   "category_bar", "scatter"}
    # Always include state bar when geo_map is requested
    if "geo_map" in charts:
        charts.add("state_bar")
    # Always include category bar when wordcloud is requested (review context)
    if "wordcloud" in charts:
        charts.add("category_bar")

    logger.info("Selected charts for query: %s", charts)
    return charts


# ── Node functions ──────────────────────────────────────

def coordinator_node(state: AgentState) -> dict:
    """Parse user question and create task plan."""
    from agents.coordinator import parse_question

    question = state.get("question", "")
    plan = parse_question(question)

    return {
        "question_summary": plan.get("question_summary", question),
        "analysis_type": plan.get("analysis_type", "descriptive"),
        "task_plan": plan.get("tasks", []),
    }


def data_analyst_node(state: AgentState) -> dict:
    """Execute data analyst tasks — iterate through task_plan if available (Bug 16.3)."""
    from agents.data_analyst import analyze
    import time

    question = state.get("question", "")
    task_plan = state.get("task_plan", [])
    all_results = []
    summaries = []
    strategy = "unknown"
    query_time = 0.0

    # Determine which sub-questions to send
    data_tasks = [t for t in task_plan if t.get("agent") == "data_analyst"]
    if not data_tasks:
        # Fallback: single query
        data_tasks = [{"task": question}]

    logger.info("Data analyst processing %d sub-task(s)", len(data_tasks))

    for dt in data_tasks:
        sub_q = dt.get("task", question)
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

    return {
        "data_results": all_results,
        "data_summary": " | ".join(s for s in summaries if s),
        "query_strategy": strategy,
        "query_time_seconds": round(query_time, 2),
    }


def nlp_node(state: AgentState) -> dict:
    """Run NLP review analysis — skips if not needed."""
    question = state.get("question", "").lower()
    atype = state.get("analysis_type", "")
    task_plan = state.get("task_plan", [])

    nlp_needed = any(w in question for w in
        ["review", "评论", "评价", "评分", "差评", "好评", "sentiment", "情感"])
    nlp_needed = nlp_needed or atype in ("diagnostic", "prescriptive")
    # Also check task_plan for nlp_insight tasks
    if not nlp_needed:
        nlp_tasks = [t for t in task_plan if t.get("agent") == "nlp_insight"]
        nlp_needed = len(nlp_tasks) > 0

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
    """Run time series forecasting — skips if not predictive."""
    atype = state.get("analysis_type", "")
    question = state.get("question", "").lower()
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
        forecast = forecast_monthly(weeks=6)
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


def visualizer_node(state: AgentState) -> dict:
    """Generate visualizations based on question content (Bug 16.1 fix)."""
    from agents.visualizer import (
        line_chart, bar_chart, geo_heatmap, matrix_heatmap,
        scatter_bubble, wordcloud_image, forecast_chart,
        confidence_interval_summary,
    )
    from agents.data_analyst import (
        get_monthly_sales, get_state_sales, get_payment_distribution,
    )

    question = state.get("question", "")
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

        # Separate forecast chart + CI summary (Bug 17)
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
                    LIMIT 15
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

        # 7. Delivery performance (new chart for delivery queries)
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
                    LIMIT 15
                """)
                if delivery_data:
                    chart = bar_chart(
                        delivery_data, x_col="customer_state", y_col="avg_on_time",
                        title="On-Time Delivery Rate by State",
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
                    SELECT payment_type, payment_installments, COUNT(*) AS cnt
                    FROM payments WHERE payment_installments <= 12
                    GROUP BY payment_type, payment_installments
                """)
                if ph:
                    chart = matrix_heatmap(
                        ph, x_col="payment_installments", y_col="payment_type",
                        value_col="cnt", title="Payment Type x Installments",
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
    """Generate business recommendations — enhanced for prescriptive queries."""
    atype = state.get("analysis_type", "")
    question = state.get("question", "").lower()

    presc_needed = atype in ("prescriptive", "diagnostic")
    presc_needed = presc_needed or any(w in question for w in
        ["建议", "优化", "改进", "策略", "recommend", "improve", "how to", "策略"])

    from agents.decision import generate_recommendations

    try:
        recs = generate_recommendations(
            analysis_summary=state.get("data_summary", "No analysis available"),
            nlp_results=state.get("nlp_results"),
            forecast_summary=state.get("forecast_summary"),
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

    if state.get("recommendations"):
        parts.append(f"\n{state['recommendations']}")

    charts = state.get("charts", [])
    if charts:
        chart_list = "\n".join([f"- {c.get('title', 'Chart')} ({c.get('type', '?')})" for c in charts])
        parts.append(f"\n### Generated Charts\n{chart_list}")

    return {"final_response": "\n\n".join(parts)}


# ── Routing functions (Bug 16.2: use task_plan) ─────────

def route_after_coordinator(state: AgentState) -> str:
    """Always go to data_analyst."""
    return "data_analyst"


def route_after_analyst(state: AgentState) -> str:
    """Route based on coordinator's task_plan, with fallback to keyword matching."""
    task_plan = state.get("task_plan", [])
    atype = state.get("analysis_type", "")
    question = state.get("question", "").lower()

    # Check task_plan for agent assignments
    plan_agents = {t.get("agent") for t in task_plan} if task_plan else set()

    nlp_needed = "nlp_insight" in plan_agents
    pred_needed = "predictor" in plan_agents

    # Fallback: keyword-based detection
    if not nlp_needed:
        nlp_needed = any(w in question for w in
            ["review", "评论", "评价", "评分", "差评", "好评", "sentiment"])
        nlp_needed = nlp_needed or atype in ("diagnostic", "prescriptive")

    if not pred_needed:
        pred_needed = atype == "predictive"
        pred_needed = pred_needed or any(w in question for w in
            ["预测", "predict", "forecast", "未来"])

    logger.info("Routing after analyst — NLP: %s, Predictor: %s (plan_agents: %s)",
                nlp_needed, pred_needed, plan_agents)

    if nlp_needed:
        return "nlp"
    if pred_needed:
        return "predictor"
    return "visualizer"


def route_after_nlp(state: AgentState) -> str:
    """After NLP, go to predictor or visualizer."""
    task_plan = state.get("task_plan", [])
    plan_agents = {t.get("agent") for t in task_plan} if task_plan else set()

    pred_needed = "predictor" in plan_agents
    if not pred_needed:
        atype = state.get("analysis_type", "")
        question = state.get("question", "").lower()
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
    graph.add_node("visualizer", visualizer_node)
    graph.add_node("decision", decision_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("coordinator")
    graph.add_edge("coordinator", "data_analyst")

    graph.add_conditional_edges("data_analyst", route_after_analyst, {
        "nlp": "nlp",
        "predictor": "predictor",
        "visualizer": "visualizer",
    })

    graph.add_conditional_edges("nlp", route_after_nlp, {
        "predictor": "predictor",
        "visualizer": "visualizer",
    })

    graph.add_edge("predictor", "visualizer")
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

    Args:
        question: Natural language question.
        thread_id: Conversation thread identifier (for MemorySaver).

    Returns:
        Final AgentState with all agent outputs.
    """
    import time
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Build initial state — inject prior context for continuity (Bug 16.4)
    prior_context = ""
    try:
        prior_state = graph.get_state(config)
        if prior_state and prior_state.values:
            prev_summary = prior_state.values.get("data_summary", "")
            prev_question = prior_state.values.get("question", "")
            if prev_summary and prev_question:
                prior_context = (
                    f"\n\n[Previous Analysis Context]\n"
                    f"Question: {prev_question}\n"
                    f"Answer summary: {prev_summary[:500]}\n"
                    f"Use this context to resolve references like '它', 'its', 'the' in the current question."
                )
    except Exception:
        pass

    question_with_context = question + prior_context if prior_context else question

    initial_state = {
        "question": question_with_context,
        "question_summary": "",
        "analysis_type": "",
        "task_plan": [],
        "data_results": [],
        "data_summary": "",
        "query_strategy": "unknown",
        "query_time_seconds": 0.0,
        "nlp_results": {},
        "nlp_summary": "",
        "forecast_results": {},
        "forecast_summary": "",
        "charts": [],
        "recommendations": "",
        "final_response": "",
        "error": "",
    }

    t0 = time.time()
    result = graph.invoke(initial_state, config)
    result["query_time_seconds"] = round(time.time() - t0, 1)
    return result
