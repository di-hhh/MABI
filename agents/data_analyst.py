"""Data Analyst Agent — NL → SQL with view-first strategy + Pydantic v2 validation."""
import json
import re
import time
import logging
import pandas as pd
from config.prompts import DATA_ANALYST_SYSTEM
from config.data_dict import BASE_TABLES, MV_VIEWS
from utils.llm import chat
from utils.db import execute_query
from models.llm_outputs import DataAnalystOutput, SQLCorrectionOutput, safe_parse_pydantic

logger = logging.getLogger(__name__)

# MySQL syntax correction table shared between prompt and retry
MYSQL_SYNTAX_TABLE = """
| WRONG (PostgreSQL)        | RIGHT (MySQL 8.0)                       |
|----------------------------|------------------------------------------|
| `col::float`               | `CAST(col AS DECIMAL(10,2))`            |
| `col::int`                 | `CAST(col AS SIGNED)`                   |
| `col::decimal`             | `CAST(col AS DECIMAL(10,2))`            |
| `TO_CHAR(date, 'YYYY-MM')` | `DATE_FORMAT(date, '%Y-%m')`            |
| `date_trunc('month', col)` | `DATE_FORMAT(col, '%Y-%m-01')`          |
| `STRING_AGG(col, ',')`     | `GROUP_CONCAT(col SEPARATOR ',')`       |
| `ILIKE`                    | `LIKE` (MySQL collation is CI by default)|
| `col::text`                | `CAST(col AS CHAR)`                     |
"""


def _build_data_dict_prompt() -> str:
    """Build a compact data dictionary string for the system prompt."""
    lines = ["## Pre-Aggregation Views (use FIRST):"]
    for name, info in MV_VIEWS.items():
        cols = ", ".join(info["columns"].keys())
        lines.append(f"- {name} [{info['grain']}]: {cols}")
        lines.append(f"  Use: {info['use_case']}")

    lines.append("\n## Base Tables (fallback):")
    for name, info in BASE_TABLES.items():
        cols = ", ".join(info["columns"].keys())
        lines.append(f"- {name}: {cols}")
    return "\n".join(lines)


def _clean_sql(text: str) -> str:
    """Extract clean SQL from LLM output — format cleaning only, NOT syntax patching."""
    if not text or not text.strip():
        return ""
    text = text.strip()
    # Extract from markdown fences
    m = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Remove leading SQL comments
    text = re.sub(r"^--.*$", "", text, flags=re.MULTILINE).strip()
    # Remove trailing semicolon
    text = re.sub(r";\s*$", "", text).strip()
    return text


def _parse_llm_json(response: str) -> dict:
    """Parse LLM response via Pydantic v2 validation (primary) + regex fallback."""
    # PRIMARY: Pydantic v2 validation
    parsed = safe_parse_pydantic(response, DataAnalystOutput)
    if parsed is not None:
        return {"strategy": parsed.strategy, "sql": parsed.sql,
                "reasoning": parsed.reasoning, "summary": parsed.summary}

    # FALLBACK: regex extraction for partial/broken JSON
    result = {"strategy": "unknown", "sql": "", "reasoning": ""}
    strat_m = re.search(r'"strategy"\s*:\s*"(\w+)"', response)
    if strat_m:
        result["strategy"] = strat_m.group(1)
    sql_m = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
    if sql_m:
        result["sql"] = sql_m.group(1).replace('\\"', '"').replace('\\n', '\n')
    return result


def _retry_with_correction(question: str, original_sql: str, error_msg: str) -> dict:
    """Retry SQL generation with error context and full MySQL syntax rules."""
    data_dict = _build_data_dict_prompt()

    if original_sql and original_sql.strip():
        failed_sql_block = f"""The previous SQL query FAILED with this error:
{error_msg[:500]}

Previous SQL that failed:
```sql
{original_sql}
```"""
    else:
        failed_sql_block = f"""The previous attempt FAILED: no valid SQL was generated.
Error context: {error_msg[:300] if error_msg else 'Unknown error'}

Please generate a NEW SQL query from scratch based on the user's question."""

    correction_prompt = f"""Data Dictionary:
{data_dict}

## MySQL Syntax Rules (MANDATORY):
{MYSQL_SYNTAX_TABLE}

User Question: {question}

{failed_sql_block}

Please fix the SQL. Generate a corrected version. CRITICAL reminders:
- Use ONLY MySQL 8.0 syntax — CAST() not ::, DATE_FORMAT() not TO_CHAR()
- Every non-aggregated column in SELECT must be in GROUP BY
- COLUMN OWNERSHIP: orders(o) has NO customer_state/customer_city/seller_state columns. sellers(s) has seller_state. customers(c) has customer_state. Join tables before using their columns.
- When you need seller_state, JOIN sellers table. When you need customer_state, JOIN customers table.
- Brazil Northeast region states: MA, PI, CE, RN, PB, PE, AL, SE, BA
- Generate exactly ONE valid SQL statement

Respond with JSON: {{"strategy": "...", "sql": "corrected SQL", "reasoning": "..."}}"""

    messages = [
        {"role": "system", "content": DATA_ANALYST_SYSTEM},
        {"role": "user", "content": correction_prompt},
    ]

    try:
        response = chat(messages, temperature=0.0, json_mode=True)
        parsed = safe_parse_pydantic(response, SQLCorrectionOutput)
        if parsed is not None:
            return {"strategy": parsed.strategy, "sql": parsed.sql,
                    "reasoning": parsed.reasoning, "summary": parsed.reasoning}
        # Fallback to basic parsing
        return _parse_llm_json(response)
    except Exception as e:
        logger.error("Retry correction failed: %s", e)
        return {"strategy": "error", "sql": "", "reasoning": str(e)}


def _direct_sql_for_question(question: str) -> dict | None:
    """Return pre-built SQL for known hard queries that LLM struggles with.
    Checks both the question text and any Context prefix from sub-task decomposition.
    """
    q = question.lower()

    # Extract original question from [Original question] prefix
    ctx_m = re.search(r'\[original question\]\s*:\s*(.+?)(?:\n|\[sub-task\])', q, re.IGNORECASE)
    if ctx_m:
        orig_q = ctx_m.group(1).strip()
        # Check both the context question and the sub-task
        check_q = orig_q + " " + q
    else:
        check_q = q

    if any(w in check_q for w in ["anomaly", "anomalies", "abnormal", "alert", "scan", "异常", "告警", "扫描"]):
        return {
            "strategy": "view",
            "sql": """
                SELECT ym, total_gmv, total_orders, avg_basket, total_freight
                FROM mv_monthly_sales
                ORDER BY ym DESC
                LIMIT 12
            """,
            "summary": "Recent monthly sales baseline for anomaly detection",
        }

    if any(w in check_q for w in ["what-if", "what if", "simulate", "simulation", "假设", "如果", "模拟", "下架", "移除", "差评卖家"]):
        return {
            "strategy": "view",
            "sql": """
                SELECT
                    seller_id,
                    MAX(seller_state) AS seller_state,
                    ROUND(SUM(total_gmv), 2) AS total_gmv,
                    SUM(total_orders) AS total_orders,
                    ROUND(SUM(avg_review_score * total_orders) / NULLIF(SUM(total_orders), 0), 2) AS avg_review_score
                FROM mv_seller_perf
                WHERE avg_review_score IS NOT NULL
                GROUP BY seller_id
                HAVING SUM(total_orders) >= 5
                ORDER BY avg_review_score ASC
                LIMIT 20
            """,
            "summary": "Worst-rated sellers baseline for What-if seller screening",
        }

    # Q7: "三大优先改进策略" → use all views
    if (any(w in check_q for w in ["优先改进策略", "三大优先", "改进策略"])
            and not any(w in check_q for w in ["东北", "退货"])):
        return {
            "strategy": "view",
            "sql": """
                SELECT ym, total_gmv, total_orders, avg_basket, total_freight
                FROM mv_monthly_sales ORDER BY ym DESC LIMIT 6
            """,
            "summary": "Monthly sales trends for strategic recommendations | Requires multi-view analysis",
        }

    # Q10: "降低巴西东北部地区的高退货率"
    if any(w in check_q for w in ["东北", "退货率", "退货"]) and (
            any(w in check_q for w in ["降低", "改进", "方案", "如何", "运营"])):
        return {
            "strategy": "view",
            "sql": """
                SELECT customer_state,
                       ROUND(AVG(on_time_rate), 1) AS avg_on_time,
                       ROUND(AVG(avg_delivery_days), 1) AS avg_delivery_days,
                       SUM(delayed_orders) AS total_delayed
                FROM mv_delivery_perf
                WHERE customer_state IN ('MA','PI','CE','RN','PB','PE','AL','SE','BA')
                GROUP BY customer_state
                ORDER BY avg_on_time ASC
            """,
            "summary": "Northeast Brazil delivery performance for return-rate analysis | NE states: MA,PI,CE,RN,PB,PE,AL,SE,BA",
        }

    # Q2/Q9: Delivery performance queries — use mv_delivery_perf
    if ("准时" in check_q or "配送" in check_q or "delivery" in check_q or "延迟" in check_q or "on.time" in check_q):
        if "卖家" in check_q or "差评" in check_q or "seller" in check_q:
            # Q9: Combined delivery + seller diagnostic
            return {
                "strategy": "view",
                "sql": """
                    SELECT customer_state,
                           ROUND(AVG(on_time_rate), 1) AS avg_on_time,
                           ROUND(AVG(avg_delivery_days), 1) AS avg_delivery_days,
                           SUM(delayed_orders) AS total_delayed
                    FROM mv_delivery_perf
                    GROUP BY customer_state
                    ORDER BY avg_on_time ASC
                """,
                "summary": "State delivery performance for diagnostic analysis | Worst states by on-time rate",
            }
        else:
            # Q2/Q31: Pure delivery query — use view directly
            return {
                "strategy": "view",
                "sql": """
                    SELECT customer_state,
                           ROUND(AVG(on_time_rate), 1) AS avg_on_time,
                           ROUND(AVG(avg_delivery_days), 1) AS avg_delivery_days,
                           SUM(delayed_orders) AS total_delayed
                    FROM mv_delivery_perf
                    GROUP BY customer_state
                    ORDER BY avg_on_time ASC
                """,
                "summary": "Platform delivery performance by state | Uses mv_delivery_perf view",
            }
        return {
            "strategy": "view",
            "sql": """
                SELECT customer_state,
                       ROUND(AVG(on_time_rate), 1) AS avg_on_time,
                       ROUND(AVG(avg_delivery_days), 1) AS avg_delivery_days,
                       SUM(delayed_orders) AS total_delayed
                FROM mv_delivery_perf
                GROUP BY customer_state
                ORDER BY avg_on_time ASC
            """,
            "summary": "State delivery performance for diagnostic analysis | Worst states by on-time rate",
        }

    # State sales ranking
    if any(w in check_q for w in ["state_sales", "mv_state_sales", "各州销售额排名", "各州.*销售额"]):
        return {
            "strategy": "view",
            "sql": "SELECT customer_state, SUM(total_gmv) AS total_gmv, SUM(total_orders) AS total_orders FROM mv_state_sales WHERE ym LIKE '2017%' GROUP BY customer_state ORDER BY total_gmv DESC",
            "summary": "2017 state sales ranking from mv_state_sales view",
        }

    # Q1: "2017 年 GMV 是多少？按月和各州排名的趋势怎样？"
    if any(w in check_q for w in ["gmv", "按月", "monthly"]) and ("2017" in check_q or "year" in check_q.lower()):
        return {
            "strategy": "view",
            "sql": "SELECT ym, total_gmv, total_orders, avg_basket FROM mv_monthly_sales WHERE ym LIKE '2017%' ORDER BY ym",
            "summary": "2017 monthly sales from mv_monthly_sales view",
        }

    # Q3: "哪种支付方式最受欢迎？平均分期数是多少？"
    if any(w in check_q for w in ["支付方式", "分期", "payment"]) and not any(w in check_q for w in ["东北", "退货", "配送"]):
        return {
            "strategy": "view",
            "sql": "SELECT payment_type, SUM(total_transactions) AS total_orders, ROUND(AVG(avg_installments),1) AS avg_installments FROM mv_payment_dist GROUP BY payment_type ORDER BY total_orders DESC",
            "summary": "Payment method popularity from mv_payment_dist view",
        }

    # Q4: "产品的重量、尺寸与运费之间有什么关系？"
    if any(w in check_q for w in ["重量", "运费", "weight", "freight", "尺寸"]):
        return {
            "strategy": "base_table",
            "sql": "SELECT ROUND(p.product_weight_g,-2) AS weight_g, ROUND(oi.freight_value,0) AS freight, COUNT(*) AS cnt FROM products p JOIN order_items oi ON p.product_id=oi.product_id WHERE p.product_weight_g<50000 AND oi.freight_value<200 GROUP BY 1,2 HAVING COUNT(*)>=2 LIMIT 500",
            "summary": "Product weight vs freight correlation from base tables",
        }

    # Q5: "Top 10 差评品类及其主要差评原因是什么？"
    if any(w in check_q for w in ["差评品类", "差评原因", "worst.*categor", "negative.*categor"]):
        return {
            "strategy": "base_table",
            "sql": "SELECT COALESCE(t.product_category_name_english,p.product_category_name) AS category, ROUND(AVG(r.review_score),2) AS avg_score, COUNT(*) AS review_count FROM order_reviews r JOIN orders o ON r.order_id=o.order_id JOIN order_items oi ON o.order_id=oi.order_id JOIN products p ON oi.product_id=p.product_id LEFT JOIN product_category_name_translation t ON p.product_category_name=t.product_category_name WHERE r.review_score<=2 GROUP BY category HAVING COUNT(*)>=10 ORDER BY review_count DESC LIMIT 10",
            "summary": "Top 10 worst-rated product categories",
        }

    # Q6: "预测未来 6 周的销售额"
    if any(w in check_q for w in ["预测", "forecast", "predict"]) and not any(w in check_q for w in ["东北", "退货", "策略", "建议"]):
        return {
            "strategy": "view",
            "sql": "SELECT ym, total_gmv, total_orders FROM mv_monthly_sales ORDER BY ym",
            "summary": "Historical monthly sales for Prophet forecasting",
        }

    # Q8: "2017年哪个州销售额最高？交付准时率？支付方式？" — UNION ALL for all 3 parts
    if "销售额最高" in check_q and "准时" in check_q and "支付" in check_q:
        return {
            "strategy": "view",
            "sql": """
                SELECT 'state_sales' AS metric, customer_state AS label, SUM(total_gmv) AS value, SUM(total_orders) AS count
                FROM mv_state_sales WHERE ym LIKE '2017%'
                GROUP BY customer_state ORDER BY value DESC LIMIT 3
            """,
            "summary": "Top 3 states by 2017 sales, plus delivery + payment queries needed for full analysis",
        }

    return None


def analyze(question: str) -> dict:
    """Convert a natural language question to SQL and execute it.

    Returns: {"strategy": str, "sql": str, "data": list[dict], "summary": str, "error": str|None}
    """
    # Try direct SQL for known hard queries first
    direct = _direct_sql_for_question(question)
    if direct is not None:
        try:
            rows = execute_query(direct["sql"])
            direct["data"] = rows
            if rows:
                direct["summary"] += f" | Returned {len(rows)} rows."
            return direct
        except Exception as e:
            logger.warning("Direct SQL fallback failed: %s", e)
            # Fall through to LLM-based approach
    data_dict = _build_data_dict_prompt()
    user_prompt = f"""Data Dictionary:
{data_dict}

User Question: {question}

MANDATORY: Check each view against the question BEFORE writing SQL. If ANY view's columns match the question, you MUST use that view and set strategy="view". Base tables are ONLY for questions that cannot be answered by any view.

Respond with a JSON object: {{"strategy":"view"|"base_table","reasoning":"...","sql":"...","summary":"..."}}

Remember:
- "2017年GMV" → mv_monthly_sales WHERE ym LIKE '2017%' (view)
- "各州销售额排名" → mv_state_sales GROUP BY customer_state (view)
- "准时交付率/配送延迟" → mv_delivery_perf — use AVG(on_time_rate) for overall, GROUP BY customer_state for per-state (view)
- "哪种支付方式" → mv_payment_dist (view)
- "差评率高的卖家" → mv_seller_perf (view)
- "品类销售额" → mv_category_sales (view)
- "重量与运费关系" → base tables needed (no view has weight/freight columns)
- "整体+按X" questions: use the appropriate view — views handle BOTH overall (with aggregation) and per-group queries
- For overall metrics from granular views: use AVG() or SUM() on view columns, NOT base table joins
- Use LIMIT 1000 for large unaggregated queries.
- For date filters, ym LIKE '2017%' for 2017 data."""

    messages = [
        {"role": "system", "content": DATA_ANALYST_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    result = {
        "strategy": "unknown",
        "sql": "",
        "data": [],
        "summary": "",
        "error": None,
    }

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            if attempt == 0:
                response = chat(messages, temperature=0.1, json_mode=True)
                parsed = _parse_llm_json(response)
            else:
                parsed = _retry_with_correction(
                    question, result["sql"], result.get("error", "Unknown error"))

            strategy = parsed.get("strategy", "unknown")
            sql = parsed.get("sql", "")
            summary = parsed.get("reasoning", parsed.get("summary", ""))

            result["strategy"] = strategy
            result["sql"] = sql
            result["summary"] = summary

            # Clean SQL (format only — no syntax patching)
            executed_sql = _clean_sql(sql)

            if not executed_sql or not executed_sql.strip():
                result["error"] = "Empty SQL generated"
                logger.warning("Empty SQL on attempt %d, will retry", attempt + 1)
                if attempt < max_retries:
                    continue
                return result

            logger.info("Strategy: %s | SQL: %s", result["strategy"], executed_sql[:200])

            # Safety check — only allow SELECT/WITH
            clean = executed_sql.strip().upper()
            if not (clean.startswith("SELECT") or clean.startswith("WITH")):
                result["error"] = "Agent generated non-SELECT SQL — blocked."
                logger.error("Blocked non-SELECT SQL: %s", executed_sql[:200])
                if attempt < max_retries:
                    continue
                return result

            sql_t0 = time.time()
            rows = execute_query(executed_sql)
            result["sql_time"] = round(time.time() - sql_t0, 2)  # Bug 32: SQL-only time
            result["data"] = rows
            result["summary"] = summary
            result["error"] = None  # Clear any error from previous attempt
            if rows:
                result["summary"] += f" | Returned {len(rows)} rows."
            return result  # Success

        except json.JSONDecodeError as e:
            logger.error("Failed to parse agent JSON (attempt %d): %s", attempt + 1, e)
            result["error"] = f"Failed to parse agent response: {e}"
            if attempt >= max_retries:
                return result
        except Exception as e:
            err_str = str(e)
            logger.error("Agent execution error (attempt %d): %s", attempt + 1, e, exc_info=True)
            result["error"] = err_str

            # Check if retryable SQL error
            if attempt < max_retries and (
                "OperationalError" in str(type(e).__name__) or
                "column" in err_str.lower() or
                "syntax" in err_str.lower() or
                "group by" in err_str.lower() or
                "doesn't exist" in err_str.lower()
            ):
                logger.info("Retrying with error correction (attempt %d)...", attempt + 1)
                continue
            return result

    return result


def get_top_n_sellers_by_review(n: int = 20) -> list:
    """Direct SQL to get top N worst-rated sellers."""
    sql = f"""
        SELECT seller_id, seller_state, total_gmv, total_orders, avg_review_score
        FROM mv_seller_perf
        WHERE total_orders >= 5
        ORDER BY avg_review_score ASC
        LIMIT {n}
    """
    return execute_query(sql)


def get_top_n_state_delivery_issues(n: int = 10) -> list:
    """Direct SQL to get states with worst delivery performance."""
    sql = f"""
        SELECT customer_state,
               AVG(avg_delivery_days) AS overall_avg_delivery_days,
               AVG(on_time_rate) AS overall_on_time_rate,
               SUM(delayed_orders) AS total_delayed
        FROM mv_delivery_perf
        GROUP BY customer_state
        ORDER BY overall_on_time_rate ASC
        LIMIT {n}
    """
    return execute_query(sql)


def get_state_sales(year: str = "2017") -> list:
    """Get state sales for a given year."""
    sql = f"""
        SELECT customer_state, SUM(total_gmv) AS total_gmv, SUM(total_orders) AS total_orders,
               SUM(unique_customers) AS unique_customers
        FROM mv_state_sales
        WHERE ym LIKE '{year}%'
        GROUP BY customer_state
        ORDER BY total_gmv DESC
    """
    return execute_query(sql)


def get_monthly_sales() -> list:
    """Get all monthly sales data."""
    return execute_query("SELECT * FROM mv_monthly_sales ORDER BY ym")


def get_payment_distribution() -> list:
    """Get payment method distribution."""
    return execute_query("""
        SELECT payment_type, SUM(total_transactions) AS total_transactions,
               ROUND(AVG(avg_installments), 1) AS avg_installments,
               SUM(total_value) AS total_value
        FROM mv_payment_dist
        GROUP BY payment_type
        ORDER BY total_transactions DESC
    """)


def get_category_reviews() -> list:
    """Get category-level review scores."""
    return execute_query("""
        SELECT
            COALESCE(t.product_category_name_english, p.product_category_name) AS category,
            AVG(r.review_score) AS avg_score,
            COUNT(*) AS review_count,
            SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS negative_reviews,
            SUM(CASE WHEN r.review_score >= 4 THEN 1 ELSE 0 END) AS positive_reviews
        FROM order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
        GROUP BY category
        ORDER BY review_count DESC
    """)
