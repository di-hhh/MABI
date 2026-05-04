"""Data Analyst Agent — NL → SQL with view-first strategy."""
import json
import re
import logging
import pandas as pd
from config.prompts import DATA_ANALYST_SYSTEM
from config.data_dict import BASE_TABLES, MV_VIEWS
from utils.llm import chat
from utils.db import execute_query

logger = logging.getLogger(__name__)


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
    """Extract clean SQL from LLM output that may contain markdown fences."""
    text = text.strip()
    # Remove markdown code fences
    m = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Remove leading SQL comments (lines starting with --)
    text = re.sub(r"^--.*$", "", text, flags=re.MULTILINE).strip()
    # Keep only the first SQL statement (before any semicolon at end of line)
    semi_idx = text.find(";")
    if semi_idx >= 0:
        # Check if there's substantive SQL after the semicolon
        after = text[semi_idx+1:].strip()
        if after and (after.upper().startswith("SELECT") or after.upper().startswith("WITH")):
            # Multiple statements — keep only first
            text = text[:semi_idx].strip()
        else:
            # Just a trailing semicolon — remove it
            text = text[:semi_idx].strip()
    return text


def _fix_mysql_syntax(sql: str) -> str:
    """Fix common PostgreSQL-isms and LLM typos that break MySQL."""
    # Replace ::float, ::int, ::decimal casts with MySQL CAST
    sql = re.sub(r"(\w+)::float\b", r"CAST(\1 AS DECIMAL(10,2))", sql)
    sql = re.sub(r"(\w+)::int\b", r"CAST(\1 AS SIGNED)", sql)
    sql = re.sub(r"(\w+)::decimal\b", r"CAST(\1 AS DECIMAL(10,2))", sql)
    # Replace TO_CHAR with DATE_FORMAT
    sql = re.sub(r"TO_CHAR\s*\((.+?),\s*'(.+?)'\)", r"DATE_FORMAT(\1, '\2')", sql, flags=re.IGNORECASE)
    # Fix common LLM typo: product_category_english → product_category_name_english
    sql = re.sub(r"\bproduct_category_english\b", "product_category_name_english", sql)
    sql = re.sub(r"\bproduct_category_name_english_english\b", "product_category_name_english", sql)
    return sql


def analyze(question: str) -> dict:
    """Convert a natural language question to SQL and execute it.

    Returns: {"strategy": str, "sql": str, "data": list[dict], "summary": str, "error": str|None}
    """
    data_dict = _build_data_dict_prompt()
    user_prompt = f"""Data Dictionary:
{data_dict}

User Question: {question}

Respond with a JSON object containing: strategy, reasoning, sql, summary.

IMPORTANT:
- First check if ANY pre-aggregation view (mv_*) can answer this question.
- Only use base tables if views cannot satisfy the required dimensions.
- Use LIMIT 1000 for large unaggregated queries.
- For date filters, patterns like '2017' map to ym LIKE '2017%'."""

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

    try:
        response = chat(messages, temperature=0.1)

        # Parse JSON from response
        json_str = response.strip()
        m = re.search(r"\{.*\}", json_str, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
        else:
            parsed = json.loads(json_str)

        result["strategy"] = parsed.get("strategy", "unknown")
        result["sql"] = parsed.get("sql", "")
        result["summary"] = parsed.get("reasoning", parsed.get("summary", ""))
        executed_sql = _fix_mysql_syntax(_clean_sql(parsed.get("sql", "")))

        logger.info("Strategy: %s | SQL: %s", result["strategy"], executed_sql[:200])

        # Safety check — only allow SELECT and CTEs (WITH ... SELECT)
        clean = executed_sql.strip().upper()
        if not (clean.startswith("SELECT") or clean.startswith("WITH")):
            result["error"] = "Agent generated non-SELECT SQL — blocked."
            logger.error("Blocked non-SELECT SQL: %s", executed_sql[:200])
            return result

        rows = execute_query(executed_sql)
        result["data"] = rows
        result["summary"] = parsed.get("reasoning", parsed.get("summary", ""))

        if rows:
            result["summary"] += f" | Returned {len(rows)} rows."

    except json.JSONDecodeError as e:
        logger.error("Failed to parse agent JSON: %s", e)
        result["error"] = f"Failed to parse agent response: {e}"
    except Exception as e:
        logger.error("Agent execution error: %s", e, exc_info=True)
        result["error"] = str(e)

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
