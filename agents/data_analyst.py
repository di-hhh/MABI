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
    m = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Remove leading SQL comments
    text = re.sub(r"^--.*$", "", text, flags=re.MULTILINE).strip()
    # Keep only first statement
    semi_idx = text.find(";")
    if semi_idx >= 0:
        after = text[semi_idx+1:].strip()
        if after and (after.upper().startswith("SELECT") or after.upper().startswith("WITH")):
            text = text[:semi_idx].strip()
        else:
            text = text[:semi_idx].strip()
    return text


def _fix_mysql_syntax(sql: str) -> str:
    """Fix common PostgreSQL-isms, LLM typos, and alias mistakes (Bug 20)."""
    # PostgreSQL casts
    sql = re.sub(r"(\w+)::float\b", r"CAST(\1 AS DECIMAL(10,2))", sql)
    sql = re.sub(r"(\w+)::int\b", r"CAST(\1 AS SIGNED)", sql)
    sql = re.sub(r"(\w+)::decimal\b", r"CAST(\1 AS DECIMAL(10,2))", sql)
    sql = re.sub(r"TO_CHAR\s*\((.+?),\s*'(.+?)'\)", r"DATE_FORMAT(\1, '\2')", sql, flags=re.IGNORECASE)

    # Fix column name typos
    sql = re.sub(r"\bproduct_category_english\b", "product_category_name_english", sql)
    sql = re.sub(r"\bproduct_category_name_english_english\b", "product_category_name_english", sql)

    # Fix common table alias mistakes (Bug 20)
    # orders table aliased as 'o' — NEVER has customer_state/customer_city columns
    # These columns belong to customers (usually aliased 'c') or geolocation
    # Try to find the correct alias for customers table
    cust_alias = "c"
    cust_match = re.search(r'customers\s+(?:AS\s+)?(\w+)', sql, re.IGNORECASE)
    if cust_match:
        cust_alias = cust_match.group(1)
    elif not re.search(r'\bc\b.*customers', sql, re.IGNORECASE):
        # If no clear alias, try to detect from JOIN clause
        join_match = re.search(r'JOIN\s+customers\s+(\w+)', sql, re.IGNORECASE)
        if join_match:
            cust_alias = join_match.group(1)

    # Always fix o.customer_* — orders table doesn't have these
    sql = re.sub(r'\bo\.customer_state\b', f'{cust_alias}.customer_state', sql)
    sql = re.sub(r'\bo\.customer_city\b', f'{cust_alias}.customer_city', sql)
    sql = re.sub(r'\bo\.customer_zip_code_prefix\b', f'{cust_alias}.customer_zip_code_prefix', sql)

    # Fix geolocation state refs in JOIN queries
    geo_match = re.search(r'geolocation\s+(?:AS\s+)?(\w+)', sql, re.IGNORECASE)
    if geo_match:
        geo_alias = geo_match.group(1)
        sql = re.sub(r'\bo\.geolocation_state\b', f'{geo_alias}.geolocation_state', sql)
    sql = re.sub(r'\bo\.geolocation_state\b', 'g.geolocation_state', sql)

    # Fix double-decimal issues
    sql = re.sub(r"CAST\((.+?)\s+AS\s+DECIMAL\(10,2\)\)\s+AS\s+DECIMAL", r"CAST(\1 AS DECIMAL(10,2))", sql)

    return sql


def _parse_llm_json(response: str) -> dict:
    """Robust LLM JSON parsing with multiple fallback strategies (Bug 20)."""
    # Strategy 1: Extract JSON object from response
    json_str = response.strip()
    m = re.search(r"\{.*\}", json_str, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Try parsing directly
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Clean up common LLM JSON issues
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([}\]])", r"\1", json_str)
    # Fix unquoted keys
    cleaned = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Try to extract SQL and strategy from partial JSON
    result = {"strategy": "unknown", "sql": "", "reasoning": ""}
    strat_m = re.search(r'"strategy"\s*:\s*"(\w+)"', response)
    if strat_m:
        result["strategy"] = strat_m.group(1)
    sql_m = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
    if sql_m:
        result["sql"] = sql_m.group(1).replace('\\"', '"').replace('\\n', '\n')
    return result


def _retry_with_correction(question: str, original_sql: str, error_msg: str) -> dict:
    """Retry SQL generation with error context for correction (Bug 20)."""
    data_dict = _build_data_dict_prompt()
    correction_prompt = f"""Data Dictionary:
{data_dict}

User Question: {question}

The previous SQL query FAILED with this error:
{error_msg[:500]}

Previous SQL that failed:
```sql
{original_sql}
```

Please fix the SQL. Generate a corrected version. Remember:
- This is MySQL — NO PostgreSQL syntax (::float, ::int, etc.)
- Every non-aggregated column in SELECT must be in GROUP BY
- The 'orders' table aliased as 'o' does NOT have customer_state/customer_city columns
- Use the correct table alias for each column

Respond with JSON: {{"strategy": "...", "sql": "corrected SQL", "reasoning": "..."}}"""

    messages = [
        {"role": "system", "content": DATA_ANALYST_SYSTEM},
        {"role": "user", "content": correction_prompt},
    ]

    try:
        response = chat(messages, temperature=0.0)
        parsed = _parse_llm_json(response)
        return parsed
    except Exception as e:
        logger.error("Retry correction failed: %s", e)
        return {"strategy": "error", "sql": "", "reasoning": str(e)}


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
- For date filters, patterns like '2017' map to ym LIKE '2017%'.
- The 'orders' table aliased as 'o' does NOT contain customer_state/customer_city/customer_zip_code_prefix columns.
- Customer location columns are in the 'customers' table (aliased as 'c')."""

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
                response = chat(messages, temperature=0.1)
            # attempt > 0 is handled by _retry_with_correction (already called)

            if attempt == 0:
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

            executed_sql = _fix_mysql_syntax(_clean_sql(sql))

            if not executed_sql.strip():
                result["error"] = "Empty SQL generated"
                if attempt < max_retries:
                    continue
                return result

            logger.info("Strategy: %s | SQL: %s", result["strategy"], executed_sql[:200])

            # Safety check
            clean = executed_sql.strip().upper()
            if not (clean.startswith("SELECT") or clean.startswith("WITH")):
                result["error"] = "Agent generated non-SELECT SQL — blocked."
                logger.error("Blocked non-SELECT SQL: %s", executed_sql[:200])
                if attempt < max_retries:
                    continue
                return result

            rows = execute_query(executed_sql)
            result["data"] = rows
            result["summary"] = summary
            if rows:
                result["summary"] += f" | Returned {len(rows)} rows."
            return result  # Success — exit retry loop

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
