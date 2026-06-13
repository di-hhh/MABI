"""Decision Intelligence Agent — synthesize analysis into business recommendations."""
import logging
from config.prompts import DECISION_SYSTEM
from utils.llm import chat

logger = logging.getLogger(__name__)


def generate_recommendations(
    analysis_summary: str,
    nlp_results: dict = None,
    forecast_summary: str = None,
    data_tables: str = "",
    scenario_results: dict = None,
    memory_context: str = "",
) -> str:
    """Generate prioritized business recommendations.

    Args:
        analysis_summary: Text summary from data analyst findings.
        nlp_results: Optional NLP analysis output.
        forecast_summary: Optional forecast summary text.
        data_tables: Optional markdown tables with actual query results.
        scenario_results: Optional What-if/anomaly outputs from Scenario Agent.
        memory_context: Optional prior-turn context for follow-up analysis.

    Returns:
        Structured recommendation text.
    """
    context_parts = [f"## Data Analysis Summary\n{analysis_summary}"]

    if nlp_results:
        negative_themes = nlp_results.get("negative_themes", [])
        positive_themes = nlp_results.get("positive_themes", [])
        context_parts.append(f"""
## NLP / Review Insights
- Average review score: {nlp_results.get('avg_score', 'N/A')}/5
- Positive reviews: {nlp_results.get('positive_pct', 'N/A')}%
- Negative reviews: {nlp_results.get('negative_pct', 'N/A')}%
- Top negative keywords: {', '.join(nlp_results.get('top_negative_keywords', []))}
- Top negative categories: {nlp_results.get('top_negative_categories', [])}
- Negative review themes: {negative_themes}
- Positive review themes: {positive_themes}
- Topic modeling summary: {nlp_results.get('topic_modeling_summary', '')}
""")

    if forecast_summary:
        context_parts.append(f"## Forecast Summary\n{forecast_summary}")

    if scenario_results and scenario_results.get("summary"):
        context_parts.append(f"## Scenario Agent Results\n{scenario_results}")

    if memory_context:
        context_parts.append(f"## Conversation Memory\n{memory_context}")

    if data_tables:
        context_parts.append(f"## Actual Query Results (use these specific numbers in your recommendations)\n{data_tables}")

    context = "\n\n".join(context_parts)

    user_prompt = f"""Based on the following analysis results, generate three prioritized business recommendations for the Olist platform management.

{context}

Please structure your response as:

### Executive Summary
(2-3 sentence overview)

### Key Findings
- Finding 1 (with specific data)
- Finding 2 (with specific data)
- ...

### Priority 1: [Action Title]
**What:** ...
**Why:** ...
**Expected Impact:** ...

### Priority 2: [Action Title]
**What:** ...
**Why:** ...
**Expected Impact:** ...

### Priority 3: [Action Title]
**What:** ...
**Why:** ...
**Expected Impact:** ...

### Risks & Caveats
- ...
"""

    messages = [
        {"role": "system", "content": DECISION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = chat(messages, temperature=0.5, max_tokens=2048)
        return response
    except Exception as e:
        logger.error("Decision agent error: %s", e, exc_info=True)
        return f"Failed to generate recommendations: {e}"


def what_if_remove_worst_sellers(top_n: int = 20) -> dict:
    """What-if: calculate impact of removing top N worst-rated sellers."""
    from utils.db import execute_query

    # Get current platform average
    current = execute_query("""
        SELECT AVG(review_score) AS avg_score, COUNT(DISTINCT review_id) AS review_count
        FROM order_reviews
        WHERE review_score IS NOT NULL
    """)
    current_avg = float(current[0]["avg_score"]) if current else 0
    current_review_count = int(current[0]["review_count"]) if current else 0

    platform = execute_query("""
        SELECT
            SUM(total_gmv) AS total_gmv,
            SUM(total_orders) AS total_orders
        FROM mv_seller_perf
    """)
    platform_gmv = float(platform[0]["total_gmv"] or 0) if platform else 0
    platform_orders = int(platform[0]["total_orders"] or 0) if platform else 0

    # Get top N worst sellers from mv_seller_perf at seller grain.
    worst_sellers = execute_query(f"""
        SELECT
            seller_id,
            MAX(seller_state) AS seller_state,
            ROUND(SUM(total_gmv), 2) AS total_gmv,
            SUM(total_orders) AS total_orders,
            ROUND(
                SUM(avg_review_score * total_orders) / NULLIF(SUM(total_orders), 0),
                2
            ) AS avg_review_score
        FROM mv_seller_perf
        WHERE avg_review_score IS NOT NULL
        GROUP BY seller_id
        HAVING SUM(total_orders) >= 5
        ORDER BY avg_review_score ASC
        LIMIT {top_n}
    """)

    if not worst_sellers:
        return {"error": "No seller review data available"}

    # Calculate what-if: remove reviews from these sellers
    worst_ids = [s["seller_id"] for s in worst_sellers]
    placeholders = ", ".join([f"'{sid}'" for sid in worst_ids])

    remaining = execute_query(f"""
        SELECT AVG(review_score) AS avg_score, COUNT(*) AS cnt
        FROM (
            SELECT DISTINCT r.review_id, r.review_score
            FROM order_reviews r
            WHERE r.review_score IS NOT NULL
              AND r.review_id NOT IN (
                SELECT DISTINCT r2.review_id
                FROM order_reviews r2
                JOIN orders o2 ON r2.order_id = o2.order_id
                JOIN order_items oi2 ON o2.order_id = oi2.order_id
                WHERE oi2.seller_id IN ({placeholders})
              )
        ) remaining_reviews
    """)
    new_avg = float(remaining[0]["avg_score"]) if remaining and remaining[0]["avg_score"] else current_avg
    improvement = new_avg - current_avg

    impacted_gmv = sum(float(s.get("total_gmv") or 0) for s in worst_sellers)
    impacted_orders = sum(int(s.get("total_orders") or 0) for s in worst_sellers)

    return {
        "current_avg_score": round(current_avg, 3),
        "new_avg_score": round(new_avg, 3),
        "improvement": round(improvement, 3),
        "improvement_pct": round(improvement / current_avg * 100, 1) if current_avg else 0,
        "removed_sellers_count": len(worst_ids),
        "current_reviews_count": current_review_count,
        "remaining_reviews_count": remaining[0]["cnt"] if remaining else 0,
        "impacted_gmv": round(impacted_gmv, 2),
        "impacted_orders": impacted_orders,
        "impacted_gmv_share_pct": round(impacted_gmv / platform_gmv * 100, 2) if platform_gmv else 0,
        "impacted_order_share_pct": round(impacted_orders / platform_orders * 100, 2) if platform_orders else 0,
        "worst_sellers": worst_sellers[:5],
        "business_interpretation": (
            "Use this as a seller-screening simulation, not an automatic removal rule: "
            "the score lift must be weighed against the GMV and order share at risk."
        ),
    }
