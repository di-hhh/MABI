"""Decision Intelligence Agent — synthesize analysis into business recommendations."""
import logging
from config.prompts import DECISION_SYSTEM
from utils.llm import chat

logger = logging.getLogger(__name__)


def generate_recommendations(
    analysis_summary: str,
    nlp_results: dict = None,
    forecast_summary: str = None,
) -> str:
    """Generate prioritized business recommendations.

    Args:
        analysis_summary: Text summary from data analyst findings.
        nlp_results: Optional NLP analysis output.
        forecast_summary: Optional forecast summary text.

    Returns:
        Structured recommendation text.
    """
    context_parts = [f"## Data Analysis Summary\n{analysis_summary}"]

    if nlp_results:
        context_parts.append(f"""
## NLP / Review Insights
- Average review score: {nlp_results.get('avg_score', 'N/A')}/5
- Positive reviews: {nlp_results.get('positive_pct', 'N/A')}%
- Negative reviews: {nlp_results.get('negative_pct', 'N/A')}%
- Top negative keywords: {', '.join(nlp_results.get('top_negative_keywords', []))}
- Top negative categories: {nlp_results.get('top_negative_categories', [])}
""")

    if forecast_summary:
        context_parts.append(f"## Forecast Summary\n{forecast_summary}")

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
    current = execute_query("SELECT AVG(review_score) AS avg_score FROM order_reviews")
    current_avg = float(current[0]["avg_score"]) if current else 0

    # Get top N worst sellers from mv_seller_perf (pre-aggregated)
    worst_sellers = execute_query(f"""
        SELECT seller_id, seller_state, total_gmv, total_orders, avg_review_score
        FROM mv_seller_perf
        WHERE total_orders >= 5 AND avg_review_score IS NOT NULL
        ORDER BY avg_review_score ASC
        LIMIT {top_n}
    """)

    if not worst_sellers:
        return {"error": "No seller review data available"}

    # Calculate what-if: remove reviews from these sellers
    worst_ids = [s["seller_id"] for s in worst_sellers]
    placeholders = ", ".join([f"'{sid}'" for sid in worst_ids])

    remaining = execute_query(f"""
        SELECT AVG(r.review_score) AS avg_score, COUNT(*) AS cnt
        FROM order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE oi.seller_id NOT IN ({placeholders})
    """)
    new_avg = float(remaining[0]["avg_score"]) if remaining and remaining[0]["avg_score"] else current_avg
    improvement = new_avg - current_avg

    return {
        "current_avg_score": round(current_avg, 3),
        "new_avg_score": round(new_avg, 3),
        "improvement": round(improvement, 3),
        "improvement_pct": round(improvement / current_avg * 100, 1) if current_avg else 0,
        "removed_sellers_count": len(worst_ids),
        "remaining_reviews_count": remaining[0]["cnt"] if remaining else 0,
        "worst_sellers": worst_sellers[:5],
    }
