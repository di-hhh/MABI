"""NLP / Review Insight Agent — LLM + Pydantic v2 Portuguese sentiment analysis."""
import logging
import re
from collections import Counter
from textblob import TextBlob
from utils.db import execute_query
from utils.llm import chat
from config.prompts import NLP_INSIGHT_SYSTEM
from models.llm_outputs import NLPSentimentOutput, safe_parse_pydantic

logger = logging.getLogger(__name__)

# Brazilian Portuguese stopwords
STOPWORDS_PT = set("""
a o e de da do das dos para por em com uma um que se nao no na os as aos
foi sua seu ele ela mais ja muito muito foi assim entre esse essa estao
porque produto entrega recebi chegou compra comprar prazo antes depois
bom boa bem ruim otimo excelente pessimo recomendo super veio comprei
""".strip().split())


def get_review_data(limit: int = 8000) -> list:
    """Fetch review data joined with product categories (sampled for performance)."""
    return execute_query(f"""
        SELECT r.review_score, r.review_comment_title, r.review_comment_message,
               COALESCE(t.product_category_name_english, p.product_category_name) AS category
        FROM (
            SELECT * FROM order_reviews
            WHERE review_comment_message IS NOT NULL
            LIMIT {limit}
        ) r
        JOIN orders o ON r.order_id = o.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
    """)


def _tokenize(text: str) -> list:
    """Simple tokenization for Portuguese text."""
    words = re.findall(r"\b[a-záàâãéêíóôõúç]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS_PT]


def _llm_sentiment_analysis(reviews_sample: list) -> dict:
    """Use LLM (NLP_INSIGHT_SYSTEM) for Portuguese sentiment analysis."""
    # Build a representative sample for LLM analysis (limit to avoid token overflow)
    sample_texts = []
    for r in reviews_sample[:15]:  # Reduced from 50 to 15 for JSON reliability
        msg = str(r.get("review_comment_message", ""))[:80]  # Truncate per-message
        score = r.get("review_score", 0)
        cat = r.get("category", "unknown")
        sample_texts.append(f"[s={score}|{cat}] {msg}")

    # Include diverse review scores
    negative_samples = [r for r in reviews_sample if r.get("review_score", 5) <= 2][:5]
    positive_samples = [r for r in reviews_sample if r.get("review_score", 5) >= 4][:5]
    mixed = negative_samples + positive_samples
    if len(mixed) < 5:
        mixed = reviews_sample[:15]

    sample_lines = []
    for r in mixed[:12]:
        msg = str(r.get("review_comment_message", ""))[:80]
        score = r.get("review_score", 0)
        cat = r.get("category", "unknown")
        sample_lines.append(f"[score={score}] [{cat}] {msg}")

    user_prompt = f"""Analyze the following sample of {len(sample_lines)} Brazilian Portuguese customer reviews from the Olist e-commerce platform.

Review samples:
{chr(10).join(sample_lines)}

Please provide:
1. Overall sentiment summary (in Chinese and English)
2. Top 5 positive keywords/themes with frequency
3. Top 5 negative keywords/themes with frequency
4. Key business insights from these reviews

Respond with JSON:
{{
  "sentiment_summary": "...",
  "top_positive_keywords": ["word1", "word2", ...],
  "top_negative_keywords": ["word1", "word2", ...],
  "key_insights": "..."
}}"""

    try:
        response = chat(
            [{"role": "system", "content": NLP_INSIGHT_SYSTEM},
             {"role": "user", "content": user_prompt}],
            temperature=0.2, max_tokens=2048, json_mode=True,
        )
        parsed = safe_parse_pydantic(response, NLPSentimentOutput)
        if parsed is not None:
            return parsed.model_dump()
        return {}
    except Exception as e:
        logger.warning("LLM sentiment analysis failed: %s", e)
        return {}


def analyze_reviews() -> dict:
    """Full review analysis: score distribution, keywords, sentiment (LLM + TextBlob).

    Uses NLP_INSIGHT_SYSTEM + LLM for Portuguese understanding,
    with statistical keyword extraction as fallback/enrichment.
    """
    logger.info("Fetching review data for NLP analysis...")
    reviews = get_review_data()
    logger.info("Fetched %d reviews", len(reviews))

    if not reviews:
        return {"error": "No review data available"}

    # Score statistics
    scores = [r["review_score"] for r in reviews]
    avg_score = sum(scores) / len(scores)

    positive_reviews = [r for r in reviews if r["review_score"] is not None and r["review_score"] >= 4]
    negative_reviews = [r for r in reviews if r["review_score"] is not None and r["review_score"] <= 2]

    pos_pct = len(positive_reviews) / len(reviews) * 100
    neg_pct = len(negative_reviews) / len(reviews) * 100

    # Statistical keyword extraction
    pos_words = []
    for r in positive_reviews:
        if r["review_comment_message"]:
            pos_words.extend(_tokenize(str(r["review_comment_message"])))

    neg_words = []
    for r in negative_reviews:
        if r["review_comment_message"]:
            neg_words.extend(_tokenize(str(r["review_comment_message"])))

    pos_counter = Counter(pos_words)
    neg_counter = Counter(neg_words)

    stat_pos_keywords = [w for w, _ in pos_counter.most_common(15)]
    stat_neg_keywords = [w for w, _ in neg_counter.most_common(15)]

    # LLM-based Portuguese sentiment analysis
    llm_insights = _llm_sentiment_analysis(reviews)

    # Merge LLM insights with statistical keywords
    llm_pos = llm_insights.get("top_positive_keywords", [])
    llm_neg = llm_insights.get("top_negative_keywords", [])

    # Combine and deduplicate, LLM first then statistical
    combined_pos = list(dict.fromkeys(llm_pos + stat_pos_keywords))[:15]
    combined_neg = list(dict.fromkeys(llm_neg + stat_neg_keywords))[:15]

    # Category-level sentiment
    cat_stats = {}
    for r in reviews:
        cat = r.get("category", "unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"scores": [], "count": 0}
        if r["review_score"] is not None:
            cat_stats[cat]["scores"].append(r["review_score"])
        cat_stats[cat]["count"] += 1

    cat_sentiment = []
    for cat, stats in cat_stats.items():
        if stats["count"] >= 5:
            cat_sentiment.append({
                "category": cat,
                "avg_score": round(sum(stats["scores"]) / len(stats["scores"]), 2) if stats["scores"] else 0,
                "review_count": stats["count"],
            })

    cat_sentiment.sort(key=lambda x: x["avg_score"])
    top_negative_cats = [c for c in cat_sentiment if c["avg_score"] < 3.5][:10]

    sentiment_summary = llm_insights.get("sentiment_summary", "")
    if not sentiment_summary:
        sentiment_summary = (
            f"Average review score: {avg_score:.1f}/5. "
            f"{pos_pct:.0f}% positive (4-5★), {neg_pct:.0f}% negative (1-2★). "
            f"Top negative keywords: {', '.join(stat_neg_keywords[:5])}."
        )

    return {
        "total_reviews": len(reviews),
        "avg_score": round(avg_score, 2),
        "positive_pct": round(pos_pct, 1),
        "negative_pct": round(neg_pct, 1),
        "score_distribution": {
            "5": scores.count(5),
            "4": scores.count(4),
            "3": scores.count(3),
            "2": scores.count(2),
            "1": scores.count(1),
        },
        "top_positive_keywords": combined_pos,
        "top_negative_keywords": combined_neg,
        "top_negative_categories": top_negative_cats,
        "category_sentiment": cat_sentiment,
        "sentiment_summary": sentiment_summary,
        "llm_insights": llm_insights.get("key_insights", ""),
    }
