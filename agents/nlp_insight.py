"""NLP / Review Insight Agent — sentiment analysis on customer reviews."""
import logging
import re
from collections import Counter
from textblob import TextBlob
from utils.db import execute_query

logger = logging.getLogger(__name__)

# Brazilian Portuguese stopwords + common words
STOPWORDS_PT = set("""
a o e de da do das dos para por em com uma um que se nao no na os as aos
foi sua seu ele ela mais ja muito muito foi assim entre esse essa estao
porque produto entrega recebi chegou compra comprar prazo antes depois
bom boa bem ruim otimo excelente pessimo recomendo super veio comprei
""".strip().split())


def get_review_data(limit: int = 8000) -> list:
    """Fetch review data joined with product categories (sampled for performance).

    Uses a subquery to sample reviews before the expensive multi-table JOIN.
    """
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


def analyze_sentiment(text: str) -> dict:
    """Run TextBlob sentiment on a single text.

    TextBlob has limitations with Portuguese, but provides a baseline.
    Returns polarity [-1, 1] and subjectivity [0, 1].
    """
    try:
        blob = TextBlob(text)
        return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}
    except Exception:
        return {"polarity": 0.0, "subjectivity": 0.0}


def analyze_reviews() -> dict:
    """Full review analysis: score distribution, keywords, sentiment."""
    logger.info("Fetching review data for NLP analysis...")
    reviews = get_review_data()
    logger.info("Fetched %d reviews", len(reviews))

    if not reviews:
        return {"error": "No review data available"}

    scores = [r["review_score"] for r in reviews]
    avg_score = sum(scores) / len(scores)

    positive_reviews = [
        r for r in reviews
        if r["review_score"] is not None and r["review_score"] >= 4
    ]
    negative_reviews = [
        r for r in reviews
        if r["review_score"] is not None and r["review_score"] <= 2
    ]

    pos_pct = len(positive_reviews) / len(reviews) * 100
    neg_pct = len(negative_reviews) / len(reviews) * 100

    # Extract keywords from review messages
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

    pos_keywords = [w for w, _ in pos_counter.most_common(15)]
    neg_keywords = [w for w, _ in neg_counter.most_common(15)]

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
        "top_positive_keywords": pos_keywords,
        "top_negative_keywords": neg_keywords,
        "top_negative_categories": top_negative_cats,
        "category_sentiment": cat_sentiment,
        "sentiment_summary": (
            f"Average review score: {avg_score:.1f}/5. "
            f"{pos_pct:.0f}% positive (4-5★), {neg_pct:.0f}% negative (1-2★). "
            f"Top negative keywords: {', '.join(neg_keywords[:5])}."
        ),
    }
