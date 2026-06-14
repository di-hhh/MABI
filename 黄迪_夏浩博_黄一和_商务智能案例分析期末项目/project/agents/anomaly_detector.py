"""Anomaly Detection Agent — scan for statistical anomalies in recent data."""
import logging
import numpy as np
import pandas as pd
from utils.db import execute_query

logger = logging.getLogger(__name__)


def detect_sales_anomalies(z_threshold: float = 2.0) -> list:
    """Detect months with anomalous GMV (z-score based)."""
    rows = execute_query("""
        SELECT ym, total_gmv, total_orders
        FROM mv_monthly_sales
        ORDER BY ym
    """)
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["total_gmv"] = df["total_gmv"].astype(float)
    gmv_mean = df["total_gmv"].mean()
    gmv_std = df["total_gmv"].std()

    anomalies = []
    for _, row in df.iterrows():
        z_score = (row["total_gmv"] - gmv_mean) / gmv_std if gmv_std > 0 else 0
        if abs(z_score) > z_threshold:
            severity = "high" if abs(z_score) >= 3 else "medium"
            anomalies.append({
                "type": "gmv_anomaly",
                "severity": severity,
                "ym": row["ym"],
                "value": row["total_gmv"],
                "z_score": round(z_score, 2),
                "direction": "high" if z_score > 0 else "low",
                "detail": f"Month {row['ym']} GMV is {abs(z_score):.1f} std deviations {'above' if z_score > 0 else 'below'} mean.",
                "action_hint": "Check campaign, seasonality, data completeness, and category/state contributors for this month.",
            })

    return anomalies


def detect_state_sales_drop(threshold_pct: float = 30.0) -> list:
    """Detect states with recent sales drops compared to their average."""
    rows = execute_query("""
        SELECT customer_state, ym, total_gmv
        FROM mv_state_sales
        ORDER BY customer_state, ym
    """)
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["total_gmv"] = df["total_gmv"].astype(float)
    anomalies = []

    for state, group in df.groupby("customer_state"):
        if len(group) < 3:
            continue
        group = group.sort_values("ym")
        recent = group.iloc[-1]
        historical_avg = group.iloc[:-1]["total_gmv"].mean()
        if historical_avg > 0:
            drop_pct = (historical_avg - recent["total_gmv"]) / historical_avg * 100
            if drop_pct > threshold_pct:
                severity = "high" if drop_pct >= 50 else "medium"
                anomalies.append({
                    "type": "state_sales_drop",
                    "severity": severity,
                    "state": state,
                    "ym": recent["ym"],
                    "recent_gmv": recent["total_gmv"],
                    "avg_gmv": round(historical_avg, 2),
                    "drop_pct": round(drop_pct, 1),
                    "detail": f"State {state} GMV dropped {drop_pct:.0f}% in {recent['ym']} compared to its historical average.",
                    "action_hint": "Inspect recent orders, seller availability, delivery delays, and marketing exposure in this state.",
                })

    return anomalies


def detect_review_score_drop(threshold: float = 0.5) -> list:
    """Detect categories with recent review score drops."""
    rows = execute_query("""
        SELECT
            COALESCE(t.product_category_name_english, p.product_category_name) AS category,
            DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS ym,
            AVG(r.review_score) AS avg_score,
            COUNT(*) AS review_count
        FROM order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
        GROUP BY category, ym
        HAVING review_count >= 10
        ORDER BY category, ym
    """)
    if not rows:
        return []

    df = pd.DataFrame(rows)
    anomalies = []

    for category, group in df.groupby("category"):
        if len(group) < 3:
            continue
        group = group.sort_values("ym")
        recent = group.iloc[-1]
        historical_avg = float(group.iloc[:-1]["avg_score"].mean())
        drop = historical_avg - float(recent["avg_score"])
        if drop > threshold:
            severity = "high" if drop >= 1.0 else "medium"
            anomalies.append({
                "type": "review_score_drop",
                "severity": severity,
                "category": category,
                "ym": recent["ym"],
                "recent_score": round(float(recent["avg_score"]), 2),
                "historical_avg": round(float(historical_avg), 2),
                "drop": round(float(drop), 2),
                "detail": f"Category '{category}' review score dropped by {drop:.1f} in {recent['ym']}.",
                "action_hint": "Review recent complaints, sellers, delivery issues, and product quality for this category.",
            })

    return anomalies


def run_all_checks() -> dict:
    """Run all anomaly detection checks.

    Returns:
        {"alerts": [...], "summary": str}
    """
    logger.info("Running anomaly detection checks...")
    alerts = []

    alerts.extend(detect_sales_anomalies())
    alerts.extend(detect_state_sales_drop())
    alerts.extend(detect_review_score_drop())

    if alerts:
        summary = f"Detected {len(alerts)} anomalies across sales, state performance, and review scores."
    else:
        summary = "No significant anomalies detected. All metrics within normal range."

    return {
        "alerts": alerts,
        "summary": summary,
        "alert_count": len(alerts),
        "alert_types": list(set(a["type"] for a in alerts)),
    }
