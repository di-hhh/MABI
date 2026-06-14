"""Regenerate report evidence directly from the configured project database."""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "submit_version" / "project" / ".env", override=False)

from agents.nlp_insight import THEME_RULES, _normalize_text, _tokenize, get_review_data
from models.predictor import forecast_monthly, get_monthly_sales_data
from utils.db import execute_query


def save_wordcloud() -> None:
    reviews = get_review_data(limit=8000)
    positive = Counter()
    negative = Counter()
    for review in reviews:
        words = _tokenize(str(review.get("review_comment_message") or ""))
        if review.get("review_score", 0) >= 4:
            positive.update(words)
        elif review.get("review_score", 5) <= 2:
            negative.update(words)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, frequencies, title, cmap in [
        (axes[0], positive, "Positive Review Keywords", "Greens"),
        (axes[1], negative, "Negative Review Keywords", "Reds"),
    ]:
        wc = WordCloud(width=720, height=440, background_color="white", colormap=cmap, max_words=55)
        wc.generate_from_frequencies(dict(frequencies.most_common(100)))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=16, pad=12)
        ax.axis("off")
    fig.suptitle("Olist Review Text Themes", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "review_wordcloud.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def primary_theme(texts: list[str]) -> str:
    counts = Counter()
    for text in texts:
        normalized = _normalize_text(text)
        for rule in THEME_RULES:
            if any(keyword in normalized for keyword in rule["keywords"]):
                counts[rule["theme"]] += 1
    return counts.most_common(1)[0][0] if counts else "Other / insufficient text"


def save_negative_categories() -> None:
    rows = execute_query("""
        SELECT COALESCE(t.product_category_name_english, p.product_category_name) AS category,
               ROUND(AVG(r.review_score), 2) AS avg_score,
               COUNT(*) AS negative_reviews
        FROM order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN product_category_name_translation t
          ON p.product_category_name = t.product_category_name
        WHERE r.review_score <= 2
        GROUP BY category
        HAVING COUNT(*) >= 10
        ORDER BY negative_reviews DESC
        LIMIT 10
    """)
    reviews = get_review_data(limit=8000)
    messages = defaultdict(list)
    for review in reviews:
        if review.get("review_score", 5) <= 2 and review.get("review_comment_message"):
            messages[str(review.get("category"))].append(str(review["review_comment_message"]))
    for row in rows:
        row["primary_reason"] = primary_theme(messages.get(str(row["category"]), []))

    df = pd.DataFrame(rows)
    df.to_csv(ASSETS / "top10_negative_categories.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axis("off")
    display = df.rename(columns={
        "category": "Category",
        "avg_score": "Avg score",
        "negative_reviews": "Negative reviews",
        "primary_reason": "Primary negative theme",
    })
    table = ax.table(cellText=display.values, colLabels=display.columns, cellLoc="left", colLoc="left",
                     colWidths=[0.27, 0.11, 0.14, 0.48], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D2C7")
        if row == 0:
            cell.set_facecolor("#E7F1EF")
            cell.set_text_props(weight="bold", color="#155E63")
    ax.set_title("Top 10 Categories by Negative Review Volume and Primary Cause", fontsize=16,
                 fontweight="bold", pad=18)
    fig.savefig(ASSETS / "top10_negative_categories.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_delivery_geography() -> None:
    rows = execute_query("""
        SELECT c.customer_state, s.seller_state,
               CASE WHEN c.customer_state = s.seller_state THEN 'Same state' ELSE 'Cross state' END AS route_type,
               COUNT(DISTINCT o.order_id) AS total_orders,
               ROUND(AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)), 1) AS avg_delivery_days,
               ROUND(AVG(CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date
                              THEN 1 ELSE 0 END) * 100, 1) AS on_time_rate,
               SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS delayed_orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN sellers s ON oi.seller_id = s.seller_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
        GROUP BY c.customer_state, s.seller_state, route_type
        HAVING COUNT(DISTINCT o.order_id) >= 20
        ORDER BY avg_delivery_days DESC
        LIMIT 15
    """)
    df = pd.DataFrame(rows)
    df.to_csv(ASSETS / "delivery_geography_diagnostic.csv", index=False, encoding="utf-8-sig")
    df["route"] = df["seller_state"] + " to " + df["customer_state"]
    plot_df = df.sort_values("avg_delivery_days")
    colors = ["#B7672B" if value == "Cross state" else "#155E63" for value in plot_df["route_type"]]
    fig, (ax, table_ax) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax.barh(plot_df["route"], plot_df["avg_delivery_days"], color=colors)
    ax.set_xlabel("Average delivery days")
    ax.set_ylabel("Seller state to customer state")
    ax.set_title("Slowest High-volume Delivery Routes")
    ax.grid(axis="x", alpha=0.2)
    table_ax.axis("off")
    summary = df[["seller_state", "customer_state", "total_orders", "avg_delivery_days", "on_time_rate"]].head(10)
    table = table_ax.table(cellText=summary.values,
                           colLabels=["Seller", "Customer", "Orders", "Avg days", "On-time %"],
                           cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D2C7")
        if row == 0:
            cell.set_facecolor("#E7F1EF")
            cell.set_text_props(weight="bold", color="#155E63")
    table_ax.set_title("Route-level Diagnostic Evidence", fontsize=13, fontweight="bold", pad=10)
    fig.suptitle("Delivery Delay by Customer and Seller Geography", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "delivery_geography_diagnostic.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_payment_installment_heatmap() -> None:
    rows = execute_query("""
        SELECT payment_type, payment_installments, COUNT(*) AS transactions
        FROM payments
        WHERE payment_installments BETWEEN 1 AND 12
        GROUP BY payment_type, payment_installments
        ORDER BY payment_type, payment_installments
    """)
    df = pd.DataFrame(rows)
    df["share_pct"] = df.groupby("payment_type")["transactions"].transform(
        lambda values: values / values.sum() * 100
    )
    pivot = df.pivot(index="payment_type", columns="payment_installments", values="share_pct")
    fig, ax = plt.subplots(figsize=(12, 5.4))
    sns.heatmap(
        pivot,
        mask=pivot.isna(),
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Share within payment type (%)"},
        ax=ax,
    )
    ax.set_title("Installment Distribution within Each Payment Type", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of installments")
    ax.set_ylabel("Payment type")
    fig.tight_layout()
    fig.savefig(ASSETS / "payment_installment_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_forecast() -> None:
    result = forecast_monthly(weeks=6)
    if result.get("error"):
        raise RuntimeError(result["error"])
    history = get_monthly_sales_data()
    forecast = pd.DataFrame(result["forecast"])
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    history_y = history["y"] / 4.345
    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    ax.plot(history["ds"], history_y, color="#155E63", linewidth=2.2,
            label="Historical average weekly GMV")
    ax.plot(forecast["ds"], forecast["yhat"], color="#B7672B", marker="o", linewidth=2.2,
            label="6-week forecast")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    color="#B7672B", alpha=0.18, label="95% confidence interval")
    ax.axvline(history["ds"].max(), color="#647176", linestyle="--", linewidth=1.2, label="Forecast start")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("GMV (BRL)")
    ax.set_xlabel("Date")
    ax.set_title("Historical Weekly GMV Baseline and Future 6-Week Forecast")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "forecast_chart.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    forecast.to_csv(ASSETS / "forecast_6_weeks.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axis("off")
    display = forecast.copy()
    display["ds"] = display["ds"].dt.strftime("%Y-%m-%d")
    display = display[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    display.columns = ["Forecast date", "Forecast GMV", "Lower bound", "Upper bound"]
    table = ax.table(cellText=display.round(2).values, colLabels=display.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D2C7")
        if row == 0:
            cell.set_facecolor("#E7F1EF")
            cell.set_text_props(weight="bold", color="#155E63")
    ax.set_title(f"Six-Week Forecast and Confidence Interval ({result['trend_direction']})",
                 fontsize=15, fontweight="bold", pad=14)
    fig.savefig(ASSETS / "confidence_interval.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    save_wordcloud()
    save_negative_categories()
    save_delivery_geography()
    save_payment_installment_heatmap()
    save_forecast()
    print("Report assets regenerated in", ASSETS)
