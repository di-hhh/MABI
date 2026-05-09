"""Time Series Predictor — Prophet-based forecasting on mv_monthly_sales.

Supports dynamic forecast granularity:
- weeks=6 (default) → weekly forecast for next 6 weeks
- days=7 → daily forecast for next 7 days
- months=N → monthly forecast for next N months
"""
import logging
import pandas as pd
import numpy as np
from prophet import Prophet
from utils.db import execute_query

logger = logging.getLogger(__name__)


def get_monthly_sales_data() -> pd.DataFrame:
    """Fetch monthly sales data from pre-aggregation view."""
    rows = execute_query("""
        SELECT ym, total_gmv, total_orders, avg_basket, total_freight
        FROM mv_monthly_sales
        ORDER BY ym
    """)
    if not rows:
        raise RuntimeError("mv_monthly_sales is empty")
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ym"] + "-01")
    df["y"] = df["total_gmv"].astype(float)
    return df.sort_values("ds")


def _train_prophet(df: pd.DataFrame) -> Prophet:
    """Train a Prophet model on the given dataframe."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df[["ds", "y"]])
    return model


def _build_weekly_forecast(model: Prophet, df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Generate weekly forecast for the next N weeks."""
    # Expand historical to daily, then aggregate to weekly for forecast period
    future = model.make_future_dataframe(periods=weeks * 7, freq="D")
    forecast_df = model.predict(future)
    last_date = df["ds"].max()

    # Keep only future rows
    future_fc = forecast_df[forecast_df["ds"] > last_date].copy()
    # Group by ISO week
    future_fc["week"] = future_fc["ds"].dt.isocalendar().week.astype(int)
    future_fc["year"] = future_fc["ds"].dt.isocalendar().year.astype(int)
    future_fc["week_label"] = future_fc["ds"].dt.strftime("%Y-W%V")

    weekly = future_fc.groupby("week_label", sort=False).agg(
        ds=("ds", "first"),
        yhat=("yhat", "mean"),
        yhat_lower=("yhat_lower", "mean"),
        yhat_upper=("yhat_upper", "mean"),
    ).reset_index(drop=True)

    expected_weeks = min(weeks, len(weekly))
    return weekly.head(expected_weeks)


def forecast_monthly(weeks: int = 6) -> dict:
    """Forecast sales for the next N weeks using Prophet.

    Uses weekly aggregation for fine-grained forecast with dynamic x-axis.
    For very short forecasts (<=7 days), uses daily granularity.

    Returns:
        {
            "forecast": list[dict] — forecast rows with ds, yhat, yhat_lower, yhat_upper, ci_range,
            "historical": list[dict] — historical rows,
            "model_summary": str,
            "last_historical_date": str,
            "forecast_granularity": "weekly" | "daily",
            "trend_direction": "up" | "down" | "flat"
        }
    """
    logger.info("Training Prophet forecast for %d weeks...", weeks)

    df = get_monthly_sales_data()
    if len(df) < 4:
        return {"error": "Not enough historical data for forecasting (need at least 4 months)"}

    model = _train_prophet(df)

    # Dynamic granularity
    if weeks <= 1:
        # Daily forecast for very short horizons
        future = model.make_future_dataframe(periods=weeks * 7, freq="D")
        granularity = "daily"
    else:
        # Weekly aggregation for longer horizons
        future = model.make_future_dataframe(periods=weeks * 7, freq="D")
        granularity = "weekly"

    forecast_df = model.predict(future)
    last_date = df["ds"].max()
    future_fc = forecast_df[forecast_df["ds"] > last_date].copy()

    if granularity == "weekly":
        future_fc["week_label"] = future_fc["ds"].dt.strftime("%Y-W%V")
        grouped = future_fc.groupby("week_label", sort=False).agg(
            ds=("ds", "first"),
            yhat=("yhat", "mean"),
            yhat_lower=("yhat_lower", "mean"),
            yhat_upper=("yhat_upper", "mean"),
        ).reset_index(drop=True)
        grouped = grouped.head(weeks)
    else:
        grouped = future_fc.head(weeks * 7)

    forecast_rows = []
    for _, row in grouped.iterrows():
        ci_half = round((row["yhat_upper"] - row["yhat_lower"]) / 2, 2)
        forecast_rows.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "yhat": round(row["yhat"], 2),
            "yhat_lower": round(row["yhat_lower"], 2),
            "yhat_upper": round(row["yhat_upper"], 2),
            "ci_range": f"+/- {ci_half:.2f}",
        })

    # Trend direction
    recent_vals = df["y"].tail(3).values
    forecast_vals = grouped["yhat"].values
    if len(forecast_vals) >= 3 and len(recent_vals) >= 3:
        trend_val = (forecast_vals[-1] - recent_vals[-1]) / recent_vals[-1] * 100
    else:
        trend_val = 0

    if trend_val > 5:
        direction = "up"
    elif trend_val < -5:
        direction = "down"
    else:
        direction = "flat"

    hist_rows = []
    for _, row in df.iterrows():
        hist_rows.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "y": row["y"],
            "ym": row["ym"],
        })

    ci_values = [r["ci_range"] for r in forecast_rows]
    ci_summary = "; ".join(f"{r['ds']}: {r['ci_range']}" for r in forecast_rows[:3])

    summary = (
        f"Based on {len(df)} months of historical data, "
        f"total GMV is forecasted to be {direction}. "
        f"Next {len(forecast_rows)} {granularity} periods forecasted. "
        f"Confidence intervals: {ci_summary}"
    )

    return {
        "forecast": forecast_rows,
        "historical": hist_rows,
        "model_summary": summary,
        "last_historical_date": last_date.strftime("%Y-%m-%d"),
        "forecast_granularity": granularity,
        "trend_direction": direction,
    }
