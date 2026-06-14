"""Time Series Predictor — Prophet-based forecasting on mv_monthly_sales.

Supports dynamic forecast granularity (Bug 24):
- ≤1 week: daily
- >1 week and ≤12 weeks: weekly
- >12 weeks and ≤12 months: monthly
- >12 months: annual
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
        # The dataset contains fewer than three complete annual cycles. A
        # yearly component overfits badly at weekly forecast points, so this
        # short-horizon model uses the underlying trend only.
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df[["ds", "y"]])
    return model


def _determine_granularity(weeks: float) -> tuple:
    """Determine forecast granularity and periods based on horizon.

    Returns: (granularity: str, periods: int, freq: str)
    """
    if weeks <= 1:
        return ("daily", int(weeks * 7), "D")
    elif weeks <= 12:
        return ("weekly", int(weeks), "W")
    elif weeks <= 48:
        # Monthly: weeks / ~4.33 = number of months
        return ("monthly", max(1, round(weeks / 4.33)), "MS")
    else:
        # Annual: weeks / ~52 = number of years
        return ("annual", max(1, round(weeks / 52)), "YS")


def forecast_monthly(weeks: int = 6) -> dict:
    """Forecast sales using Prophet with dynamic granularity.

    Args:
        weeks: Forecast horizon in weeks.

    Returns:
        {
            "forecast": list[dict] — forecast rows,
            "historical": list[dict] — historical rows,
            "model_summary": str,
            "last_historical_date": str,
            "forecast_granularity": "daily"|"weekly"|"monthly"|"annual",
            "forecast_periods": int,
            "trend_direction": "up"|"down"|"flat"
        }
    """
    logger.info("Training Prophet forecast for %d weeks...", weeks)

    df = get_monthly_sales_data()
    if len(df) < 4:
        return {"error": "Not enough historical data for forecasting (need at least 4 months)"}

    granularity, periods, freq = _determine_granularity(weeks)
    logger.info("Forecast granularity: %s, periods: %d, freq: %s", granularity, periods, freq)

    # mv_monthly_sales stores monthly totals. Convert the target to the requested
    # reporting scale before fitting, otherwise weekly/daily predictions would
    # incorrectly retain a monthly-total magnitude.
    model_df = df.copy()
    if granularity == "weekly":
        model_df["y"] = model_df["y"] / 4.345
    elif granularity == "daily":
        model_df["y"] = model_df["y"] / 30.437

    model = _train_prophet(model_df)

    # Build future dataframe with appropriate frequency
    if granularity == "annual":
        # For annual, we need to extend far enough
        future_days = weeks * 7 + 30
        future = model.make_future_dataframe(periods=future_days, freq="D")
    elif granularity == "monthly":
        future = model.make_future_dataframe(periods=periods, freq="MS")
    else:
        future = model.make_future_dataframe(periods=max(periods, 1) * 7, freq="D")

    np.random.seed(42)
    forecast_df = model.predict(future)
    last_date = df["ds"].max()
    future_fc = forecast_df[forecast_df["ds"] > last_date].copy()

    # Aggregate based on granularity
    if granularity == "daily":
        grouped = future_fc.head(periods).copy()
        grouped["label"] = grouped["ds"].dt.strftime("%m-%d")
    elif granularity == "weekly":
        future_fc["week_label"] = future_fc["ds"].dt.strftime("%Y-W%V")
        grouped = future_fc.groupby("week_label", sort=False).agg(
            ds=("ds", "first"),
            yhat=("yhat", "mean"),
            yhat_lower=("yhat_lower", "mean"),
            yhat_upper=("yhat_upper", "mean"),
        ).reset_index(drop=True)
        grouped = grouped.head(periods)
    elif granularity == "monthly":
        future_fc["month_label"] = future_fc["ds"].dt.strftime("%Y-%m")
        grouped = future_fc.groupby("month_label", sort=False).agg(
            ds=("ds", "first"),
            yhat=("yhat", "mean"),
            yhat_lower=("yhat_lower", "mean"),
            yhat_upper=("yhat_upper", "mean"),
        ).reset_index(drop=True)
        grouped = grouped.head(periods)
    else:  # annual
        future_fc["year_label"] = future_fc["ds"].dt.year.astype(str)
        grouped = future_fc.groupby("year_label", sort=False).agg(
            ds=("ds", "first"),
            yhat=("yhat", "mean"),
            yhat_lower=("yhat_lower", "mean"),
            yhat_upper=("yhat_upper", "mean"),
        ).reset_index(drop=True)
        grouped = grouped.head(periods)

    # GMV cannot be negative. Prophet is unconstrained by default, so keep the
    # business-facing forecast and confidence interval within the valid domain.
    grouped["yhat"] = grouped["yhat"].clip(lower=0)
    grouped["yhat_lower"] = grouped["yhat_lower"].clip(lower=0)
    grouped["yhat_upper"] = grouped[["yhat_upper", "yhat"]].max(axis=1)

    # Build forecast rows
    forecast_rows = []
    for _, row in grouped.iterrows():
        ci_half = round((row["yhat_upper"] - row["yhat_lower"]) / 2, 2)
        if granularity == "monthly":
            ds_str = row["ds"].strftime("%Y-%m")
        elif granularity == "annual":
            ds_str = str(int(row["ds"].year))
        else:
            ds_str = row["ds"].strftime("%Y-%m-%d")
        forecast_rows.append({
            "ds": ds_str,
            "yhat": round(row["yhat"], 2),
            "yhat_lower": round(row["yhat_lower"], 2),
            "yhat_upper": round(row["yhat_upper"], 2),
            "ci_range": f"+/- {ci_half:.2f}",
        })

    # Trend direction
    recent_vals = model_df["y"].tail(3).values
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

    # Historical rows
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
        "forecast_unit": "weekly_gmv" if granularity == "weekly" else (
            "daily_gmv" if granularity == "daily" else "period_gmv"
        ),
        "forecast_periods": len(forecast_rows),
        "trend_direction": direction,
    }
