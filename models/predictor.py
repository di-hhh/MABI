"""Time Series Predictor — Prophet-based forecasting on mv_monthly_sales."""
import logging
import pandas as pd
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


def forecast_monthly(months: int = 2) -> dict:
    """Forecast monthly sales for the next N months using Prophet
    (covers approximately 6 weeks for 2 months).

    Returns:
        {
            "forecast": list[dict] — forecast rows with date, yhat, yhat_lower, yhat_upper,
            "historical": list[dict] — historical rows,
            "model_summary": str,
            "last_historical_date": str,
            "trend_direction": "up" | "down" | "flat"
        }
    """
    logger.info("Training Prophet forecast for %d months (≈6 weeks)...", months)

    df = get_monthly_sales_data()
    if len(df) < 4:
        return {"error": "Not enough historical data for forecasting (need at least 4 months)"}

    # Fit Prophet on monthly data
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df[["ds", "y"]])

    # Generate monthly future dataframe (6 weeks ≈ 2 months)
    # Use MS (month start) to align with historical data format (YYYY-MM-01)
    future = model.make_future_dataframe(periods=months, freq="MS")
    forecast_df = model.predict(future)

    # Extract forecast period only
    last_date = df["ds"].max()
    future_forecast = forecast_df[forecast_df["ds"] > last_date]

    forecast_rows = []
    for _, row in future_forecast.iterrows():
        forecast_rows.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "yhat": round(row["yhat"], 2),
            "yhat_lower": round(row["yhat_lower"], 2),
            "yhat_upper": round(row["yhat_upper"], 2),
        })

    # Trend direction
    recent = df["y"].tail(3).values
    forecast_vals = future_forecast["yhat"].values
    if len(forecast_vals) >= 3 and len(recent) >= 3:
        trend_val = (forecast_vals[-1] - recent[-1]) / recent[-1] * 100
    else:
        trend_val = 0

    if trend_val > 5:
        direction = "up"
    elif trend_val < -5:
        direction = "down"
    else:
        direction = "flat"

    # Historical data as dicts for plotting
    hist_rows = []
    for _, row in df.iterrows():
        hist_rows.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "y": row["y"],
            "ym": row["ym"],
        })

    summary = (
        f"Based on {len(df)} months of historical data, "
        f"total GMV is forecasted to be {direction}. "
        f"Forecast range: {forecast_rows[0]['yhat_lower']:.0f} to {forecast_rows[-1]['yhat_upper']:.0f} BRL."
    )

    return {
        "forecast": forecast_rows,
        "historical": hist_rows,
        "model_summary": summary,
        "last_historical_date": last_date.strftime("%Y-%m-%d"),
        "trend_direction": direction,
    }
