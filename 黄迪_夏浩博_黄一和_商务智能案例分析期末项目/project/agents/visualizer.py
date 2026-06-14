"""Visualization Agent — auto-select chart types and generate plotly/matplotlib figures."""
import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

OUTPUT_DIR = "dashboard/static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import time as _time
_counter = int(_time.time()) % 100000

def _save(fig, filename: str):
    """Save plotly fig to PNG and HTML, return relative paths."""
    global _counter
    _counter += 1
    tag = f"{_counter % 1000:03d}"
    png_path = os.path.join(OUTPUT_DIR, f"{tag}_{filename}.png")
    html_path = os.path.join(OUTPUT_DIR, f"{tag}_{filename}.html")
    fig.write_image(png_path, width=1200, height=600, scale=2)
    fig.write_html(html_path)
    logger.info("Saved chart: %s", png_path)
    return {"png": png_path, "html": html_path}


def line_chart(data: list, x_col: str, y_cols: list, title: str,
               secondary_y: str = "total_orders") -> dict:
    """Time series line chart (no forecast overlay — forecast is separate chart).
    Uses dual y-axis when secondary_y column is present.
    """
    df = pd.DataFrame(data)
    fig = go.Figure()

    # Convert ym (e.g. "2017-01") to datetime
    x_vals = df[x_col]
    if x_col == "ym" or (len(df) > 0 and isinstance(df[x_col].iloc[0], str)
                         and len(str(df[x_col].iloc[0])) == 7
                         and str(df[x_col].iloc[0])[4] == "-"):
        x_vals = pd.to_datetime(df[x_col] + "-01")
        df["_ds"] = x_vals
        effective_x = "_ds"
    else:
        effective_x = x_col

    for yc in y_cols:
        if yc in df.columns:
            use_secondary = (yc == secondary_y)
            fig.add_trace(go.Scatter(
                x=x_vals if effective_x != "_ds" else df["_ds"],
                y=df[yc], mode="lines+markers", name=yc,
                yaxis="y2" if use_secondary else "y",
            ))

    layout = dict(
        title=title, xaxis_title="Date",
        template="plotly_white", hovermode="x unified",
    )
    if secondary_y and secondary_y in df.columns:
        layout["yaxis"] = dict(title="GMV (BRL)", side="left")
        layout["yaxis2"] = dict(title="Orders", side="right", overlaying="y")

    fig.update_layout(**layout)
    return _save(fig, "line_chart")


def forecast_chart(forecast_data: list, forecast_granularity: str = "weekly",
                   title: str = "Sales Forecast") -> dict:
    """Separate forecast chart (Bug 17 + Bug 24):
    - Solid deep yellow line
    - Confidence interval shaded band
    - Dynamic x-axis tick based on granularity
    """
    fdf = pd.DataFrame(forecast_data)
    if fdf.empty:
        return {"png": "", "html": ""}

    # Parse dates: handle different ds formats (YYYY-MM-DD, YYYY-MM, YYYY)
    ds_sample = str(fdf["ds"].iloc[0])
    if len(ds_sample) == 4:
        # Annual: just use integer years
        f_x = fdf["ds"].astype(int)
        x_is_datetime = False
    elif len(ds_sample) == 7:
        f_x = pd.to_datetime(fdf["ds"] + "-01")
        x_is_datetime = True
    else:
        f_x = pd.to_datetime(fdf["ds"])
        x_is_datetime = True

    fig = go.Figure()

    # Forecast line — solid deep yellow
    fig.add_trace(go.Scatter(
        x=f_x, y=fdf["yhat"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#CC9900", width=2.5),
        marker=dict(color="#CC9900", size=6),
    ))

    # Confidence interval band
    if "yhat_lower" in fdf.columns and "yhat_upper" in fdf.columns:
        if x_is_datetime:
            band_x = list(f_x) + list(f_x[::-1])
        else:
            band_x = list(f_x) + list(f_x[::-1])
        fig.add_trace(go.Scatter(
            x=band_x,
            y=list(fdf["yhat_upper"]) + list(fdf["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(204,153,0,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval",
            hovertemplate="CI: %{y:,.0f}<extra></extra>",
        ))

    # X-axis interval: dynamic based on granularity
    if x_is_datetime:
        if forecast_granularity == "daily":
            dtick = 86400000 * 1
            tickfmt = "%m-%d"
        elif forecast_granularity == "weekly":
            dtick = 86400000 * 7
            tickfmt = "%Y-%m-%d"
        elif forecast_granularity == "monthly":
            dtick = "M1"
            tickfmt = "%Y-%m"
        elif forecast_granularity == "annual":
            dtick = "M12"
            tickfmt = "%Y"
        else:
            dtick = 86400000 * 7
            tickfmt = "%Y-%m-%d"
        fig.update_xaxes(dtick=dtick, tickformat=tickfmt)
    else:
        # Annual with integer x
        fig.update_xaxes(dtick=1)

    fig.update_layout(
        title=title,
        xaxis_title="Forecast Period",
        yaxis_title="GMV (BRL)",
        template="plotly_white",
        hovermode="x unified",
    )

    return _save(fig, "forecast_chart")


def confidence_interval_summary(forecast_data: list) -> str:
    """Generate CI summary text in +/- xxx.xxx format."""
    lines = ["### Confidence Intervals"]
    for r in forecast_data:
        ci = r.get("ci_range", "")
        ds = r.get("ds", "")[:10]
        yhat = r.get("yhat", 0)
        lines.append(f"- **{ds}**: {yhat:,.0f} BRL ({ci})")
    return "\n".join(lines)


def bar_chart(data: list, x_col: str, y_col: str, title: str,
              color_col: str = None, orientation: str = "v") -> dict:
    """Horizontal or vertical bar chart."""
    df = pd.DataFrame(data)
    if orientation == "h":
        fig = px.bar(df, y=x_col, x=y_col, title=title, color=color_col,
                     orientation="h", template="plotly_white")
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col,
                     template="plotly_white")
    fig.update_layout(template="plotly_white")
    return _save(fig, "bar_chart")


def geo_heatmap(data: list, state_col: str, value_col: str, title: str,
                loc_data: list = None) -> dict:
    """Brazil state-level choropleth map with enlarged display area (Bug 17)."""
    df = pd.DataFrame(data)

    # Aggregate by state
    state_agg = df.groupby(state_col)[value_col].sum().reset_index()

    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

    try:
        fig = px.choropleth(
            state_agg,
            geojson=geojson_url,
            locations=state_col,
            featureidkey="properties.sigla",
            color=value_col,
            title=title,
            template="plotly_white",
            color_continuous_scale="Blues",
            scope="south america",
        )
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            center=dict(lat=-14, lon=-55),
            projection_scale=4.5,  # larger map (was default ~1)
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            height=600,
        )
    except Exception:
        # Fallback: scatter_geo with larger projection
        try:
            from utils.db import execute_query
            loc_rows = execute_query("""
                SELECT geolocation_state,
                       AVG(geolocation_lat) AS lat,
                       AVG(geolocation_lng) AS lng
                FROM geolocation
                GROUP BY geolocation_state
            """)
            loc_df = pd.DataFrame(loc_rows) if loc_rows else pd.DataFrame()
            if not loc_df.empty:
                state_agg = state_agg.merge(
                    loc_df, left_on=state_col, right_on="geolocation_state", how="left"
                )
            fig = px.scatter_geo(
                state_agg.dropna(subset=["lat"]),
                lat="lat", lon="lng",
                size=value_col,
                color=value_col,
                hover_name=state_col,
                title=title + " (dot map)",
                template="plotly_white",
                color_continuous_scale="Blues",
                scope="south america",
            )
            fig.update_geos(
                center=dict(lat=-14, lon=-55),
                projection_scale=4.5,
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                height=600,
            )
        except Exception:
            fig = px.bar(state_agg, x=state_col, y=value_col, title=title,
                         template="plotly_white")

    fig.update_layout(template="plotly_white")
    return _save(fig, "geo_heatmap")


def matrix_heatmap(data: list, x_col: str, y_col: str, value_col: str, title: str) -> dict:
    """Matrix/pivot heatmap."""
    df = pd.DataFrame(data)
    pivot = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc="sum")
    fig = px.imshow(
        pivot,
        title=title,
        aspect="auto",
        template="plotly_white",
        color_continuous_scale="YlGnBu",
        text_auto=".1f" if value_col.endswith("pct") else False,
    )
    fig.update_traces(hovertemplate="%{y}<br>Installments: %{x}<br>Value: %{z:.1f}<extra></extra>")
    fig.update_layout(template="plotly_white")
    return _save(fig, "matrix_heatmap")


def scatter_bubble(data: list, x_col: str, y_col: str, title: str,
                   size_col: str = None, color_col: str = None) -> dict:
    """Scatter / bubble chart with axis range filtering (Bug 17)."""
    df = pd.DataFrame(data)

    # Filter outliers — use 95th percentile for cleaner display
    if len(df) > 10:
        x_p95 = df[x_col].quantile(0.95)
        y_p95 = df[y_col].quantile(0.95)
        df_plot = df[(df[x_col] <= x_p95) & (df[y_col] <= y_p95)].copy()
        if len(df_plot) < 10:
            df_plot = df  # fallback
    else:
        df_plot = df

    fig = px.scatter(df_plot, x=x_col, y=y_col, size=size_col, color=color_col,
                     title=title, template="plotly_white", opacity=0.6)

    # Set reasonable axis ranges based on filtered data
    x_range = [df_plot[x_col].min() * 0.9, df_plot[x_col].max() * 1.05]
    y_range = [df_plot[y_col].min() * 0.9, df_plot[y_col].max() * 1.05]
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    fig.update_layout(template="plotly_white")
    return _save(fig, "scatter_bubble")


def wordcloud_image(positive_words: list, negative_words: list, title: str = "Review Word Cloud") -> dict:
    """Generate comparison word cloud — positive vs negative reviews."""
    pos_text = " ".join(positive_words) if positive_words else "good great excellent"
    neg_text = " ".join(negative_words) if negative_words else "bad poor terrible"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    wc_pos = WordCloud(width=600, height=400, background_color="white",
                       colormap="Greens", max_words=50).generate(pos_text)
    axes[0].imshow(wc_pos, interpolation="bilinear")
    axes[0].set_title("Positive Review Keywords", fontsize=14)
    axes[0].axis("off")

    wc_neg = WordCloud(width=600, height=400, background_color="white",
                       colormap="Reds", max_words=50).generate(neg_text)
    axes[1].imshow(wc_neg, interpolation="bilinear")
    axes[1].set_title("Negative Review Keywords", fontsize=14)
    axes[1].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    global _counter
    _counter += 1
    path = os.path.join(OUTPUT_DIR, f"{_counter % 1000:03d}_wordcloud.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved wordcloud: %s", path)
    return {"png": path, "html": None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = [
        {"state": "SP", "sales": 1000},
        {"state": "RJ", "sales": 600},
        {"state": "MG", "sales": 400},
    ]
    bar_chart(data, "state", "sales", "Test Bar Chart")
    print("Test chart saved.")
