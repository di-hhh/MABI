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


_counter = 0

def _save(fig, filename: str):
    """Save plotly fig to PNG and HTML, return relative paths."""
    global _counter
    _counter += 1
    png_path = os.path.join(OUTPUT_DIR, f"{_counter:02d}_{filename}.png")
    html_path = os.path.join(OUTPUT_DIR, f"{_counter:02d}_{filename}.html")
    fig.write_image(png_path, width=1200, height=600, scale=2)
    fig.write_html(html_path)
    logger.info("Saved chart: %s", png_path)
    return {"png": png_path, "html": html_path}


def line_chart(data: list, x_col: str, y_cols: list, title: str,
               forecast_data: list = None, forecast_x_col: str = None,
               forecast_y_col: str = "yhat", forecast_lower: str = "yhat_lower",
               forecast_upper: str = "yhat_upper", secondary_y: str = "total_orders") -> dict:
    """Time series line chart, optionally with forecast overlay.
    Uses dual y-axis when secondary_y column is present.
    """
    df = pd.DataFrame(data)
    fig = go.Figure()

    # Convert ym (e.g. "2017-01") to datetime for proper alignment with forecast
    x_vals = df[x_col]
    if x_col == "ym" or (len(df) > 0 and isinstance(df[x_col].iloc[0], str) and len(str(df[x_col].iloc[0])) == 7 and str(df[x_col].iloc[0])[4] == "-"):
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

    if forecast_data:
        fdf = pd.DataFrame(forecast_data)
        f_x = pd.to_datetime(fdf[forecast_x_col])
        fig.add_trace(go.Scatter(
            x=f_x, y=fdf[forecast_y_col],
            mode="lines", name="Forecast", line=dict(dash="dash", color="red"),
        ))
        if forecast_lower in fdf.columns and forecast_upper in fdf.columns:
            fig.add_trace(go.Scatter(
                x=list(f_x) + list(f_x[::-1]),
                y=list(fdf[forecast_upper]) + list(fdf[forecast_lower][::-1]),
                fill="toself", fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval",
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
    """Brazil state-level choropleth map."""
    df = pd.DataFrame(data)

    # Aggregate by state
    state_agg = df.groupby(state_col)[value_col].sum().reset_index()

    # Brazil states GeoJSON URL (codeforamerica)
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
        )
    except Exception:
        # Fallback: use scatter_geo with smaller dots (better than nothing)
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
                projection_scale=2.5,
            )
        except Exception:
            fig = px.bar(state_agg, x=state_col, y=value_col, title=title,
                         template="plotly_white")

    fig.update_layout(template="plotly_white")
    return _save(fig, "geo_heatmap")


def matrix_heatmap(data: list, x_col: str, y_col: str, value_col: str, title: str) -> dict:
    """Matrix/pivot heatmap."""
    df = pd.DataFrame(data)
    pivot = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc="mean")
    fig = px.imshow(pivot, title=title, aspect="auto", template="plotly_white",
                    color_continuous_scale="RdBu_r")
    fig.update_layout(template="plotly_white")
    return _save(fig, "matrix_heatmap")


def scatter_bubble(data: list, x_col: str, y_col: str, title: str,
                   size_col: str = None, color_col: str = None) -> dict:
    """Scatter / bubble chart."""
    df = pd.DataFrame(data)
    fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                     title=title, template="plotly_white", opacity=0.6)
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
    path = os.path.join(OUTPUT_DIR, f"{_counter:02d}_wordcloud.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved wordcloud: %s", path)
    return {"png": path, "html": None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test
    data = [
        {"state": "SP", "sales": 1000},
        {"state": "RJ", "sales": 600},
        {"state": "MG", "sales": 400},
    ]
    bar_chart(data, "state", "sales", "Test Bar Chart")
    print("Test chart saved.")
