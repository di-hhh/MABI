"""Streamlit Web Dashboard — Agentic BI for Olist E-Commerce Analysis."""
import streamlit as st
import pandas as pd
import os
import sys
import logging
from datetime import datetime

# Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("logs/app.log", encoding="utf-8"), logging.StreamHandler()],
)
os.makedirs("logs", exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

from agents.graph import run_query
from agents.data_analyst import (
    get_monthly_sales, get_state_sales, get_payment_distribution,
    get_top_n_sellers_by_review, get_top_n_state_delivery_issues,
)
from agents.nlp_insight import analyze_reviews
from agents.decision import what_if_remove_worst_sellers
from agents.anomaly_detector import run_all_checks
from agents.visualizer import line_chart, bar_chart, geo_heatmap, scatter_bubble, wordcloud_image, matrix_heatmap
from models.predictor import forecast_monthly
from utils.db import execute_query


def _embed_html(html_path: str, height: int = 500):
    """Embed a Plotly HTML file. Uses st.components.v1.html (deprecated but works reliably)."""
    if not html_path or not os.path.exists(html_path):
        st.warning("Chart not available.")
        return
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height, scrolling=True)


# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Agentic BI — Olist E-Commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Agentic BI")
    st.markdown("**Olist E-Commerce Analysis**")
    st.markdown("---")

    st.markdown("### 🎯 Quick Actions")
    if st.button("📈 Monthly Sales Trend"):
        st.session_state.prompt = "Show me the monthly sales trend with forecast for the next 6 weeks"
    if st.button("🗺 State Sales Map"):
        st.session_state.prompt = "Show me the sales distribution across Brazilian states"
    if st.button("💳 Payment Analysis"):
        st.session_state.prompt = "Which payment methods are most popular and what are the installment patterns?"
    if st.button("⭐ Review Analysis"):
        st.session_state.prompt = "Analyze customer review sentiment and find the top 10 worst-rated categories"
    if st.button("🚚 Delivery Performance"):
        st.session_state.prompt = "Which states have the worst delivery on-time rates?"
    if st.button("🛡 Anomaly Scan"):
        st.session_state.prompt = "Scan for any anomalies in recent data"

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.caption(f"LLM: {os.getenv('DEEPSEEK_V4_FLASH', 'deepseek-v4-pro')}")
    st.caption(f"DB: {os.getenv('MYSQL_DATABASE', 'N/A')}")

    st.markdown("---")
    st.caption("Agentic BI Final Project | Built with LangGraph + Streamlit")

# ── Initialize session state ──────────────────────────────
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Main layout ──────────────────────────────────────────
st.title("🤖 Agentic BI — Olist E-Commerce Intelligence")

# Two-column layout
left_col, right_col = st.columns([0.45, 0.55])

with left_col:
    st.markdown("### 💬 Ask an Analysis Question")

    # Display chat history
    chat_container = st.container(height=400)
    with chat_container:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(entry["response"])
                meta_parts = []
                if entry.get("charts"):
                    meta_parts.append(f"{len(entry['charts'])} charts")
                if entry.get("query_strategy"):
                    strategy_label = "View" if entry["query_strategy"] == "view" else "Base Table"
                    meta_parts.append(f"Strategy: {strategy_label}")
                if entry.get("query_time"):
                    meta_parts.append(f"Time: {entry['query_time']}s")
                if meta_parts:
                    st.caption(" | ".join(meta_parts))

    # Input box
    prompt = st.chat_input("Type your analysis question...", key="main_input")

    if prompt:
        st.session_state.prompt = prompt
        st.session_state.pending_question = prompt  # Show immediately
    elif st.session_state.prompt:
        prompt = st.session_state.prompt
        st.session_state.prompt = ""

    if prompt or st.session_state.pending_question:
        current_q = prompt or st.session_state.pending_question

        if st.session_state.pending_question and not prompt:
            # Show user question immediately before analysis starts
            with st.chat_message("user"):
                st.markdown(st.session_state.pending_question)

        with st.spinner("🔄 Analyzing... this may take 30-60 seconds"):
            try:
                result = run_query(current_q, thread_id=st.session_state.thread_id)

                response_text = result.get("final_response", "No response generated.")
                charts = result.get("charts", [])
                query_time = result.get("query_time_seconds", 0)
                query_strategy = result.get("query_strategy", "unknown")

                st.session_state.chat_history.append({
                    "question": current_q,
                    "response": response_text,
                    "charts": charts,
                    "query_strategy": query_strategy,
                    "query_time": query_time,
                })
                st.session_state.last_analysis = result
                st.session_state.pending_question = None

                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logging.error("Analysis failed for prompt: %s | Error: %s", current_q, e, exc_info=True)
                st.session_state.pending_question = None

    # Show "no history yet" message
    if not st.session_state.chat_history and not st.session_state.pending_question:
        st.info("👆 Ask a question above or click a Quick Action in the sidebar to get started!")

with right_col:
    st.markdown("### 📈 Visualizations & Results")

    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]

        # Show query info banner
        strategy_label = "Pre-Aggregation View" if latest.get("query_strategy") == "view" else "Base Table"
        query_time = latest.get("query_time", 0)
        st.info(f"Query Strategy: **{strategy_label}** | Total Time: **{query_time}s**")

        # Show charts
        tabs = st.tabs(["📊 Charts", "📋 Analysis", "🔍 Raw Data"])

        with tabs[0]:
            charts = latest.get("charts", [])
            if charts:
                for i, chart in enumerate(charts):
                    if chart.get("html"):
                        _embed_html(chart["html"], height=500)
                    st.caption(f"**{chart.get('title', f'Chart {i+1}')}**")
            else:
                st.info("No charts generated for this query.")

        with tabs[1]:
            st.markdown(latest.get("response", "No analysis text available."))

        with tabs[2]:
            result = st.session_state.last_analysis
            if result:
                if result.get("data_results"):
                    for r in result["data_results"]:
                        if r.get("data"):
                            st.dataframe(pd.DataFrame(r["data"]), use_container_width=True)
                            st.caption(f"Strategy: {r.get('strategy', 'N/A')}")
                elif result.get("data_summary"):
                    st.text(result["data_summary"])
    else:
        # Show default dashboard overview when no query yet
        st.markdown("#### 📊 Dashboard Overview")
        try:
            monthly = get_monthly_sales()
            if monthly:
                fig_data = line_chart(
                    monthly, x_col="ym",
                    y_cols=["total_gmv", "total_orders"],
                    title="Monthly Sales Overview",
                )
                if fig_data.get("html"):
                    _embed_html(fig_data["html"], height=450)
        except Exception as e:
            st.warning(f"Could not load overview chart: {e}")

        # Quick stats
        st.markdown("#### Key Metrics")
        col_a, col_b, col_c = st.columns(3)
        try:
            stats = execute_query("""
                SELECT
                    SUM(total_orders) AS total_orders,
                    ROUND(SUM(total_gmv), 0) AS total_gmv,
                    ROUND(AVG(avg_basket), 0) AS avg_basket
                FROM mv_monthly_sales
            """)
            if stats:
                s = stats[0]
                col_a.metric("Total Orders", f"{s['total_orders']:,}")
                col_b.metric("Total GMV", f"R$ {s['total_gmv']:,.0f}")
                col_c.metric("Avg Basket", f"R$ {s['avg_basket']:,.0f}")
        except Exception:
            pass

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Python, Streamlit, LangGraph, MySQL | Agentic BI Final Project")
