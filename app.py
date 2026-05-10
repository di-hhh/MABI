"""Streamlit Web Dashboard — Agentic BI for Olist E-Commerce Analysis."""
import streamlit as st
import pandas as pd
import os
import sys
import logging
import threading
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
    """Embed a Plotly HTML file."""
    if not html_path or not os.path.exists(html_path):
        st.warning("Chart not available.")
        return
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height, scrolling=True)


def _run_base_table_comparison(question: str, result_holder: dict, stop_event: threading.Event):
    """Background thread: re-run the SAME query using base tables for performance comparison (Bug 28)."""
    import time
    try:
        from agents.data_analyst import analyze
        t0 = time.time()
        # Force base_table strategy — the LLM MUST use base table JOINs to get the same result
        forced_q = (f"IMPORTANT: You MUST use base tables (JOIN orders, order_items, etc.) "
                    f"for this query. Do NOT use any mv_* pre-aggregation views.\n\n{question}")
        result = analyze(forced_q)
        elapsed = round(time.time() - t0, 2)

        if stop_event.is_set():
            result_holder["status"] = "cancelled"
            result_holder["base_time"] = None  # Don't show partial time
            return

        if result.get("error"):
            result_holder["status"] = "error"
            result_holder["error"] = result["error"]
            return

        result_holder["status"] = "done"
        result_holder["base_time"] = elapsed
        result_holder["base_strategy"] = result.get("strategy", "error")
        result_holder["base_data"] = result.get("data", [])
    except Exception as e:
        if stop_event.is_set():
            result_holder["status"] = "cancelled"
        else:
            result_holder["status"] = "error"
            result_holder["error"] = str(e)


def _kill_comparison_thread():
    """Kill any running comparison thread before starting a new query (Bug 28)."""
    comp = st.session_state.get("comparison")
    if comp and comp["holder"]["status"] == "running":
        comp["stop_event"].set()
        comp["holder"]["status"] = "cancelled"
        # Wait briefly for thread to die
        if comp.get("thread") and comp["thread"].is_alive():
            comp["thread"].join(timeout=2)
    st.session_state.comparison = None


# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Agentic BI — Olist E-Commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
if "comparison" not in st.session_state:
    st.session_state.comparison = None

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

# ── Main layout ──────────────────────────────────────────
st.title("🤖 Agentic BI — Olist E-Commerce Intelligence")

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
                    total_str = f"Total: {entry['query_time']}s"
                    if entry.get("sql_time") and entry["sql_time"] > 0:
                        total_str += f" (SQL: {entry['sql_time']}s)"
                    meta_parts.append(total_str)
                if entry.get("base_time"):
                    meta_parts.append(f"Base table: {entry['base_time']}s")
                if meta_parts:
                    st.caption(" | ".join(meta_parts))

    # Input box
    prompt = st.chat_input("Type your analysis question...", key="main_input")

    if prompt:
        st.session_state.prompt = prompt
        st.session_state.pending_question = prompt
    elif st.session_state.prompt:
        prompt = st.session_state.prompt
        st.session_state.prompt = ""

    if prompt or st.session_state.pending_question:
        current_q = prompt or st.session_state.pending_question

        if st.session_state.pending_question and not prompt:
            with st.chat_message("user"):
                st.markdown(st.session_state.pending_question)

        # Bug 28: Kill any running comparison thread before starting a new query
        _kill_comparison_thread()

        with st.spinner("🔄 Analyzing... this may take 1-2 minutes"):
            try:
                result = run_query(current_q, thread_id=st.session_state.thread_id)

                response_text = result.get("final_response", "No response generated.")
                charts = result.get("charts", [])
                query_time = result.get("query_time_seconds", 0)
                query_strategy = result.get("query_strategy", "unknown")

                # Determine if ANY data result actually used a view
                data_results = result.get("data_results", [])
                any_view_hit = any(
                    r.get("strategy") == "view" and not r.get("error")
                    for r in data_results
                )

                sql_time = result.get("sql_time_seconds", 0)

                entry = {
                    "question": current_q,
                    "response": response_text,
                    "charts": charts,
                    "query_strategy": query_strategy,
                    "query_time": query_time,
                    "sql_time": sql_time,
                    "base_time": None,
                }

                # Bug 32: Only start comparison when view was ACTUALLY used
                if any_view_hit:
                    comp_holder = {"status": "running", "view_time": query_time}
                    stop_event = threading.Event()
                    thread = threading.Thread(
                        target=_run_base_table_comparison,
                        args=(current_q, comp_holder, stop_event),
                        daemon=True,
                    )
                    thread.start()
                    st.session_state.comparison = {
                        "holder": comp_holder,
                        "stop_event": stop_event,
                        "thread": thread,
                        "view_time": query_time,
                    }
                else:
                    st.session_state.comparison = None

                st.session_state.chat_history.append(entry)
                st.session_state.last_analysis = result
                st.session_state.pending_question = None
                st.session_state.prompt = ""

                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logging.error("Analysis failed for prompt: %s | Error: %s", current_q, e, exc_info=True)
                st.session_state.pending_question = None
                st.session_state.prompt = ""

    if not st.session_state.chat_history and not st.session_state.pending_question:
        st.info("👆 Ask a question above or click a Quick Action in the sidebar to get started!")

with right_col:
    st.markdown("### 📈 Visualizations & Results")

    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]

        # Bug 32: Proper timing strategy
        query_time = latest.get("query_time", 0)
        sql_time = latest.get("sql_time", 0)
        if latest.get("query_strategy") == "view":
            banner_text = (f"Query Strategy: **Pre-Aggregation View** | "
                          f"Pre-Aggregation Total Time: **{query_time}s**")
            if sql_time > 0:
                banner_text += f" | Pre-Aggregation SQL Time: **{sql_time}s**"
        else:
            banner_text = (f"Query Strategy: **Base Table** | "
                          f"Base Table Total Time: **{query_time}s**")
            if sql_time > 0:
                banner_text += f" | Base Table SQL Time: **{sql_time}s**"

        # Bug 32: Comparison status with SQL time
        comp = st.session_state.comparison
        if comp:
            status = comp["holder"]["status"]
            if status == "running":
                banner_text += " | 🔄 Background comparison running..."
            elif status == "done":
                bt = comp["holder"].get("base_time")
                if bt and bt > 0:
                    banner_text += f" | ⚡ Base Table SQL Time: **{bt}s**"
                    if latest.get("base_time") is None:
                        latest["base_time"] = bt
                st.session_state.comparison = None  # Auto-cleanup
            elif status == "cancelled":
                banner_text += " | ⏹ Background comparison cancelled"
                st.session_state.comparison = None
            elif status == "error":
                banner_text += " | ❌ Background comparison error"

        st.info(banner_text)

        # Bug 19: Stop comparison button
        if comp and comp["holder"]["status"] == "running":
            if st.button("⏹ Stop Comparison Query", key="stop_comp"):
                comp["stop_event"].set()
                comp["holder"]["status"] = "cancelled"
                st.rerun()

        # Show charts
        tabs = st.tabs(["📊 Charts", "📋 Analysis", "🔍 Raw Data"])

        with tabs[0]:
            charts = latest.get("charts", [])
            if charts:
                for i, chart in enumerate(charts):
                    ctype = chart.get("type", "")
                    if ctype == "text":
                        # Display text content (e.g., CI summary)
                        st.markdown(chart.get("text", ""))
                    elif chart.get("html"):
                        _embed_html(chart["html"], height=500)
                    elif chart.get("png"):
                        st.image(chart["png"], use_container_width=True)
                    st.caption(f"**{chart.get('title', f'Chart {i+1}')}**")
            else:
                st.info("No charts generated for this query.")

        with tabs[1]:
            st.markdown(latest.get("response", "No analysis text available."))

        with tabs[2]:
            result = st.session_state.last_analysis
            if result:
                # Show data results — deduplicate by SQL
                seen_sqls = set()
                if result.get("data_results"):
                    for i, r in enumerate(result["data_results"]):
                        sql_key = r.get("sql", "")[:100] if r.get("sql") else str(i)
                        if sql_key in seen_sqls:
                            continue  # Skip duplicate data from retries
                        seen_sqls.add(sql_key)
                        if r.get("data") and not r.get("error"):
                            st.dataframe(pd.DataFrame(r["data"]), use_container_width=True)
                            st.caption(f"Strategy: {r.get('strategy', 'N/A')} | {r.get('summary', '')[:80]}")
                elif result.get("data_summary"):
                    st.text(result["data_summary"])

                # Bug 28: Show comparison cancellation message if cancelled
                comp = st.session_state.comparison
                if comp and comp["holder"]["status"] == "cancelled":
                    st.info("⏹ Backend Comparison Query Thread Cancelled")
                elif comp and comp["holder"]["status"] == "done":
                    base_data = comp["holder"].get("base_data", [])
                    if base_data:
                        st.markdown("---")
                        st.markdown("**Base Table Comparison Results:**")
                        st.dataframe(pd.DataFrame(base_data), use_container_width=True)
                        bt = comp["holder"].get("base_time", 0)
                        vt = comp.get("view_time", 0)
                        st.caption(f"Base Table Total Time: {bt}s | Pre-Aggregation Biew Total Time: {vt}s")
    else:
        # Default dashboard overview
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

st.markdown("---")
st.caption("Built with Python, Streamlit, LangGraph, MySQL | Agentic BI Final Project")
