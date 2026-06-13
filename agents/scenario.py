"""What-if and anomaly scenario agent.

This agent is intentionally deterministic: it computes scenario outputs from
the database instead of asking the LLM to invent numbers. The Decision Agent can
then interpret these outputs in business language.
"""
from __future__ import annotations

import logging
import re
from decimal import Decimal
from datetime import date, datetime

from agents.anomaly_detector import run_all_checks
from agents.decision import what_if_remove_worst_sellers

logger = logging.getLogger(__name__)


WHAT_IF_SIGNALS = [
    "what-if", "what if", "scenario", "simulate", "simulation", "hypothetical",
    "remove", "exclude", "screening",
    "假设", "如果", "模拟", "情景", "下架", "移除", "剔除", "筛选",
]

ANOMALY_SIGNALS = [
    "anomaly", "anomalies", "abnormal", "outlier", "alert", "scan", "spike", "drop",
    "异常", "告警", "预警", "扫描", "突增", "暴跌", "异常检测", "波动",
]


def _contains_any(text: str, signals: list[str]) -> bool:
    lowered = text.lower()
    return any(signal.lower() in lowered for signal in signals)


def _extract_top_n(question: str, default: int = 20) -> int:
    match = re.search(r"(?:top\s*)?(\d+)", question, flags=re.IGNORECASE)
    if not match:
        return default
    return max(1, min(100, int(match.group(1))))


def _jsonable(value):
    """Convert DB/numpy scalar values into MemorySaver-serializable objects."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def should_run_what_if(question: str, analysis_type: str = "") -> bool:
    q = question.lower()
    return _contains_any(q, WHAT_IF_SIGNALS) and any(
        signal in q
        for signal in [
            "seller", "卖家", "remove", "exclude", "screening", "下架", "移除",
            "剔除", "what", "如果", "假设", "模拟",
        ]
    )


def should_run_anomaly(question: str, analysis_type: str = "") -> bool:
    q = question.lower()
    if _contains_any(q, ANOMALY_SIGNALS):
        return True
    return analysis_type == "diagnostic" and any(w in q for w in ["drop", "spike", "异常", "波动"])


def run_scenario_analysis(question: str, analysis_type: str = "") -> dict:
    """Run requested what-if and anomaly checks.

    Returns a stable dict for graph state and Streamlit rendering.
    """
    result = {
        "ran": [],
        "what_if": {},
        "anomaly": {},
        "summary": "",
    }

    summaries: list[str] = []

    if should_run_what_if(question, analysis_type):
        top_n = _extract_top_n(question)
        try:
            what_if = what_if_remove_worst_sellers(top_n=top_n)
            result["what_if"] = what_if
            result["ran"].append("what_if_remove_worst_sellers")
            if not what_if.get("error"):
                summaries.append(
                    "What-if: removing/screening the worst "
                    f"{what_if.get('removed_sellers_count', top_n)} sellers changes average review score "
                    f"from {what_if.get('current_avg_score')} to {what_if.get('new_avg_score')} "
                    f"({what_if.get('improvement_pct')}%)."
                )
        except Exception as exc:
            logger.exception("What-if scenario failed")
            result["what_if"] = {"error": str(exc)}

    if should_run_anomaly(question, analysis_type):
        try:
            anomaly = run_all_checks()
            result["anomaly"] = anomaly
            result["ran"].append("anomaly_detection")
            summaries.append(anomaly.get("summary", "Anomaly scan completed."))
        except Exception as exc:
            logger.exception("Anomaly detection failed")
            result["anomaly"] = {"error": str(exc)}

    result["summary"] = " ".join(summaries)
    return _jsonable(result)
