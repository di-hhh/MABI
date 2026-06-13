"""Conversation memory helpers for follow-up BI analysis.

The graph already uses LangGraph's MemorySaver. This module turns the saved
state into a compact business context so follow-up questions such as
"these states" or "continue with those sellers" can reuse the previous result.
"""
from __future__ import annotations

from collections import Counter
import re


FOLLOW_UP_SIGNALS = [
    "these", "those", "them", "that", "same", "above", "previous",
    "continue", "follow up", "再", "继续", "进一步", "这些", "那些",
    "上述", "上面", "它们", "他们", "这些州", "这些品类", "这些卖家", "刚才",
]


ENTITY_COLUMNS = {
    "states": ["customer_state", "seller_state", "state"],
    "categories": ["category", "product_category_english", "product_category_name"],
    "sellers": ["seller_id"],
    "payments": ["payment_type"],
}

_THREAD_MEMORY: dict[str, dict] = {}


def get_thread_memory(thread_id: str) -> dict:
    """Return the latest completed state stored for this app process."""
    return _THREAD_MEMORY.get(thread_id, {})


def save_thread_memory(thread_id: str, state: dict) -> None:
    """Persist the latest completed state for robust follow-up resolution."""
    if thread_id:
        _THREAD_MEMORY[thread_id] = dict(state)


def _as_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _collect_rows(prev_values: dict, row_limit: int = 30) -> list[dict]:
    rows: list[dict] = []
    for result in prev_values.get("data_results", []) or []:
        for row in (result.get("data") or [])[:row_limit]:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _top_entity_values(rows: list[dict], columns: list[str], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        lowered = {str(k).lower(): v for k, v in row.items()}
        for col in columns:
            if col in lowered:
                value = _as_text(lowered[col])
                if value:
                    counter[value] += 1
    return [value for value, _ in counter.most_common(limit)]


def build_memory_snapshot(prev_values: dict | None) -> dict:
    """Extract a compact memory snapshot from the previous graph state."""
    if not prev_values:
        return {}

    rows = _collect_rows(prev_values)
    snapshot = {
        "previous_question": _as_text(prev_values.get("question")),
        "previous_resolved_question": _as_text(prev_values.get("resolved_question")),
        "previous_summary": _as_text(prev_values.get("data_summary"))[:700],
        "previous_recommendations": _as_text(prev_values.get("recommendations"))[:700],
        "analysis_type": _as_text(prev_values.get("analysis_type")),
    }

    for name, columns in ENTITY_COLUMNS.items():
        values = _top_entity_values(rows, columns)
        if values:
            snapshot[name] = values

    scenario = prev_values.get("scenario_results") or {}
    if scenario.get("what_if"):
        snapshot["last_what_if"] = scenario["what_if"]
    if scenario.get("anomaly"):
        snapshot["last_anomaly_summary"] = scenario["anomaly"].get("summary", "")

    return {k: v for k, v in snapshot.items() if v}


def is_follow_up_question(question: str) -> bool:
    """Return True when the question appears to reference prior context."""
    q = question.strip().lower()
    if any(signal in q for signal in FOLLOW_UP_SIGNALS):
        return True
    # Short imperative follow-ups are often contextual even without pronouns.
    return len(q) <= 36 and any(w in q for w in ["why", "原因", "建议", "优化", "show", "compare"])


def format_memory_context(snapshot: dict) -> str:
    """Format memory as a concise context block for prompts and UI."""
    if not snapshot:
        return ""

    lines = []
    if snapshot.get("previous_question"):
        lines.append(f"Previous question: {snapshot['previous_question']}")
    if snapshot.get("states"):
        lines.append(f"Previous focus states: {', '.join(snapshot['states'])}")
    if snapshot.get("categories"):
        lines.append(f"Previous focus categories: {', '.join(snapshot['categories'])}")
    if snapshot.get("sellers"):
        lines.append(f"Previous focus sellers: {', '.join(snapshot['sellers'])}")
    if snapshot.get("payments"):
        lines.append(f"Previous payment methods: {', '.join(snapshot['payments'])}")
    if snapshot.get("previous_summary"):
        lines.append(f"Previous findings: {snapshot['previous_summary']}")
    if snapshot.get("last_anomaly_summary"):
        lines.append(f"Previous anomaly scan: {snapshot['last_anomaly_summary']}")
    return "\n".join(lines)


def resolve_follow_up_question(question: str, snapshot: dict) -> tuple[str, str]:
    """Attach memory context to follow-up questions.

    Returns (resolved_question, memory_context). Non-follow-up questions keep the
    original wording and do not expose memory_context. This keeps memory useful
    for true follow-up analysis without leaking internal context into unrelated
    turns.
    """
    memory_context = format_memory_context(snapshot)
    if not memory_context or not is_follow_up_question(question):
        return question, ""

    entity_hint_parts = []
    for label in ("states", "categories", "sellers", "payments"):
        if snapshot.get(label):
            entity_hint_parts.append(f"{label}={', '.join(snapshot[label])}")
    entity_hint = "; ".join(entity_hint_parts)

    resolved = (
        f"{question}\n\n"
        "[Conversation memory for resolving references]\n"
        f"{memory_context}\n"
    )
    if entity_hint:
        resolved += f"Reference mapping: {entity_hint}\n"
    resolved += (
        "If the user says these/those/above/same, interpret it using the "
        "previous focus entities and findings."
    )
    return resolved, memory_context
