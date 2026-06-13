"""Focused tests for bonus-feature plumbing.

These tests avoid LLM calls and exercise the deterministic pieces added for the
bonus requirements: review topic extraction, follow-up memory, and scenario
serialization.
"""
from decimal import Decimal

from agents.memory import build_memory_snapshot, resolve_follow_up_question
from agents.nlp_insight import _extract_review_themes
from agents.scenario import _jsonable


def test_review_theme_extraction_detects_delivery_delay():
    reviews = [
        {
            "review_score": 1,
            "review_comment_message": "entrega com atraso e muita demora",
            "category": "health_beauty",
        },
        {
            "review_score": 2,
            "review_comment_message": "prazo ruim, transportadora atrasou",
            "category": "health_beauty",
        },
    ]

    themes = _extract_review_themes(reviews, positive=False)

    assert themes
    assert themes[0]["theme"] == "Delivery delay / logistics reliability"
    assert themes[0]["review_count"] == 2


def test_memory_resolves_follow_up_entities():
    prev_state = {
        "question": "Which states have the worst delivery on-time rates?",
        "data_summary": "Delivery performance by state",
        "recommendations": "Prioritize AL and MA",
        "data_results": [
            {"data": [{"customer_state": "AL"}, {"customer_state": "MA"}]},
        ],
    }

    snapshot = build_memory_snapshot(prev_state)
    resolved, context = resolve_follow_up_question(
        "For these states, what should we improve?",
        snapshot,
    )

    assert "AL" in context
    assert "MA" in context
    assert "[Conversation memory" in resolved


def test_scenario_jsonable_converts_decimal_and_numpy_like_values():
    class NumpyLike:
        def item(self):
            return 3.14

    payload = {
        "decimal": Decimal("1.23"),
        "nested": [{"value": NumpyLike()}],
    }

    converted = _jsonable(payload)

    assert converted["decimal"] == 1.23
    assert converted["nested"][0]["value"] == 3.14
