"""Focused tests for bonus-feature plumbing.

These tests avoid LLM calls and exercise the deterministic pieces added for the
bonus requirements: review topic extraction, follow-up memory, and scenario
serialization.
"""
from decimal import Decimal

from agents.memory import build_memory_snapshot, resolve_follow_up_question
from agents.nlp_insight import _extract_review_themes
from agents.scenario import _jsonable
from models.predictor import forecast_monthly


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


def test_forecast_never_returns_negative_gmv(monkeypatch):
    import pandas as pd

    history = pd.DataFrame({
        "ym": ["2018-01", "2018-02", "2018-03", "2018-04"],
        "ds": pd.to_datetime(["2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01"]),
        "y": [100.0, 90.0, 80.0, 70.0],
    })

    class FakeModel:
        def make_future_dataframe(self, periods, freq):
            return pd.DataFrame({"ds": pd.date_range("2018-04-01", periods=periods + 2, freq="D")})

        def predict(self, future):
            result = future.copy()
            result["yhat"] = -10.0
            result["yhat_lower"] = -20.0
            result["yhat_upper"] = -5.0
            return result

    monkeypatch.setattr("models.predictor.get_monthly_sales_data", lambda: history)
    monkeypatch.setattr("models.predictor._train_prophet", lambda df: FakeModel())

    result = forecast_monthly(weeks=1)

    assert result["forecast"]
    assert all(row["yhat"] >= 0 for row in result["forecast"])
    assert all(row["yhat_lower"] >= 0 for row in result["forecast"])
    assert all(row["yhat_upper"] >= row["yhat"] for row in result["forecast"])
