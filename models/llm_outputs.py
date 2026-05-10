"""Pydantic v2 models for strict LLM output validation (Bug 20 + Bug 26).

All LLM-generated outputs pass through Pydantic validation before use.
This eliminates JSON parse fragility and ensures type safety.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ── Data Analyst ────────────────────────────────────────

class DataAnalystOutput(BaseModel):
    """Validated output from Data Analyst Agent LLM."""
    strategy: str = Field(default="base_table", description="view or base_table")
    reasoning: str = Field(default="", description="Why this approach was chosen")
    sql: str = Field(default="", description="The SQL query to execute")
    summary: str = Field(default="", description="What this query returns")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in ("view", "base_table", "error", "unknown"):
            raise ValueError(f"Invalid strategy: {v}")
        return v

    @field_validator("sql")
    @classmethod
    def block_dangerous_sql(cls, v: str) -> str:
        upper = v.strip().upper()
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER", "CREATE"]
        for kw in dangerous:
            if upper.startswith(kw):
                raise ValueError(f"Blocked dangerous SQL starting with {kw}")
        return v

    @field_validator("sql")
    @classmethod
    def fix_wrong_table_names(cls, v: str) -> str:
        """Fix all hallucinated table names (Bug 33)."""
        import re as _re
        # Strip olist_ prefix and _dataset suffix from any identifier
        v = _re.sub(r'\bolist_', '', v, flags=_re.IGNORECASE)
        v = _re.sub(r'_dataset\b', '', v, flags=_re.IGNORECASE)
        # Fix singular → plural
        fixes = {" order_item ": " order_items ", " order_review ": " order_reviews ",
                 " product ": " products ", " order_payment ": " payments "}
        # Fix known wrong names with word boundaries
        wrong_tables = {
            "order_payments": "payments", "order_review": "order_reviews",
            "order_item": "order_items", "order_items_dataset": "order_items",
            "orders_dataset": "orders", "products_dataset": "products",
            "customers_dataset": "customers", "sellers_dataset": "sellers",
            "order_reviews_dataset": "order_reviews", "payments_dataset": "payments",
            "geolocation_dataset": "geolocation",
        }
        for wrong, correct in wrong_tables.items():
            v = _re.sub(r'\b' + wrong + r'\b', correct, v, flags=_re.IGNORECASE)
        return v


# ── Coordinator ─────────────────────────────────────────

class CoordinatorTask(BaseModel):
    """A single task in the coordinator's plan."""
    agent: str = Field(description="Agent name: data_analyst, visualizer, nlp_insight, decision, predictor")
    task: str = Field(description="Specific instruction for this agent")


class CoordinatorOutput(BaseModel):
    """Validated output from Coordinator Agent LLM."""
    question_summary: str = Field(default="", description="Brief restatement of user question")
    analysis_type: str = Field(default="descriptive", description="descriptive, diagnostic, predictive, or prescriptive")
    tasks: list[CoordinatorTask] = Field(default_factory=list)
    final_synthesis: str = Field(default="", description="How to combine results")

    @field_validator("analysis_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid = {"descriptive", "diagnostic", "predictive", "prescriptive"}
        if v not in valid:
            # Default to descriptive for unknown types
            return "descriptive"
        return v

    @field_validator("tasks")
    @classmethod
    def ensure_data_analyst(cls, v: list) -> list:
        """Always ensure at least one data_analyst task."""
        if not any(t.agent == "data_analyst" for t in v):
            v.append(CoordinatorTask(agent="data_analyst", task="Analyze the user question"))
        return v


# ── SQL Correction (retry) ──────────────────────────────

class SQLCorrectionOutput(BaseModel):
    """Validated output from SQL correction retry."""
    strategy: str = Field(default="base_table")
    sql: str = Field(description="Corrected SQL query")
    reasoning: str = Field(default="")


# ── Chart Selection ─────────────────────────────────────

class ChartSelectionOutput(BaseModel):
    """Validated chart type selection from LLM."""
    charts: list[str] = Field(default_factory=list, description="Selected chart types")

    @field_validator("charts")
    @classmethod
    def validate_chart_types(cls, v: list[str]) -> list[str]:
        valid = {"line", "geo_map", "state_bar", "payment_bar", "payment_heatmap",
                 "wordcloud", "category_bar", "scatter", "delivery_bar", "basket_bar"}
        return [c for c in v if c in valid]


# ── NLP Sentiment ───────────────────────────────────────

class NLPSentimentOutput(BaseModel):
    """Validated NLP sentiment analysis output."""
    sentiment_summary: str = Field(default="")
    top_positive_keywords: list[str] = Field(default_factory=list)
    top_negative_keywords: list[str] = Field(default_factory=list)
    key_insights: str = Field(default="")


# ── Parser helpers ──────────────────────────────────────

import re as _re
import json as _json
import logging

_logger = logging.getLogger(__name__)


def safe_parse_pydantic(response_text: str, model_class: type[BaseModel]) -> Optional[BaseModel]:
    """Extract and validate JSON from LLM response using Pydantic model.

    Uses aggressive JSON repair BEFORE Pydantic validation (Bug 33).
    Returns None only if all strategies fail.
    """
    text = response_text.strip()
    candidates = []

    # Strategy 1: Extract JSON from markdown fences
    m = _re.search(r"```(?:json)?\s*\n?(.*?)```", text, _re.DOTALL)
    if m:
        candidates.append(m.group(1).strip())

    # Strategy 2: Extract JSON array
    m = _re.search(r"\[.*\]", text, _re.DOTALL)
    if m:
        candidates.append(m.group(0))

    # Strategy 3: Extract JSON object
    m = _re.search(r"\{.*\}", text, _re.DOTALL)
    if m:
        candidates.append(m.group(0))

    candidates.append(text)

    for candidate in candidates:
        if not candidate.strip():
            continue
        # Aggressive JSON repair
        for repaired in _repair_json(candidate):
            try:
                return model_class.model_validate_json(repaired)
            except Exception:
                continue

    _logger.warning("All Pydantic parse strategies failed for %s", model_class.__name__)
    return None


def _repair_json(text: str) -> list[str]:
    """Generate progressively repaired JSON candidates (Bug 33)."""
    results = []
    t = text.strip()

    # 1: Raw text
    results.append(t)

    # 2: Remove trailing commas before ] or }
    results.append(_re.sub(r",\s*([}\]])", r"\1", t))

    # 3: Quote unquoted keys
    c = _re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', t)
    c = _re.sub(r",\s*([}\]])", r"\1", c)
    results.append(c)

    # 4: Single quotes → double quotes for keys
    c = _re.sub(r"'([^']*)':", r'"\1":', t)
    c = _re.sub(r",\s*([}\]])", r"\1", c)
    results.append(c)

    # 5: Handle escaped quotes inside strings
    c = _re.sub(r'\\"', '"', t)
    c = _re.sub(r",\s*([}\]])", r"\1", c)
    results.append(c)

    # 6: Complete missing closing braces
    open_b = t.count("{") - t.count("}")
    if open_b > 0:
        c = t.rstrip() + ("}" * open_b)
        c = _re.sub(r",\s*([}\]])", r"\1", c)
        c = _re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', c)
        results.append(c)

    # 7: Remove non-JSON prefix/suffix
    m = _re.search(r"[\{\[].*[\}\]]", t, _re.DOTALL)
    if m:
        results.append(m.group(0))

    return results
