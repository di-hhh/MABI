"""Coordinator Agent — parse user questions and route to specialist agents."""
import json
import re
import logging
from config.prompts import COORDINATOR_SYSTEM
from utils.llm import chat

logger = logging.getLogger(__name__)


def parse_question(question: str) -> dict:
    """Parse a user question and return a task plan.

    Returns:
        {
            "question_summary": str,
            "tasks": [{"agent": str, "task": str}, ...],
            "final_synthesis": str,
            "analysis_type": str
        }
    """
    user_prompt = f"""User Question: {question}

Analyze this question and produce a JSON task plan. Determine which agents are needed.
Available agents: data_analyst, visualizer, nlp_insight, decision, predictor.

Respond ONLY with valid JSON containing: question_summary, tasks, final_synthesis."""

    messages = [
        {"role": "system", "content": COORDINATOR_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = chat(messages, temperature=0.2, max_tokens=1024)
        json_str = response.strip()
        m = re.search(r"\{.*\}", json_str, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
        else:
            parsed = json.loads(json_str)

        # Classify analysis type
        q_lower = question.lower()
        if any(w in q_lower for w in ["预测", "predict", "forecast", "趋势预测", "未来"]):
            parsed["analysis_type"] = "predictive"
        elif any(w in q_lower for w in ["建议", "优化", "改进", "策略", "recommend", "improve", "how to", "what if"]):
            parsed["analysis_type"] = "prescriptive"
        elif any(w in q_lower for w in ["为什么", "原因", "why", "差评", "负面", "延迟"]):
            parsed["analysis_type"] = "diagnostic"
        else:
            parsed["analysis_type"] = "descriptive"

        logger.info("Coordinator plan: %s → %d tasks [%s]",
                    parsed.get("question_summary", ""),
                    len(parsed.get("tasks", [])),
                    parsed.get("analysis_type", "descriptive"))

        return parsed

    except Exception as e:
        logger.error("Coordinator parsing error: %s", e, exc_info=True)

        # Fallback: simple heuristic parsing
        return _heuristic_plan(question)


def _heuristic_plan(question: str) -> dict:
    """Simple rule-based task planning when LLM fails."""
    q = question.lower()
    tasks = []

    # Always need data analyst
    tasks.append({"agent": "data_analyst", "task": question})

    # Add visualizer for any data question
    if any(w in q for w in ["趋势", "trend", "对比", "分布", "图", "chart", "visual", "排名", "top"]):
        tasks.append({"agent": "visualizer", "task": "Generate appropriate chart for: " + question})

    # Add NLP for review/sentiment questions
    if any(w in q for w in ["评论", "评价", "评分", "review", "差评", "好评", "sentiment", "情感"]):
        tasks.append({"agent": "nlp_insight", "task": "Analyze review sentiment: " + question})

    # Add predictor for forecast questions
    if any(w in q for w in ["预测", "predict", "forecast", "未来", "趋势"]):
        tasks.append({"agent": "predictor", "task": "Generate time series forecast: " + question})

    # Add decision for prescriptive questions
    if any(w in q for w in ["建议", "优化", "改进", "策略", "recommend", "improve", "how to", "策略"]):
        tasks.append({"agent": "decision", "task": "Generate recommendations: " + question})

    atype = "descriptive"
    if any(w in q for w in ["预测", "predict", "forecast"]):
        atype = "predictive"
    elif any(w in q for w in ["建议", "优化", "改进", "策略", "recommend"]):
        atype = "prescriptive"
    elif any(w in q for w in ["为什么", "原因", "why", "差评"]):
        atype = "diagnostic"

    return {
        "question_summary": question[:100],
        "tasks": tasks,
        "final_synthesis": "Summarize findings from all agents into a coherent response.",
        "analysis_type": atype,
    }
