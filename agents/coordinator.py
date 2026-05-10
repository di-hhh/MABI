"""Coordinator Agent — parse user questions with Pydantic v2 validation."""
import logging
from config.prompts import COORDINATOR_SYSTEM
from utils.llm import chat
from models.llm_outputs import CoordinatorOutput, CoordinatorTask, safe_parse_pydantic

logger = logging.getLogger(__name__)


def parse_question(question: str) -> dict:
    """Parse user question → validated task plan via Pydantic v2.

    Returns dict with: question_summary, analysis_type, tasks, final_synthesis.
    """
    user_prompt = f"""User Question: {question}

Analyze this question and produce a JSON task plan.
Available agents: data_analyst, visualizer, nlp_insight, decision, predictor.

CRITICAL: Output ONLY valid JSON. No markdown, no explanation.

Example:
{{"question_summary":"Summary","analysis_type":"descriptive","tasks":[{{"agent":"data_analyst","task":"Query"}}],"final_synthesis":"Combine results"}}"""

    try:
        response = chat(
            [{"role": "system", "content": COORDINATOR_SYSTEM},
             {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=1024, json_mode=True,
        )

        # Pydantic v2 validation
        parsed = safe_parse_pydantic(response, CoordinatorOutput)

        if parsed is not None:
            logger.info("Coordinator plan: %s → %d tasks [%s]",
                        parsed.question_summary, len(parsed.tasks), parsed.analysis_type)
            return {
                "question_summary": parsed.question_summary,
                "analysis_type": parsed.analysis_type,
                "tasks": [t.model_dump() for t in parsed.tasks],
                "final_synthesis": parsed.final_synthesis,
            }

    except Exception as e:
        logger.error("Coordinator error: %s", e, exc_info=True)

    # Fallback
    logger.warning("Pydantic parse failed, using heuristic plan")
    return _heuristic_plan(question)


def _heuristic_plan(question: str) -> dict:
    """Rule-based task planning when LLM/Pydantic fails."""
    q = question.lower()

    # Detect multi-part questions and split into separate data_analyst sub-tasks
    has_state = any(w in q for w in ["哪个州", "销售额最高", "各州", "州排名"])
    has_delivery = any(w in q for w in ["准时", "配送", "交付", "延迟"])
    has_payment = any(w in q for w in ["支付", "分期"])
    part_count = sum([has_state, has_delivery, has_payment])

    if part_count >= 2:
        tasks = []
        if has_state:
            tasks.append({"agent": "data_analyst",
                          "task": "查询2017年各州销售额排名，找出销售额最高的州。使用 mv_state_sales 预聚合视图。"})
        if has_delivery:
            tasks.append({"agent": "data_analyst",
                          "task": "查询平台整体交付准时率。使用 mv_delivery_perf 预聚合视图，计算所有州的平均 on_time_rate。"})
        if has_payment:
            tasks.append({"agent": "data_analyst",
                          "task": "查询哪种支付方式最受欢迎。使用 mv_payment_dist 预聚合视图，按 payment_type 聚合总交易数。"})
    else:
        tasks = [{"agent": "data_analyst", "task": question}]

    if any(w in q for w in ["预测", "predict", "forecast", "未来"]):
        atype = "predictive"
    elif any(w in q for w in ["如何降低", "如何提升", "如何改善", "怎样降低", "如何优化", "如何改进"]):
        atype = "prescriptive"
    elif any(w in q for w in ["为什么", "原因", "why", "哪些.*差评", "差评率", "退货"]):
        atype = "diagnostic"
    elif any(w in q for w in ["建议", "优化", "改进", "策略", "recommend", "how to", "方案"]):
        atype = "prescriptive"
    else:
        atype = "descriptive"

    tasks.append({"agent": "visualizer", "task": f"Generate charts for: {question}"})

    if atype in ("diagnostic", "prescriptive"):
        tasks.append({"agent": "nlp_insight", "task": f"Analyze reviews for: {question}"})
    if atype == "predictive":
        tasks.append({"agent": "predictor", "task": f"Forecast for: {question}"})
    if atype in ("prescriptive", "diagnostic"):
        tasks.append({"agent": "decision", "task": f"Recommendations for: {question}"})

    return {
        "question_summary": question[:100],
        "analysis_type": atype,
        "tasks": tasks,
        "final_synthesis": "Combine all agent outputs into a coherent response.",
    }
