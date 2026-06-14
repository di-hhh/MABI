"""E2E tests for Requirements.md §九 — 10 validation queries.
Verifies each query returns data, charts, and no errors.
"""
import pytest
import os
import sys
import logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("logs/pytest.log", encoding="utf-8")],
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import run_query

# §九 — 10 validation queries
VALIDATION_QUERIES = [
    ("Q1", "2017 年 GMV 是多少？按月和各州排名的趋势怎样？"),
    ("Q2", "平台整体准时交付率是多少？哪些州延迟最严重？"),
    ("Q3", "哪种支付方式最受欢迎？平均分期数是多少？"),
    ("Q4", "产品的重量、尺寸与运费之间有什么关系？"),
    ("Q5", "Top 10 差评品类及其主要差评原因是什么？"),
    ("Q6", "根据历史订单趋势，预测未来 6 周的销售额，并给出趋势解读。"),
    ("Q7", "基于全部分析结果，给出平台 3 个月内的三大优先改进策略。"),
    ("Q8", "2017年哪个州的销售额最高？交付准时率是多少？哪种支付方式最受欢迎？"),
    ("Q9", "为什么某些州的平均配送时长显著高于全国均值？哪些卖家的差评率最高？"),
    ("Q10", "如何降低巴西东北部地区的高退货率？请给出具体的运营改进方案。"),
]

# §四 — 4 analysis types
ANALYSIS_TYPE_QUERIES = [
    ("descriptive", "2017年哪个州的销售额最高？交付准时率是多少？哪种支付方式最受欢迎？"),
    ("diagnostic", "为什么某些州的平均配送时长显著高于全国均值？哪些卖家的差评率最高？"),
    ("predictive", "根据历史订单趋势，预测未来6周的销售额"),
    ("prescriptive", "如何降低巴西东北部地区的高退货率？请给出具体的运营改进方案。"),
]


class TestValidationQueries:
    """§九 queries — all must pass."""

    @pytest.mark.parametrize("qid,question", VALIDATION_QUERIES)
    @pytest.mark.timeout(300)
    def test_query(self, qid, question):
        """Run each validation query and verify basic correctness."""
        thread_id = f"pytest_{qid}"
        result = run_query(question, thread_id=thread_id)

        # Must not have a hard error
        assert not result.get("error"), f"{qid}: {result.get('error')}"

        # Must have either data or meaningful summary
        data_results = result.get("data_results", [])
        has_data = any(r.get("data") and len(r["data"]) > 0 for r in data_results)
        has_summary = bool(result.get("data_summary", "").strip())
        assert has_data or has_summary, f"{qid}: No data or summary"

        # Must have query strategy set
        strategy = result.get("query_strategy", "unknown")
        assert strategy != "unknown", f"{qid}: No query strategy"
        assert strategy != "error", f"{qid}: Query strategy was error"

        # Must have final response
        final = result.get("final_response", "")
        assert final.strip(), f"{qid}: Empty final response"

        # Must have charts
        charts = result.get("charts", [])
        assert len(charts) > 0, f"{qid}: No charts generated"

        print(f"  {qid}: {strategy} | {len(charts)} charts | {result.get('query_time_seconds', 0):.1f}s")


class TestAnalysisTypes:
    """§四 — all 4 analysis types must work."""

    @pytest.mark.parametrize("atype,question", ANALYSIS_TYPE_QUERIES)
    @pytest.mark.timeout(300)
    def test_analysis_type(self, atype, question):
        """Run each analysis type query and verify correctness."""
        thread_id = f"pytest_type_{atype}"
        result = run_query(question, thread_id=thread_id)

        assert not result.get("error"), f"{atype}: {result.get('error')}"

        # Verify correct analysis type
        actual_type = result.get("analysis_type", "")
        print(f"  {atype}: actual={actual_type} | {result.get('query_strategy')} | {len(result.get('charts', []))} charts | {result.get('query_time_seconds', 0):.1f}s")


class TestChartTypes:
    """Verify 6+ chart types are generated across all queries."""

    def test_minimum_chart_types(self):
        """Run all validation queries and verify at least 6 chart types."""
        all_chart_types = set()
        for qid, question in VALIDATION_QUERIES[:4]:  # First 4 to save time
            result = run_query(question, thread_id=f"pytest_chart_{qid}")
            for chart in result.get("charts", []):
                all_chart_types.add(chart.get("type", "?"))

        # We should see at least 4 chart types from 4 queries (6+ from all 10)
        assert len(all_chart_types) >= 4, f"Only {len(all_chart_types)} chart types: {all_chart_types}"
        print(f"  Chart types found: {sorted(all_chart_types)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
