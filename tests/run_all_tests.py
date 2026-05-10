"""Comprehensive auto-test: ALL §四 + §九 queries with validation.
Run: python tests/run_all_tests.py
"""
import sys, os, time, logging, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("logs/test_all.log", encoding="utf-8")],
)

from agents.graph import run_query

QUERIES = {
    # §四 scenarios
    "§四.1-descriptive": "2017年哪个州的销售额最高？交付准时率是多少？哪种支付方式最受欢迎？",
    "§四.2-diagnostic": "为什么某些州的平均配送时长显著高于全国均值？哪些卖家的差评率最高？",
    "§四.3-predictive": "根据历史订单趋势，预测未来6周的销售额",
    "§四.4-prescriptive": "如何降低巴西东北部地区的高退货率？请给出具体的运营改进方案。",
    # §九 queries (distinct from §四)
    "Q1": "2017 年 GMV 是多少？按月和各州排名的趋势怎样？",
    "Q2": "平台整体准时交付率是多少？哪些州延迟最严重？",
    "Q3": "哪种支付方式最受欢迎？平均分期数是多少？",
    "Q4": "产品的重量、尺寸与运费之间有什么关系？",
    "Q5": "Top 10 差评品类及其主要差评原因是什么？",
    "Q6": "根据历史订单趋势，预测未来 6 周的销售额，并给出趋势解读。",
    "Q7": "基于全部分析结果，给出平台 3 个月内的三大优先改进策略。",
    "Q8": "2017年哪个州的销售额最高？交付准时率是多少？哪种支付方式最受欢迎？",
    "Q9": "为什么某些州的平均配送时长显著高于全国均值？哪些卖家的差评率最高？",
    "Q10": "如何降低巴西东北部地区的高退货率？请给出具体的运营改进方案。",
}

passed = 0
failed = 0
results = []

for label, question in QUERIES.items():
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Q: {question[:80]}...")
    t0 = time.time()

    try:
        result = run_query(question, thread_id=f"autotest_{label}")
        elapsed = time.time() - t0
        error = result.get("error", "")
        strategy = result.get("query_strategy", "?")
        n_charts = len(result.get("charts", []))
        has_data = any(r.get("data") and len(r["data"]) > 0
                       for r in result.get("data_results", []))
        has_summary = bool(result.get("data_summary", "").strip())
        atype = result.get("analysis_type", "?")

        status = "PASS"
        issues = []
        if error:
            status = "FAIL"
            issues.append(f"Error: {error[:100]}")
        if n_charts == 0:
            status = "FAIL"
            issues.append("No charts")
        if not has_data and not has_summary:
            status = "FAIL"
            issues.append("No data or summary")
        if strategy == "error":
            status = "FAIL"
            issues.append("Strategy=error")

        chart_types = set(c.get("type", "?") for c in result.get("charts", []))
        print(f"  [{status}] {elapsed:.1f}s | {strategy} | {n_charts} charts | type={atype}")
        print(f"  Charts: {sorted(chart_types)}")
        if issues:
            print(f"  Issues: {'; '.join(issues)}")

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        results.append({
            "label": label, "status": status, "time": round(elapsed, 1),
            "strategy": strategy, "charts": n_charts, "chart_types": sorted(chart_types),
            "analysis_type": atype, "issues": issues,
        })

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] Exception: {e}")
        failed += 1
        results.append({"label": label, "status": "EXCEPTION", "time": round(elapsed, 1), "error": str(e)[:200]})

print(f"\n{'='*60}")
print(f"RESULTS: {passed} PASS, {failed} FAIL out of {len(QUERIES)}")
print(f"{'='*60}")

# Summary table
print(f"\n{'Label':<25} {'Status':<6} {'Time':>6}s {'Strategy':<12} {'Charts':>6} {'Types'}")
print("-" * 90)
for r in results:
    print(f"{r['label']:<25} {r['status']:<6} {r['time']:>6.1f}s {r.get('strategy','?'):<12} {r.get('charts',0):>6} {','.join(r.get('chart_types',[]))}")

if failed > 0:
    print("\nFAILURES:")
    for r in results:
        if r["status"] != "PASS":
            print(f"  {r['label']}: {r.get('issues', r.get('error', 'unknown'))}")

exit(0 if failed == 0 else 1)
