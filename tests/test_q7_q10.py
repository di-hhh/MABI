import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.data_analyst import analyze

for label, q in [
    ("Q7", "基于全部分析结果，给出平台 3 个月内的三大优先改进策略。"),
    ("Q10", "如何降低巴西东北部地区的高退货率？请给出具体的运营改进方案。"),
]:
    r = analyze(q)
    err = r.get('error') or 'none'
    print(f"{label}: strategy={r['strategy']}, rows={len(r['data'])}, err={str(err)[:120]}")
