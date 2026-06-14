import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.graph import run_query
r = run_query("如何降低巴西东北部地区的高退货率？请给出具体的运营改进方案。", thread_id="q10_v3")
err = r.get("error") or "none"
print(f"strategy={r.get('query_strategy','?')}, charts={len(r.get('charts',[]))}, err={str(err)[:120]}")
