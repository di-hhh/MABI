import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.graph import run_query
r = run_query("为什么某些州的平均配送时长显著高于全国均值？哪些卖家的差评率最高？", thread_id="q9_final")
err = r.get("error") or "none"
print(f"strategy={r.get('query_strategy','?')}, charts={len(r.get('charts',[]))}, err={str(err)[:120]}")
