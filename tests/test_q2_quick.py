import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.data_analyst import analyze
r = analyze("平台整体准时交付率是多少？哪些州延迟最严重？")
err = r.get('error') or 'none'
print(f"strategy={r['strategy']}, rows={len(r['data'])}, err={str(err)[:100]}")
