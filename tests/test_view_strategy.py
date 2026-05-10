"""Quick test of view-first strategy."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.data_analyst import analyze

tests = [
    ("Q2-delivery", "平台整体准时交付率是多少？哪些州延迟最严重？"),
    ("Q3-payment", "哪种支付方式最受欢迎？平均分期数是多少？"),
    ("Q1-monthly", "2017 年 GMV 是多少？按月和各州排名的趋势怎样？"),
    ("Q8-state", "2017年哪个州的销售额最高？交付准时率是多少？哪种支付方式最受欢迎？"),
]

for label, q in tests:
    t0 = time.time()
    r = analyze(q)
    elapsed = time.time() - t0
    strategy = r.get("strategy", "?")
    rows = len(r.get("data", []))
    err = r.get("error", "")
    status = "PASS" if not err and rows > 0 else "FAIL"
    print(f"[{status}] {label}: strategy={strategy} | {elapsed:.1f}s | {rows} rows")
    if err:
        print(f"  ERROR: {err[:150]}")
