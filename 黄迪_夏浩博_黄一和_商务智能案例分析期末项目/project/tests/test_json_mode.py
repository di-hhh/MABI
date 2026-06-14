import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm import chat

print("Testing json_mode=True...")
try:
    r = chat([{"role": "user", "content": 'Say {"hello": "world"} in JSON only'}],
             temperature=0.0, max_tokens=50, json_mode=True)
    print(f"json_mode=True OK: {r[:80]}")
except Exception as e:
    print(f"json_mode=True FAILED: {e}")

print("\nTesting json_mode=False...")
try:
    r = chat([{"role": "user", "content": 'Say {"hello": "world"} in JSON only'}],
             temperature=0.0, max_tokens=50, json_mode=False)
    print(f"json_mode=False OK: {r[:80]}")
except Exception as e:
    print(f"json_mode=False FAILED: {e}")
