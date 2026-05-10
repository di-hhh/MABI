import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llm_outputs import DataAnalystOutput
tests = [
    "SELECT * FROM olist_order_items_dataset",
    "SELECT * FROM olist_products",
    "SELECT * FROM olist_orders",
    "SELECT * FROM order_reviews_dataset",
    "SELECT * FROM olist_order_items",
]
for t in tests:
    d = DataAnalystOutput(strategy="view", sql=t, reasoning="")
    print(f"  {t}\n  -> {d.sql}\n")
