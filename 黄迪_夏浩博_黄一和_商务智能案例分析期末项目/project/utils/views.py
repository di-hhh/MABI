"""Pre-aggregation materialized views — created as tables for MySQL compatibility.

All 6 views per Requirements.md §III. Refresh via rebuild_all_views().
Note: ym is backtick-quoted because it's a MySQL reserved word.
"""
import logging
from utils.db import execute_sql, execute_query

logger = logging.getLogger(__name__)

VIEW_DEFS = {
    "mv_monthly_sales": {
        "create": """
            CREATE TABLE mv_monthly_sales (
                ym VARCHAR(7) PRIMARY KEY,
                total_gmv DOUBLE,
                total_orders BIGINT,
                avg_basket DOUBLE,
                total_freight DOUBLE
            )
        """,
        "insert": """
            INSERT INTO mv_monthly_sales (ym, total_gmv, total_orders, avg_basket, total_freight)
            SELECT
                DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS ym,
                SUM(oi.price) AS total_gmv,
                COUNT(DISTINCT o.order_id) AS total_orders,
                ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2) AS avg_basket,
                SUM(oi.freight_value) AS total_freight
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY ym
            ORDER BY ym
        """,
    },
    "mv_state_sales": {
        "create": """
            CREATE TABLE mv_state_sales (
                ym VARCHAR(7),
                customer_state VARCHAR(8),
                total_gmv DOUBLE,
                total_orders BIGINT,
                unique_customers BIGINT,
                PRIMARY KEY (ym, customer_state)
            )
        """,
        "insert": """
            INSERT INTO mv_state_sales (ym, customer_state, total_gmv, total_orders, unique_customers)
            SELECT
                DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS ym,
                c.customer_state,
                SUM(oi.price) AS total_gmv,
                COUNT(DISTINCT o.order_id) AS total_orders,
                COUNT(DISTINCT c.customer_unique_id) AS unique_customers
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_status = 'delivered'
            GROUP BY ym, c.customer_state
            ORDER BY ym, c.customer_state
        """,
    },
    "mv_category_sales": {
        "create": """
            CREATE TABLE mv_category_sales (
                ym VARCHAR(7),
                product_category_english VARCHAR(256),
                total_gmv DOUBLE,
                total_orders BIGINT,
                avg_price DOUBLE,
                PRIMARY KEY (ym, product_category_english)
            )
        """,
        "insert": """
            INSERT INTO mv_category_sales (ym, product_category_english, total_gmv, total_orders, avg_price)
            SELECT
                DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS ym,
                COALESCE(t.product_category_name_english, p.product_category_name) AS product_category_english,
                SUM(oi.price) AS total_gmv,
                COUNT(DISTINCT o.order_id) AS total_orders,
                ROUND(AVG(oi.price), 2) AS avg_price
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
            LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
            WHERE o.order_status = 'delivered'
            GROUP BY ym, product_category_english
            ORDER BY ym, product_category_english
        """,
    },
    "mv_delivery_perf": {
        "create": """
            CREATE TABLE mv_delivery_perf (
                ym VARCHAR(7),
                customer_state VARCHAR(8),
                avg_delivery_days DOUBLE,
                on_time_rate DOUBLE,
                delayed_orders BIGINT,
                PRIMARY KEY (ym, customer_state)
            )
        """,
        "insert": """
            INSERT INTO mv_delivery_perf (ym, customer_state, avg_delivery_days, on_time_rate, delayed_orders)
            SELECT
                DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS ym,
                c.customer_state,
                ROUND(AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)), 1) AS avg_delivery_days,
                ROUND(
                    SUM(CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 1 ELSE 0 END)
                    / COUNT(*) * 100, 1
                ) AS on_time_rate,
                SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS delayed_orders
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_status = 'delivered'
              AND o.order_delivered_customer_date IS NOT NULL
              AND o.order_estimated_delivery_date IS NOT NULL
            GROUP BY ym, c.customer_state
            ORDER BY ym, c.customer_state
        """,
    },
    "mv_seller_perf": {
        "create": """
            CREATE TABLE mv_seller_perf (
                ym VARCHAR(7),
                seller_id VARCHAR(64),
                seller_state VARCHAR(8),
                total_gmv DOUBLE,
                total_orders BIGINT,
                avg_review_score DOUBLE,
                PRIMARY KEY (ym, seller_id)
            )
        """,
        "insert": """
            INSERT INTO mv_seller_perf (ym, seller_id, seller_state, total_gmv, total_orders, avg_review_score)
            SELECT
                DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS ym,
                s.seller_id,
                s.seller_state,
                SUM(oi.price) AS total_gmv,
                COUNT(DISTINCT o.order_id) AS total_orders,
                ROUND(AVG(r.review_score), 2) AS avg_review_score
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN sellers s ON oi.seller_id = s.seller_id
            LEFT JOIN order_reviews r ON o.order_id = r.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY ym, s.seller_id, s.seller_state
            ORDER BY ym, s.seller_id
        """,
    },
    "mv_payment_dist": {
        "create": """
            CREATE TABLE mv_payment_dist (
                ym VARCHAR(7),
                payment_type VARCHAR(32),
                total_transactions BIGINT,
                avg_installments DOUBLE,
                total_value DOUBLE,
                PRIMARY KEY (ym, payment_type)
            )
        """,
        "insert": """
            INSERT INTO mv_payment_dist (ym, payment_type, total_transactions, avg_installments, total_value)
            SELECT
                DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS ym,
                p.payment_type,
                COUNT(*) AS total_transactions,
                ROUND(AVG(p.payment_installments), 1) AS avg_installments,
                SUM(p.payment_value) AS total_value
            FROM orders o
            JOIN payments p ON o.order_id = p.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY ym, p.payment_type
            ORDER BY ym, p.payment_type
        """,
    },
}


def rebuild_all_views():
    """Drop and recreate all 6 pre-aggregation tables."""
    for name, sqls in VIEW_DEFS.items():
        logger.info("Rebuilding %s...", name)
        execute_sql(f"DROP TABLE IF EXISTS {name}")
        execute_sql(sqls["create"])
        execute_sql(sqls["insert"])
        count = execute_query(f"SELECT COUNT(*) AS cnt FROM {name}")
        logger.info("  %s -> %d rows", name, count[0]["cnt"] if count else 0)
    logger.info("All 6 pre-aggregation views rebuilt.")


def get_all_view_data():
    """Return dict of all view data for the system."""
    data = {}
    for name in VIEW_DEFS:
        try:
            data[name] = execute_query(f"SELECT * FROM {name}")
        except Exception:
            data[name] = []
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    rebuild_all_views()
