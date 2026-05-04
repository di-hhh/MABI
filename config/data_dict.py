"""Data dictionary — base tables + pre-aggregation views for the Data Analyst Agent."""
# ── Base tables ──────────────────────────────────────────
BASE_TABLES = {
    "orders": {
        "description": "Core order records — one row per order",
        "columns": {
            "order_id": "Unique order identifier (PK)",
            "customer_id": "Foreign key to customers",
            "order_status": "Status: delivered, shipped, canceled, etc.",
            "order_purchase_timestamp": "Order placement timestamp",
            "order_approved_at": "Order approval timestamp",
            "order_delivered_carrier_date": "Date carrier received package",
            "order_delivered_customer_date": "Date delivered to customer",
            "order_estimated_delivery_date": "Estimated delivery date",
        },
    },
    "order_items": {
        "description": "Line items within each order — one row per product per order",
        "columns": {
            "order_id": "Foreign key to orders",
            "order_item_id": "Item sequence within order",
            "product_id": "Foreign key to products",
            "seller_id": "Foreign key to sellers",
            "shipping_limit_date": "Shipping deadline",
            "price": "Item sale price (BRL)",
            "freight_value": "Shipping cost (BRL)",
        },
    },
    "products": {
        "description": "Product catalog with physical attributes",
        "columns": {
            "product_id": "Unique product identifier (PK)",
            "product_category_name": "Category name in Portuguese",
            "product_name_length": "Product name character count",
            "product_description_length": "Description character count",
            "product_photos_qty": "Number of product photos",
            "product_weight_g": "Weight in grams",
            "product_length_cm": "Length in cm",
            "product_height_cm": "Height in cm",
            "product_width_cm": "Width in cm",
        },
    },
    "customers": {
        "description": "Customer master data",
        "columns": {
            "customer_id": "Unique customer identifier (PK)",
            "customer_unique_id": "Deduplicated customer key (one person may have multiple customer_ids)",
            "customer_zip_code_prefix": "Zip code prefix",
            "customer_city": "City name",
            "customer_state": "State abbreviation (e.g. SP, RJ)",
        },
    },
    "sellers": {
        "description": "Seller master data",
        "columns": {
            "seller_id": "Unique seller identifier (PK)",
            "seller_zip_code_prefix": "Zip code prefix",
            "seller_city": "City name",
            "seller_state": "State abbreviation",
        },
    },
    "payments": {
        "description": "Payment transactions per order",
        "columns": {
            "order_id": "Foreign key to orders",
            "payment_sequential": "Payment sequence number (1, 2, ...)",
            "payment_type": "credit_card, boleto, voucher, debit_card",
            "payment_installments": "Number of installments",
            "payment_value": "Payment amount (BRL)",
        },
    },
    "order_reviews": {
        "description": "Customer reviews after delivery",
        "columns": {
            "review_id": "Unique review identifier (PK)",
            "order_id": "Foreign key to orders",
            "review_score": "Rating 1-5",
            "review_comment_title": "Review title (Portuguese)",
            "review_comment_message": "Review body text (Portuguese)",
            "review_creation_date": "Review submission date",
            "review_answer_timestamp": "Seller response date",
        },
    },
    "geolocation": {
        "description": "Geographic coordinates for zip code prefixes",
        "columns": {
            "geolocation_zip_code_prefix": "Zip code prefix",
            "geolocation_lat": "Latitude",
            "geolocation_lng": "Longitude",
            "geolocation_city": "City name",
            "geolocation_state": "State abbreviation",
        },
    },
    "product_category_name_translation": {
        "description": "Portuguese → English category name translation",
        "columns": {
            "product_category_name": "Original Portuguese category name (PK)",
            "product_category_name_english": "English translation",
        },
    },
}

# ── Pre-aggregation views ─────────────────────────────────
MV_VIEWS = {
    "mv_monthly_sales": {
        "description": "Monthly sales summary — year-month grain",
        "grain": "ym",
        "columns": {
            "ym": "YYYY-MM format",
            "total_gmv": "Sum of price across delivered orders",
            "total_orders": "Distinct order count",
            "avg_basket": "Average order value (GMV/orders)",
            "total_freight": "Sum of freight charges",
        },
        "use_case": "Monthly sales trends, GMV growth, average basket analysis",
    },
    "mv_state_sales": {
        "description": "Monthly sales by customer state",
        "grain": "ym + customer_state",
        "columns": {
            "ym": "YYYY-MM format",
            "customer_state": "State abbreviation",
            "total_gmv": "Sum of price for state-month",
            "total_orders": "Order count for state-month",
            "unique_customers": "Unique customer count",
        },
        "use_case": "State sales rankings, regional market comparison",
    },
    "mv_category_sales": {
        "description": "Monthly sales by product category",
        "grain": "ym + product_category_english",
        "columns": {
            "ym": "YYYY-MM format",
            "product_category_english": "English category name",
            "total_gmv": "Sum of price for category-month",
            "total_orders": "Order count for category-month",
            "avg_price": "Average item price",
        },
        "use_case": "Category performance analysis, declining categories identification",
    },
    "mv_delivery_perf": {
        "description": "Monthly delivery performance by customer state",
        "grain": "ym + customer_state",
        "columns": {
            "ym": "YYYY-MM format",
            "customer_state": "State abbreviation",
            "avg_delivery_days": "Average days from purchase to delivery",
            "on_time_rate": "Percentage delivered on or before estimated date",
            "delayed_orders": "Count of late deliveries",
        },
        "use_case": "Delivery delay diagnostics, on-time rate analysis",
    },
    "mv_seller_perf": {
        "description": "Monthly seller performance metrics",
        "grain": "ym + seller_id",
        "columns": {
            "ym": "YYYY-MM format",
            "seller_id": "Seller identifier",
            "seller_state": "Seller's state",
            "total_gmv": "GMV generated by seller",
            "total_orders": "Order count for seller",
            "avg_review_score": "Average review score",
        },
        "use_case": "Seller performance monitoring, low-rated seller identification",
    },
    "mv_payment_dist": {
        "description": "Monthly payment method distribution",
        "grain": "ym + payment_type",
        "columns": {
            "ym": "YYYY-MM format",
            "payment_type": "Payment method",
            "total_transactions": "Transaction count",
            "avg_installments": "Average installment count",
            "total_value": "Total payment amount",
        },
        "use_case": "Payment preference analysis, installment rate comparison",
    },
}
