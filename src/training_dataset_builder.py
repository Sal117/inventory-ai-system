# src/training_dataset_builder.py

import pandas as pd
import numpy as np
import os

DATA_DIR = "data"


# ============================================================
# 1. LOAD ALL DATA (MATCHES YOUR REAL CSV HEADERS)
# ============================================================

def load_data():
    sales = pd.read_csv(os.path.join(DATA_DIR, "daily_sales.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    promotions = pd.read_csv(os.path.join(DATA_DIR, "promotions.csv"))
    supplier_perf = pd.read_csv(os.path.join(DATA_DIR, "supplier_performance.csv"))
    inventory_log = pd.read_csv(os.path.join(DATA_DIR, "inventory_log.csv"))
    purchase_orders = pd.read_csv(os.path.join(DATA_DIR, "purchase_orders.csv"))

    # Convert dates
    sales["date"] = pd.to_datetime(sales["date"])
    promotions["date"] = pd.to_datetime(promotions["date"])
    inventory_log["date"] = pd.to_datetime(inventory_log["date"])

    purchase_orders["expected_date"] = pd.to_datetime(purchase_orders["expected_date"])
    purchase_orders["received_date"] = pd.to_datetime(purchase_orders["received_date"])

    return sales, items, promotions, supplier_perf, inventory_log, purchase_orders


# ============================================================
# 2. FEATURE ENGINEERING (REALISTIC)
# ============================================================

def build_features(sales, items, promotions, supplier_perf, inventory_log):

    df = sales.copy()
    df = df.sort_values(["item_name", "date"])

    # -----------------------------------------
    # Time Features
    # -----------------------------------------
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # -----------------------------------------
    # Merge Item Attributes
    # -----------------------------------------
    df = df.merge(items, on="item_name", how="left")

    # -----------------------------------------
    # Merge Promotions
    # -----------------------------------------
    df = df.merge(promotions, on="date", how="left")
    df["lift_factor"] = df["lift_factor"].fillna(1.0)

    # -----------------------------------------
    # Supplier performance
    # -----------------------------------------
    df = df.merge(supplier_perf, on="supplier", how="left")
    df["on_time_rate"] = df["on_time_rate"].fillna(supplier_perf["on_time_rate"].mean())
    df["avg_delay_days"] = df["avg_delay_days"].fillna(supplier_perf["avg_delay_days"].mean())

    # -----------------------------------------
    # Inventory Log – use ending_stock
    # -----------------------------------------
    df = df.merge(
        inventory_log[["date", "item_name", "ending_stock"]],
        on=["date", "item_name"],
        how="left"
    )
    df["ending_stock"] = df["ending_stock"].ffill().fillna(0)

    # ============================================================
    # Lags & Rolling Statistics (per item)
    # ============================================================

    df["lag_1"] = df.groupby("item_name")["quantity"].shift(1)
    df["lag_7"] = df.groupby("item_name")["quantity"].shift(7)
    df["lag_14"] = df.groupby("item_name")["quantity"].shift(14)
    df["lag_30"] = df.groupby("item_name")["quantity"].shift(30)

    df["ma_7"] = df.groupby("item_name")["quantity"].rolling(7).mean().reset_index(0, drop=True)
    df["ma_14"] = df.groupby("item_name")["quantity"].rolling(14).mean().reset_index(0, drop=True)
    df["ma_30"] = df.groupby("item_name")["quantity"].rolling(30).mean().reset_index(0, drop=True)

    df["std_7"] = df.groupby("item_name")["quantity"].rolling(7).std().reset_index(0, drop=True)
    df["std_14"] = df.groupby("item_name")["quantity"].rolling(14).std().reset_index(0, drop=True)

    df.fillna(0, inplace=True)

    return df


# ============================================================
# 3. TARGET GENERATION (FORECAST LABELS)
# ============================================================

def generate_targets(df):

    df = df.sort_values(["item_name", "date"]).copy()

    df["y_tomorrow"] = df.groupby("item_name")["quantity"].shift(-1).fillna(0)

    df["y_next_7"] = (
        df.groupby("item_name")["quantity"].rolling(7).sum().shift(-7)
        .reset_index(0, drop=True).fillna(0)
    )

    df["y_next_30"] = (
        df.groupby("item_name")["quantity"].rolling(30).sum().shift(-30)
        .reset_index(0, drop=True).fillna(0)
    )

    return df


# ============================================================
# 4. REORDER FLAG (CLASSIFICATION TARGET)
# ============================================================

def add_reorder_flag(df, service_level=0.90):

    Z = 1.28  # 90% service level
    df["safety_stock"] = Z * df["std_14"] * np.sqrt(7)

    df["reorder_flag"] = (
        (df["y_next_7"] + df["safety_stock"]) > df["ending_stock"]
    ).astype(int)

    return df


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def build_training_dataset():

    sales, items, promotions, supplier_perf, inventory_log, purchase_orders = load_data()

    print("🔧 Building features…")
    df = build_features(sales, items, promotions, supplier_perf, inventory_log)

    print("🎯 Generating forecast labels…")
    df = generate_targets(df)

    print("📦 Computing reorder classification target…")
    df = add_reorder_flag(df)

    df = df.dropna()

    # ============================================================
    # FIX: Cast all categorical columns to string
    # ============================================================
    categorical_cols = ["promo_type", "category", "supplier", "item_name"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Save final dataset
    output_path = os.path.join(DATA_DIR, "training_data.parquet")
    df.to_parquet(output_path, index=False)

    print(f"✅ ML training dataset saved → {output_path}")


if __name__ == "__main__":
    build_training_dataset()
