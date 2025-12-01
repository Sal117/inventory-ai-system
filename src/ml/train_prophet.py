# ===============================================================
# src/ml/train_prophet.py
# FINAL TRAINABLE VERSION (Used by Dashboard)
# ===============================================================

import os
import pickle
import pandas as pd
from prophet import Prophet

PROPHET_DIR = "models/prophet"
os.makedirs(PROPHET_DIR, exist_ok=True)


# =====================================================================
# INTERNAL — SAFE TRAINER
# =====================================================================
def _fit_prophet_model(df: pd.DataFrame):
    """
    Internal helper for fitting Prophet safely.
    """
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=4.0,
    )
    model.fit(df)
    return model


# =====================================================================
# TRAIN A SINGLE ITEM
# =====================================================================
def train_prophet_single(item_name: str, df_item: pd.DataFrame):
    """
    Train Prophet for a single item from user-uploaded sales data.
    df_item must contain: date, quantity
    """

    df = df_item.sort_values("date").copy()

    # Minimum data requirement
    if len(df) < 8:
        return {
            "status": "error",
            "msg": f"⚠ Not enough data to train Prophet for '{item_name}'. Need at least 8 rows."
        }

    # Prophet-friendly dataset
    df = df.rename(columns={"date": "ds", "quantity": "y"})
    df = df[["ds", "y"]].dropna()

    try:
        model = _fit_prophet_model(df)

        # Quick future test to ensure model is valid
        _ = model.predict(model.make_future_dataframe(periods=3))

        # Save model
        save_path = os.path.join(PROPHET_DIR, f"{item_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        return {
            "status": "success",
            "msg": f"✅ Prophet model trained for '{item_name}'",
            "path": save_path
        }

    except Exception as e:
        return {
            "status": "error",
            "msg": f"❌ Training failed for '{item_name}': {e}"
        }


# =====================================================================
# TRAIN ALL ITEMS
# =====================================================================
def train_prophet_all(df_sales: pd.DataFrame):
    """
    Train Prophet models for ALL items in user-uploaded sales data.
    Used by the dashboard when clicking:
        "Train ALL Models (Prophet + XGB)"
    """

    items = sorted(df_sales["item"].unique())
    results = []

    for item in items:
        df_item = df_sales[df_sales["item"] == item]
        res = train_prophet_single(item, df_item)
        results.append({"item": item, **res})

    return results
