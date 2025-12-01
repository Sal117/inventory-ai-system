# ===============================================================
# src/ml/ml_predictor.py
# FINAL HYBRID FORECASTER (Prophet + XGBoost + FeatureOrder)
# ===============================================================

import os
import json
import pickle
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor, XGBClassifier

MODEL_DIR = "models"
PROPHET_DIR = os.path.join(MODEL_DIR, "prophet")
XGB_DIR = os.path.join(MODEL_DIR, "xgb")

os.makedirs(PROPHET_DIR, exist_ok=True)
os.makedirs(XGB_DIR, exist_ok=True)


class MLPredictor:

    def __init__(self):
        self.prophet_models = {}
        self.feature_order = None

        self.xgb_tomorrow = None
        self.xgb_next7 = None
        self.xgb_next30 = None
        self.xgb_reorder = None

        self._load_models()

    # -----------------------------------------------------------
    # LOAD ALL MODELS + FEATURE ORDER
    # -----------------------------------------------------------
    def _load_models(self):

        # Load Prophet models
        for f in os.listdir(PROPHET_DIR):
            if f.endswith(".pkl"):
                item = f.replace(".pkl", "")
                with open(os.path.join(PROPHET_DIR, f), "rb") as fp:
                    self.prophet_models[item] = pickle.load(fp)

        # Load feature order
        feature_path = os.path.join(XGB_DIR, "feature_order.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                self.feature_order = json.load(f)

        # Safe loader for XGB models
        def load_xgb(path, model):
            if os.path.exists(path):
                try:
                    model.load_model(path)
                    return model
                except:
                    return None
            return None
        
        self.xgb_tomorrow = load_xgb(os.path.join(XGB_DIR, "xgb_tomorrow.json"), XGBRegressor())
        if self.xgb_tomorrow and self.feature_order:
            self.xgb_tomorrow.get_booster().feature_names = self.feature_order

        self.xgb_next7 = load_xgb(os.path.join(XGB_DIR, "xgb_next7.json"), XGBRegressor())
        if self.xgb_next7 and self.feature_order:
            self.xgb_next7.get_booster().feature_names = self.feature_order

        self.xgb_next30 = load_xgb(os.path.join(XGB_DIR, "xgb_next30.json"), XGBRegressor())
        if self.xgb_next30 and self.feature_order:
            self.xgb_next30.get_booster().feature_names = self.feature_order

        self.xgb_reorder = load_xgb(os.path.join(XGB_DIR, "xgb_reorder.json"), XGBClassifier())
        if self.xgb_reorder and self.feature_order:
            self.xgb_reorder.get_booster().feature_names = self.feature_order


    # -----------------------------------------------------------
    # CHECK IF PROPHET MODEL EXISTS
    # -----------------------------------------------------------
    def model_exists(self, item_name):
        return item_name in self.prophet_models


    # -----------------------------------------------------------
    # TRAIN SINGLE PROPHET MODEL
    # -----------------------------------------------------------
    def train_single_item(self, item_name, df_item):
        df = df_item.sort_values("date").copy()
        df = df.rename(columns={"date": "ds", "quantity": "y"})

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.5
        )
        model.fit(df)

        path = os.path.join(PROPHET_DIR, f"{item_name}.pkl")
        with open(path, "wb") as fp:
            pickle.dump(model, fp)

        self.prophet_models[item_name] = model


    # -----------------------------------------------------------
    # BUILD XGB FEATURE ROW (MATCHES TRAINING EXACTLY)
    # -----------------------------------------------------------
    def build_feature_row(self, df_item):
        df = df_item.sort_values("date").copy()
        last = df.iloc[-1]

        row = {}
        d = last["date"]

        # Time
        row["day_of_week"] = d.dayofweek
        row["is_weekend"] = int(d.dayofweek in [5, 6])
        row["month"] = d.month
        row["quarter"] = d.quarter

        # Lags
        for lag in [1, 7, 14, 30]:
            if len(df) > lag:
                row[f"lag_{lag}"] = df["quantity"].iloc[-lag]
            else:
                row[f"lag_{lag}"] = df["quantity"].mean()

        # Rolling
        row["ma_7"] = df["quantity"].tail(7).mean()
        row["ma_14"] = df["quantity"].tail(14).mean()
        row["ma_30"] = df["quantity"].tail(30).mean()
        row["std_7"] = df["quantity"].tail(7).std()
        row["std_14"] = df["quantity"].tail(14).std()

        # Metadata
        metadata_cols = [
            "ending_stock", "lift_factor", "on_time_rate",
            "avg_delay_days", "category", "supplier",
            "unit_cost", "shelf_life_days"
        ]
        for col in metadata_cols:
            row[col] = last.get(col, 0)

        # Convert to DataFrame
        df_row = pd.DataFrame([row]).fillna(0)

        # -----------------------------------------------------
        # FIX: enforce feature_order.json EXACT ORDER
        # -----------------------------------------------------
        if self.feature_order:
            for col in self.feature_order:
                if col not in df_row:
                    df_row[col] = 0
            df_row = df_row[self.feature_order]

        return df_row


    # -----------------------------------------------------------
    # HYBRID FORECAST (Prophet + XGB)
    # -----------------------------------------------------------
    def forecast_item(self, item_name, df_item, days=30):

        df_item = df_item.sort_values("date")

        if item_name not in self.prophet_models:
            return {"error": f"No Prophet model found for {item_name}"}

        model = self.prophet_models[item_name]

        # Prophet forecast
        future = model.make_future_dataframe(periods=days)
        prophet_fc = model.predict(future).tail(days)["yhat"].values

        # If no XGB models exist → return Prophet only
        if not self.feature_order or \
           self.xgb_tomorrow is None or \
           self.xgb_next7 is None or \
           self.xgb_next30 is None:

            return {
                "item_name": item_name,
                "next_day": float(prophet_fc[0]),
                "next_7_days": float(sum(prophet_fc[:7])),
                "next_30_days": float(sum(prophet_fc)),
                "daily_forecast": list(prophet_fc),
                "weights": {"prophet": 1.0, "xgb": 0.0}
            }

        # Build XGB row
        row = self.build_feature_row(df_item)

        # XGB predictions
        x1 = float(self.xgb_tomorrow.predict(row)[0])
        x7 = float(self.xgb_next7.predict(row)[0])
        x30 = float(self.xgb_next30.predict(row)[0])

        # Hybrid weighted average
        w_p = 0.65
        w_x = 0.35

        next_day = w_p * prophet_fc[0] + w_x * x1
        next_7 = w_p * sum(prophet_fc[:7]) + w_x * x7
        next_30 = w_p * sum(prophet_fc) + w_x * x30

        # Distribute 30-day forecast proportionally
        scale = next_30 / sum(prophet_fc) if sum(prophet_fc) > 0 else 1
        daily_fc = [p * scale for p in prophet_fc]

        return {
            "item_name": item_name,
            "next_day": float(next_day),
            "next_7_days": float(next_7),
            "next_30_days": float(next_30),
            "daily_forecast": daily_fc,
            "weights": {"prophet": w_p, "xgb": w_x}
        }


    # -----------------------------------------------------------
    # REORDER CLASSIFIER
    # -----------------------------------------------------------
    def predict_reorder(self, item_name, df_item):

        if self.xgb_reorder is None or not self.feature_order:
            return {
                "item_name": item_name,
                "reorder_probability": 0.2,
                "recommend_reorder": False
            }

        row = self.build_feature_row(df_item)
        prob = float(self.xgb_reorder.predict_proba(row)[0][1])

        return {
            "item_name": item_name,
            "reorder_probability": prob,
            "recommend_reorder": prob > 0.60
        }
