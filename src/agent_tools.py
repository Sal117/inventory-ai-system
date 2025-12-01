# src/agent_tools.py

import pandas as pd
import numpy as np
from difflib import get_close_matches
import matplotlib.pyplot as plt
import io, base64

from src.data_cleaner import DataCleaner
from src.ml.ml_predictor import MLPredictor


class AgentTools:

    def __init__(self, df_sales, inventory_df):

        self.df_sales = df_sales.copy()
        self.inventory_df = inventory_df.copy()

        self.cleaner = DataCleaner()
        self.predictor = MLPredictor()

        # Normalize item names
        self.df_sales["item"] = self.df_sales["item"].astype(str)
        self.inventory_df["item"] = self.inventory_df["item"].astype(str)

        self.items = list(self.inventory_df["item"].unique())

    # =====================================================================
    # SAFE ERROR WRAPPER
    # =====================================================================
    def _error(self, message: str):
        return {"error": True, "message": message}

    # =====================================================================
    # FUZZY ITEM NAME MATCHING
    # =====================================================================
    def _match_item_name(self, raw_name):

        if not raw_name:
            return None

        raw = raw_name.strip().lower()
        lower_items = [i.lower() for i in self.items]

        # 1. Exact match
        if raw in lower_items:
            return self.items[lower_items.index(raw)]

        # 2. Token match
        token_hits = [i for i in self.items if raw in i.lower()]
        if token_hits:
            return token_hits[0]

        # 3. Fuzzy
        close = get_close_matches(raw, lower_items, n=1, cutoff=0.55)
        if close:
            return self.items[lower_items.index(close[0])]

        return None

    # =====================================================================
    # VALIDATE + FETCH ITEM SALES
    # =====================================================================
    def _validate_item(self, item_name):

        if not item_name or not isinstance(item_name, str):
            return False, self._error("Item name missing or invalid.")

        matched = self._match_item_name(item_name)
        if not matched:
            return False, self._error(f"Item '{item_name}' not found.")

        df_item = self.df_sales[self.df_sales["item"] == matched]

        if df_item.empty:
            return False, self._error(f"No sales history available for '{matched}'.")

        return True, df_item

    # =====================================================================
    # CLEAN SALES HISTORY
    # =====================================================================
    def clean_sales_data(self, item_name):

        valid, df_item = self._validate_item(item_name)
        if not valid:
            return df_item

        df_clean = self.cleaner.clean_sales(df_item)
        return df_clean.to_dict(orient="records")
    
    # =====================================================================
    # INTERNAL: CREATE FORECAST PLOT (Matplotlib → Base64)
    # =====================================================================
    

    def _plot_forecast_chart(self, df_hist, df_fc, item_name):
        # ===========================
        # 🔥 DEBUG BLOCK — REQUIRED
        # ===========================
        print("🔧 DEBUG — Entered _plot_forecast_chart")
        print("   df_hist columns:", df_hist.columns.tolist())
        print("   df_fc columns:", df_fc.columns.tolist())

        # Check required columns
        print("   df_hist has 'date':", "date" in df_hist.columns)
        print("   df_hist has 'quantity':", "quantity" in df_hist.columns)
        print("   df_fc has 'ds':", "ds" in df_fc.columns)
        print("   df_fc has 'yhat':", "yhat" in df_fc.columns)
        print("   SAMPLE df_fc HEAD:")
        print(df_fc.head().to_string())
        print("====================================")
        
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(df_hist["date"], df_hist["quantity"], label="History", linewidth=2)
            plt.plot(df_fc["ds"], df_fc["yhat"], label="Forecast", linestyle="--")

            plt.title(f"Forecast for {item_name}")
            plt.xlabel("Date")
            plt.ylabel("Quantity")
            plt.legend()
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()

            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")

        except Exception:
            return None


    # =====================================================================
    # AI FORECASTING (Prophet + XGB Hybrid) + CHART SUPPORT (SAFE VERSION)
    # =====================================================================
    def forecast_item(self, item_name, days=30):

        valid, df_item = self._validate_item(item_name)
        if not valid:
            return df_item

        df_clean = self.cleaner.clean_sales(df_item)

        try:
            raw_result = self.predictor.forecast_item(item_name, df_clean, days=days)
        except Exception as e:
            return self._error(f"Forecasting failed: {str(e)}")

        # ----------------------------------------------------------
        # 🔥 CASE A — predictor returned a DataFrame (new models)
        # ----------------------------------------------------------
        if isinstance(raw_result, pd.DataFrame):
            forecast_df = raw_result

        # ----------------------------------------------------------
        # 🔥 CASE B — predictor returned a dict (old models)
        # ----------------------------------------------------------
        elif isinstance(raw_result, dict):
            # ========================================
            # 🔥 NEW: Convert old dict → DataFrame
            # ========================================
            if "daily_forecast" in raw_result:
                # Build a DataFrame with required columns
                forecast_df = pd.DataFrame({
                    "ds": pd.date_range(
                        start=df_item["date"].max() + pd.Timedelta(days=1),
                        periods=len(raw_result["daily_forecast"]),
                        freq="D"
                    ),
                    "yhat": raw_result["daily_forecast"]
                })
            else:
                # No usable data → return as-is
                return raw_result


        else:
            return self._error("Unsupported forecast format returned by predictor.")

        # ----------------------------------------------------------
        # Convert cleaned df_item (history) to dict
        # ----------------------------------------------------------
        history_records = df_item.to_dict(orient="records")

        # ----------------------------------------------------------
        # Convert forecast df to dict
        # ----------------------------------------------------------
        forecast_records = forecast_df.to_dict(orient="records")

        # ----------------------------------------------------------
        # Generate real chart from history + forecast
        # ----------------------------------------------------------
        chart_b64 = self._plot_forecast_chart(
            df_hist=df_item,
            df_fc=forecast_df,
            item_name=item_name
        )
        # 🔥 DEBUG: Is chart created?
        print("📊 DEBUG — chart_b64 exists:", chart_b64 is not None)
        if chart_b64:
            print("📊 DEBUG — chart_b64 length:", len(chart_b64))

        return {
            "item_name": item_name,
            "history": history_records,
            "forecast": forecast_records,
            "days": days,
            "chart_data": chart_b64
        }





    # =====================================================================
    # AI REORDER DECISION (ML Classifier)
    # =====================================================================
    def predict_reorder(self, item_name):

        valid, df_item = self._validate_item(item_name)
        if not valid:
            return df_item

        df_clean = self.cleaner.clean_sales(df_item)

        try:
            result = self.predictor.predict_reorder(item_name, df_clean)
        except Exception as e:
            return self._error(f"Reorder prediction failed: {str(e)}")

        return result

    # =====================================================================
    # COMBINED RISK REPORT (Forecast + Reorder)
    # =====================================================================
    def get_reorder_report(self, item_name):

        valid, df_item = self._validate_item(item_name)
        if not valid:
            return df_item

        df_clean = self.cleaner.clean_sales(df_item)

        try:
            forecast = self.predictor.forecast_item(item_name, df_clean, days=30)
            reorder = self.predictor.predict_reorder(item_name, df_clean)
        except Exception as e:
            return self._error(f"Failed generating report: {str(e)}")

        inv_row = self.inventory_df[self.inventory_df["item"] == item_name]
        current_stock = int(inv_row["current_stock"].iloc[0]) if not inv_row.empty else None

        return {
            "item_name": item_name,
            "current_stock": current_stock,
            "forecast": forecast,
            "reorder": reorder,
        }
    
    
    # =====================================================================
    # DEMAND VOLATILITY TOOL (Real Data)
    # =====================================================================
    def get_item_volatility(self, item_name, window=7):

        valid, df_item = self._validate_item(item_name)
        if not valid:
            return df_item

        df_item = df_item.sort_values("date")

        # Compute metrics
        try:
            std_dev = float(df_item["quantity"].std())
            rolling_std = df_item["quantity"].rolling(window).std().fillna(0).tolist()

            mean_qty = float(df_item["quantity"].mean())
            max_qty = int(df_item["quantity"].max())
            min_qty = int(df_item["quantity"].min())

        except Exception as e:
            return self._error(f"Volatility calculation failed: {str(e)}")

        return {
            "item_name": item_name,
            "std_dev": std_dev,
            "mean_quantity": mean_qty,
            "min_quantity": min_qty,
            "max_quantity": max_qty,
            "rolling_std": rolling_std,
            "window": window,
            "message": f"Volatility based on {window}-day rolling standard deviation."
        }

    # =====================================================================
    # LIST INVENTORY
    # =====================================================================
    def get_inventory_levels(self):
        return self.inventory_df.to_dict(orient="records")

    # =====================================================================
    # FULL SYSTEM RISK REPORT (ALL ITEMS)
    # =====================================================================
    def get_full_risk_report(self):

        reports = []
        for _, row in self.inventory_df.iterrows():
            item = row["item"]
            reports.append(self.get_reorder_report(item))

        return reports
