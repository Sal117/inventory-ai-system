# src/demand_forecaster.py

import pandas as pd
from src.ml.ml_predictor import MLPredictor
from src.data_cleaner import DataCleaner


class DemandForecaster:
    """
    Responsible ONLY for calling the hybrid AI model.
    No Prophet training, no Holt-Winters — all forecasting
    is done using the trained MLPredictor (Prophet + XGB blend).
    """

    def __init__(self):
        self.cleaner = DataCleaner()
        self.model = MLPredictor()

    # ---------------------------------------------------------
    # MAIN FORECAST INTERFACE (called by agent_tools)
    # ---------------------------------------------------------
    def forecast(self, df_raw, item_name, days=30):
        """
        Returns a hybrid AI forecast in Prophet-style format:
        [ { "ds": "2025-01-01", "yhat": 123 }, ... ]
        """

        # Clean daily data
        df = self.cleaner.clean_sales(df_raw)
        if df.empty:
            return pd.DataFrame({"ds": [], "yhat": []})

        # Call hybrid AI model
        results = self.model.forecast_item(item_name, df, days=days)

        # Convert blended values into a time-series DataFrame
        start_date = df["date"].max()

        forecast_dates = pd.date_range(start=start_date, periods=days + 1, freq="D")[1:]
        yhat_values = []

        # Use next_day, next_7, next_30 blended output
        # For daily values, distribute proportionally
        daily_forecast = results["next_day"]

        for i in range(days):
            yhat_values.append(float(daily_forecast))

        forecast_df = pd.DataFrame({
            "ds": forecast_dates.astype(str),
            "yhat": yhat_values
        })

        return forecast_df

    # ---------------------------------------------------------
    # Convenience methods for agent
    # ---------------------------------------------------------
    def forecast_next_day(self, df_raw, item_name):
        df = self.cleaner.clean_sales(df_raw)
        return self.model.forecast_item(item_name, df, days=1)["next_day"]

    def forecast_next_7(self, df_raw, item_name):
        df = self.cleaner.clean_sales(df_raw)
        return self.model.forecast_item(item_name, df, days=7)["next_7_days"]

    def forecast_next_30(self, df_raw, item_name):
        df = self.cleaner.clean_sales(df_raw)
        return self.model.forecast_item(item_name, df, days=30)["next_30_days"]
