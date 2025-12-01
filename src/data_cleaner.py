# src/data_cleaner.py

import pandas as pd
import numpy as np


class DataCleaner:

    # ============================================================
    # CLEAN SALES DATA (stable + multi-item safe)
    # ============================================================
    def clean_sales(self, df):

        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "quantity", "item"])

        df = df.copy()

        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]

        date_col = next((c for c in df.columns if c in ["date", "day", "timestamp"]), None)
        qty_col = next((c for c in df.columns if c in ["quantity", "qty", "sales", "units"]), None)
        item_col = next((c for c in df.columns if c in ["item", "product", "item_name", "sku"]), None)

        if item_col is None:
            df["item"] = "Unknown_Item"
            item_col = "item"

        if qty_col is None:
            df["quantity"] = 0
            qty_col = "quantity"

        if date_col is None:
            df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
            date_col = "date"

        # Clean date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        # Clean quantity
        df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
        df = df[df[qty_col] >= 0]

        if len(df) > 15:
            limit = df[qty_col].quantile(0.99)
            df[qty_col] = np.minimum(df[qty_col], limit)

        # ---------------------------------------------------------
        # FIX: Group by item + date (keeps all items)
        # ---------------------------------------------------------
        df = df.groupby([item_col, date_col], as_index=False)[qty_col].sum()

        # ---------------------------------------------------------
        # Generate continuous date range per item
        # ---------------------------------------------------------
        frames = []

        for item in df[item_col].unique():

            df_item = df[df[item_col] == item].copy()
            df_item = df_item.sort_values(date_col)

            start = df_item[date_col].min()
            end = df_item[date_col].max()

            full = pd.date_range(start=start, end=end, freq="D")

            df_item = df_item.set_index(date_col).reindex(full)

            df_item.index.name = "date"
            df_item[item_col] = item
            df_item[qty_col] = df_item[qty_col].fillna(0)

            frames.append(df_item.reset_index())

        df = pd.concat(frames, ignore_index=True)

        # Rename final columns
        df.rename(columns={qty_col: "quantity", item_col: "item"}, inplace=True)

        # Metadata columns
        optional = [
            "ending_stock", "lift_factor", "on_time_rate",
            "avg_delay_days", "category", "supplier",
            "unit_cost", "shelf_life_days"
        ]

        for col in optional:
            if col not in df.columns:
                df[col] = 0

        return df

    # ============================================================
    # CLEAN INVENTORY
    # ============================================================
    def clean_inventory(self, df):

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        if "date" in df:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in ["starting_stock", "ending_stock", "current_stock", "stock", "on_hand"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                df[col] = df[col].clip(lower=0)

        return df
