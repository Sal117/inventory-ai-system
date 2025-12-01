# src/data_loader.py

import os
import pandas as pd
import numpy as np

DATA_DIR = "data"


class DataLoader:
    """
    UNIVERSAL SMART DATA LOADER (Option A + B)
    -----------------------------------------
    - Accepts ANY CSV (manual upload or sample)
    - Auto-detects dataset type (sales, inventory, items, promotions, supplier…)
    - Auto-detects item/date/quantity/stock columns
    - Cleans messy names, missing dates, duplicates
    - Returns a safe dataset dictionary for the dashboard
    """

    # ============================================================
    # INIT
    # ============================================================
    def __init__(self, base_dir=DATA_DIR):
        self.base_dir = base_dir

        # Possible column alias families
        self.date_aliases = ["date", "day", "sales_date", "created_at", "timestamp"]
        self.item_aliases = ["item", "item_name", "product", "sku", "product_name"]
        self.qty_aliases = ["quantity", "qty", "sales", "sold", "units"]
        self.stock_aliases = ["current_stock", "stock", "inventory", "on_hand", "ending_stock"]
        self.cost_aliases = ["unit_cost", "cost", "price"]
        self.supplier_aliases = ["supplier", "vendor", "brand"]
        self.category_aliases = ["category", "group", "type"]
        self.delay_aliases = ["avg_delay_days", "delay", "lead_time"]

    # ============================================================
    # INTERNAL COLUMN FINDER
    # ============================================================
    def _find_column(self, df, alias_list):
        cols = df.columns
        lower = [c.lower().strip() for c in cols]

        for alias in alias_list:
            if alias in lower:
                return cols[lower.index(alias)]
        return None

    # ============================================================
    # UNIVERSAL CSV CLEANER
    # ============================================================
    def clean_csv(self, df):
        if df.empty:
            return df

        # Normalize column names
        df = df.copy()
        df.rename(columns={c: c.lower().strip() for c in df.columns}, inplace=True)

        # ------------ Detect item ------------
        item_col = self._find_column(df, self.item_aliases)
        if item_col:
            df.rename(columns={item_col: "item"}, inplace=True)
            df["item"] = df["item"].astype(str)
        else:
            df["item"] = "Unknown_Item"

        # ------------ Detect date ------------
        date_col = self._find_column(df, self.date_aliases)
        if date_col:
            df.rename(columns={date_col: "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            # If no date → create a fake sequential date
            df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="D")

        df = df.dropna(subset=["date"])

        # ------------ Detect quantity ------------
        qty_col = self._find_column(df, self.qty_aliases)
        if qty_col:
            df.rename(columns={qty_col: "quantity"}, inplace=True)
        else:
            df["quantity"] = 0

        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)

        # ------------ Detect stock ------------
        stock_col = self._find_column(df, self.stock_aliases)
        if stock_col:
            df.rename(columns={stock_col: "current_stock"}, inplace=True)
            df["current_stock"] = pd.to_numeric(df["current_stock"], errors="coerce").fillna(0)
        else:
            df["current_stock"] = np.nan  # Means "not inventory dataset"

        # ------------ Optional metadata ------------
        for alias, final in [
            (self.category_aliases, "category"),
            (self.supplier_aliases, "supplier"),
            (self.cost_aliases, "unit_cost"),
            (self.delay_aliases, "avg_delay_days"),
        ]:
            col = self._find_column(df, alias)
            if col:
                df.rename(columns={col: final}, inplace=True)
            if final not in df:
                df[final] = np.nan

        # ------------ Remove duplicates ------------
        df = df.drop_duplicates()

        return df.reset_index(drop=True)

    # ============================================================
    # CLASSIFY DATASET TYPE
    # ============================================================
    def classify_dataset(self, df):
        cols = df.columns

        has_item = "item" in cols
        has_date = "date" in cols
        has_qty = "quantity" in cols
        has_stock = "current_stock" in cols

        # ------------ SALES ------------
        if has_item and has_date and has_qty:
            return "sales"

        # ------------ INVENTORY ------------
        if has_item and has_stock:
            return "inventory"

        # ------------ ITEMS MASTER ------------
        if has_item and ("unit_cost" in cols or "category" in cols or "supplier" in cols):
            return "items"

        # ------------ PROMOTIONS (if has date but no qty) ------------
        if has_item and has_date and not has_qty:
            return "promotions"

        # ------------ SUPPLIER PERFORMANCE ------------
        if "avg_delay_days" in cols or "supplier" in cols:
            return "supplier"

        # ------------ PURCHASE ORDERS ------------
        if "expected_date" in cols or "received_date" in cols:
            return "purchase_orders"

        return "other"

    # ============================================================
    # LOAD USER CSV (Option B)
    # ============================================================
    def load_any_csv(self, file):
        try:
            df = pd.read_csv(file)
            dataset_type = self.classify_dataset(df)
            df = self.clean_csv(df)
            return df, dataset_type
        except Exception as e:
            print("❌ Error loading CSV:", e)
            return pd.DataFrame(), "other"

    # ============================================================
    # LOAD PROJECT SAMPLE DATA (Option A)
    # ============================================================
    def _load(self, filename):
        path = os.path.join(self.base_dir, filename)

        if not os.path.exists(path):
            return pd.DataFrame(), "missing"

        try:
            df = pd.read_csv(path)
            df = self.clean_csv(df)
            dataset_type = self.classify_dataset(df)
            return df, dataset_type
        except:
            return pd.DataFrame(), "missing"

    #
    # ============================================================
    # MAIN: LOAD ALL SAMPLE DATA (FIXED - ONLY 3 FILES)
    # ============================================================
    def load_all(self):
        """
        Fixed loader: loads ONLY the three required sample datasets:
        1. daily_sales.csv       → sales
        2. items.csv             → items
        3. inventory_levels.csv  → inventory

        Everything else is ignored to prevent errors.
        """

        output = {
            "sales": pd.DataFrame(),
            "inventory": pd.DataFrame(),
            "items": pd.DataFrame()
        }

        # ---- FINAL 3 REQUIRED FILES ----
        fixed_files = {
            "daily_sales.csv": "sales",
            "items.csv": "items",
            "inventory_levels.csv": "inventory",
        }

        for filename, target_key in fixed_files.items():
            path = os.path.join(self.base_dir, filename)

            if not os.path.exists(path):
                print(f"⚠ Missing: {filename}")
                continue

            try:
                df = pd.read_csv(path)

                # Clean using universal cleaner
                df = self.clean_csv(df)

                # Assign cleaned dataframe
                output[target_key] = df.reset_index(drop=True)

            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
                output[target_key] = pd.DataFrame()

        return output
