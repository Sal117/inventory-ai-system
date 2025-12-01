import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_sales(
        n_items=50,
        start_date="2023-01-01",
        end_date="2024-12-31",
        seed=42
    ):
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    all_rows = []

    for i in range(1, n_items + 1):
        item_name = f"Item_{i:03d}"

        # -------------------------
        # SEASONALITY COMPONENTS
        # -------------------------
        base = np.random.randint(5, 40)                                    # base demand
        weekly_pattern = (np.sin(np.arange(len(dates)) * 2 * np.pi / 7) + 1) * 0.5
        monthly_pattern = (np.sin(np.arange(len(dates)) * 2 * np.pi / 30) + 1) * 0.3

        # -------------------------
        # RANDOM NOISE
        # -------------------------
        noise = np.random.normal(0, 2, len(dates))

        # -------------------------
        # PROMOTION SPIKES (3 random)
        # -------------------------
        promo = np.zeros(len(dates))
        promo_days = np.random.choice(len(dates), size=3, replace=False)

        for p in promo_days:
            for offset in range(3):
                if p + offset < len(dates):
                    promo[p + offset] += np.random.randint(15, 35)

        # -------------------------
        # FINAL QUANTITY
        # -------------------------
        quantity = base + weekly_pattern + monthly_pattern + noise + promo
        quantity = np.clip(quantity, 0, None).round().astype(int)

        for d, q in zip(dates, quantity):
            all_rows.append([d, item_name, q])

    df = pd.DataFrame(all_rows, columns=["date", "item", "quantity"])
    return df


def generate_inventory_levels(n_items=50):
    items = [f"Item_{i:03d}" for i in range(1, n_items + 1)]
    current_stock = np.random.randint(50, 300, size=n_items)

    inventory_df = pd.DataFrame({
        "item": items,
        "current_stock": current_stock
    })
    return inventory_df


if __name__ == "__main__":
    print("🟢 Generating synthetic test datasets...")

    df_sales = generate_synthetic_sales()
    df_sales.to_csv("data/synthetic_sales.csv", index=False)
    print("✔ Saved → data/synthetic_sales.csv")

    df_inv = generate_inventory_levels()
    df_inv.to_csv("data/inventory_levels.csv", index=False)
    print("✔ Saved → data/inventory_levels.csv")

    print("🎉 Test data generation complete!")
