# ===============================================================
# src/ml/train_xgb.py
# PRODUCTION-STABLE VERSION (NaN-safe + Feature-order safe)
# ===============================================================

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score

MODEL_DIR = "models/xgb"
os.makedirs(MODEL_DIR, exist_ok=True)


# ===============================================================
# FEATURE BUILDING
# ===============================================================
def build_training_features(df_sales: pd.DataFrame):

    df = df_sales.sort_values(["item", "date"]).copy()
    rows = []

    for item in df["item"].unique():

        df_item = df[df["item"] == item].sort_values("date")
        q = df_item["quantity"].values
        N = len(df_item)

        if N < 8:
            continue  # Not enough data

        for i in range(N - 1):

            row = {
                "item": item,
                "date": df_item["date"].iloc[i],
                "quantity": q[i]
            }

            d = df_item["date"].iloc[i]

            # Time
            row["day_of_week"] = d.dayofweek
            row["is_weekend"] = int(d.dayofweek in [5, 6])
            row["month"] = d.month
            row["quarter"] = d.quarter

            # Lags
            for lag in [1, 7, 14, 30]:
                row[f"lag_{lag}"] = q[i - lag] if i - lag >= 0 else q[0]

            # Rolling averages
            row["ma_7"] = df_item["quantity"].iloc[max(0, i-7):i].mean()
            row["ma_14"] = df_item["quantity"].iloc[max(0, i-14):i].mean()
            row["ma_30"] = df_item["quantity"].iloc[max(0, i-30):i].mean()

            row["std_7"] = df_item["quantity"].iloc[max(0, i-7):i].std()
            row["std_14"] = df_item["quantity"].iloc[max(0, i-14):i].std()

            # Metadata
            metadata_cols = [
                "ending_stock", "lift_factor", "on_time_rate",
                "avg_delay_days", "category", "supplier",
                "unit_cost", "shelf_life_days"
            ]
            for col in metadata_cols:
                row[col] = df_item[col].iloc[i] if col in df_item else 0

            # Targets
            row["y_tomorrow"] = q[i + 1]
            row["y_next_7"] = q[i+1:i+8].sum() if i + 7 < N else np.nan
            row["y_next_30"] = q[i+1:i+31].sum() if i + 30 < N else np.nan

            row["reorder_flag"] = int(row["y_tomorrow"] > q[i] * 1.10)

            rows.append(row)

    df2 = pd.DataFrame(rows)
    df2 = df2.dropna(subset=["y_tomorrow"])

    return df2


# ===============================================================
# MODEL BUILDERS
# ===============================================================
def train_regressor(X, y):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )
    model.fit(X, y)
    return model


def train_classifier(X, y):
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=350,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )
    model.fit(X, y)
    return model


# ===============================================================
# TRAIN ALL (GLOBAL XGB MODELS)
# ===============================================================
def train_xgb_all(df_sales: pd.DataFrame):

    print("🔧 Building training dataset...")
    df = build_training_features(df_sales)

    if df.empty:
        return {"status": "error", "message": "Not enough data for training."}

    drop_cols = ["item", "date", "quantity",
                 "y_tomorrow", "y_next_7", "y_next_30", "reorder_flag"]

    feature_cols = sorted([c for c in df.columns if c not in drop_cols])

    # Save feature order
    with open(os.path.join(MODEL_DIR, "feature_order.json"), "w") as f:
        json.dump(feature_cols, f)

    df = df.sort_values("date")
    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    if train.empty or test.empty:
        print("⚠ Not enough data for split — using dummy models.")
        return {"status": "error", "message": "Dataset too small."}

    # ===============================================================
    # NEXT-DAY DEMAND
    # ===============================================================
    print("\n🔵 Training NEXT-DAY model...")
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    model_t = train_regressor(X_train, train["y_tomorrow"])
    pred_t = model_t.predict(X_test)
    print("   RMSE =", np.sqrt(mean_squared_error(test["y_tomorrow"], pred_t)))
    model_t.get_booster().set_attr(feature_names=",".join(feature_cols))
    model_t.save_model(f"{MODEL_DIR}/xgb_tomorrow.json")


    # ===============================================================
    # NEXT 7 DAYS
    # ===============================================================
    train7 = train.dropna(subset=["y_next_7"])
    test7 = test.dropna(subset=["y_next_7"])

    if len(train7) > 50 and len(test7) > 10:
        print("\n🔵 Training NEXT 7 DAYS model...")
        X_train7 = train7[feature_cols]
        X_test7 = test7[feature_cols]

        model_7 = train_regressor(X_train7, train7["y_next_7"])
        pred_7 = model_7.predict(X_test7)
        print("   RMSE =", np.sqrt(mean_squared_error(test7["y_next_7"], pred_7)))
        model_7.get_booster().set_attr(feature_names=",".join(feature_cols))
        model_7.save_model(f"{MODEL_DIR}/xgb_next7.json")
    else:
        print("⚠ Skipping NEXT-7 model (not enough data)")


    # ===============================================================
    # NEXT 30 DAYS
    # ===============================================================
    train30 = train.dropna(subset=["y_next_30"])
    test30 = test.dropna(subset=["y_next_30"])

    if len(train30) > 50 and len(test30) > 10:
        print("\n🔵 Training NEXT 30 DAYS model...")
        X_train30 = train30[feature_cols]
        X_test30 = test30[feature_cols]

        model_30 = train_regressor(X_train30, train30["y_next_30"])
        pred_30 = model_30.predict(X_test30)
        print("   RMSE =", np.sqrt(mean_squared_error(test30["y_next_30"], pred_30)))
        model_30.get_booster().set_attr(feature_names=",".join(feature_cols))
        model_30.save_model(f"{MODEL_DIR}/xgb_next30.json")
    else:
        print("⚠ Skipping NEXT-30 model (not enough data)")

    #
    # =============================================================== 
    # REORDER CLASSIFIER
    # ===============================================================
    print("\n🟠 Training REORDER CLASSIFIER...")

    model_r = train_classifier(X_train, train["reorder_flag"])
    pred_r = (model_r.predict_proba(X_test)[:, 1] > 0.5).astype(int)

    print("   F1 =", f1_score(test["reorder_flag"], pred_r))
    print("   Precision =", precision_score(test["reorder_flag"], pred_r))
    print("   Recall =", recall_score(test["reorder_flag"], pred_r))

    # -----------------------------------------------------------
    # 🔧 FIX: Embed feature names inside the booster
    # -----------------------------------------------------------
    booster_r = model_r.get_booster()
    booster_r.set_attr(feature_names=",".join(feature_cols))

    # Save model
    model_r.save_model(f"{MODEL_DIR}/xgb_reorder.json")

    print("\n🎉 XGB training completed successfully!")

    return {
        "status": "success",
        "models": [
            "xgb_tomorrow.json",
            "xgb_next7.json",
            "xgb_next30.json",
            "xgb_reorder.json",
            "feature_order.json"
        ]
    }
