import sys, os
sys.path.append(os.path.abspath("src"))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import base64
import re


from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
#Qroq Model 
from src.ai_assistant import InventoryAIAgent

# Prophet + XGB training
from src.ml.train_prophet import train_prophet_all, train_prophet_single
from src.ml.train_xgb import train_xgb_all

# Hybrid predictor
from src.ml.ml_predictor import MLPredictor


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Inventory System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📦"
)
st.title("📦 AI Inventory Management System")  

loader = DataLoader()
cleaner = DataCleaner()


# ============================================================
# SIDEBAR NAVIGATION + THEMING
# ============================================================
st.sidebar.header("📌 Navigation")

theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)

if theme == "Dark":
    st.markdown("""
    <style>
        .stApp { background-color:#0E1117 !important; color:#FAFAFA !important; }
        .css-1n76uvr, .css-1cypcdb { background-color:#262730 !important; }
        .stMarkdown, .stText, label { color:#FAFAFA !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background-color:#FFFFFF !important; color:#000000 !important; }
        .css-1n76uvr, .css-1cypcdb { background-color:#F2F2F2 !important; }
        .stMarkdown, .stText, label { color:#000000 !important; }
    </style>
    """, unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    [
        "🏠 Start",
        "📊 Data Explorer",
        "📈 Forecasting",
        "🧮 Reorder Intelligence",
        "🧠 Analytics",
        "🤖 AI Assistant",
        "📤 Export Reports",
        "ℹ About"
    ]
)


# ============================================================
# 1. START PAGE
# ============================================================
if page == "🏠 Start":
    st.markdown("""
<style>

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blink {
    50% { border-color: transparent; }
}

.intro-container {
    font-family: 'Segoe UI', sans-serif;
    color: white;
    background: linear-gradient(135deg, #0e1217 0%, #17202a 100%);
    padding: 35px;
    border-radius: 16px;
    animation: fadeInUp 1.0s ease-out;
    margin-bottom: 30px;
}

.intro-title {
    font-size: 32px;
    font-weight: bold;
    white-space: nowrap;
    overflow: hidden;
    width: 0;
    border-right: 2px solid #fff;
    animation: typing 3s steps(30, end), blink .7s step-end infinite alternate;
}

.intro-subtitle {
    font-size: 18px;
    margin-top: 20px;
    opacity: 0;
    animation: fadeInUp 1.2s ease-out 3s forwards;
}

</style>

<div class="intro-container">
    <div class="intro-title">AI Inventory Management System</div>
    <div class="intro-subtitle">
        Smart forecasting, real-time stock risks, and powerful AI-generated insights.  
        Load your data or use sample data to get started.
    </div>
</div>
""", unsafe_allow_html=True)
    
    

    st.markdown("""
Welcome to your **AI-powered inventory management system**!  
This tool helps you forecast demand, detect stock risks, generate reports,  
and make smarter purchasing decisions — automatically.

### 🔍 How to use the system

**1️⃣ Load Data**  
- Use **Sample Data** to explore the system instantly  
- Or upload your own files (Sales, Items, Inventory)

**2️⃣ Explore Your Data**  
Go to **📊 Data Explorer** to view cleaned tables, trends, and statistics.

**3️⃣ Forecast Future Demand**  
Use **📈 Forecasting** to generate 30-day hybrid predictions  
(Prophet + XGBoost) for any item.

**4️⃣ AI Reorder Intelligence**  
Go to **🧮 Reorder Intelligence** to get:
- Stockout risk
- Days until depletion
- Suggested reorder quantities
- AI-driven risk scoring

**5️⃣ Generate Reports**  
Download premium **PDF** and **Excel** reports from **📤 Export Reports**,  
including:
- Inventory Summary  
- Forecast Report  
- Reorder Risk Report  
- Volatility Report  

---

If you’re new:  
👉 Start by clicking **Load Sample Data** below to explore the system.
""")



   


    
    st.markdown("Choose data loading method:")

    #
    # -------------------------------------
    # OPTION A: LOAD SAMPLE DATA
    # -------------------------------------
    st.write("### 📂 Option A — Use Sample Dataset")

    if st.button("Load Sample Data"):
        dataset = loader.load_all()   # Now loads ONLY: sales, items, inventory

        # --- Clean datasets safely ---
        dataset["sales"] = cleaner.clean_sales(dataset["sales"])

        if dataset.get("inventory") is not None:
            dataset["inventory"] = cleaner.clean_inventory(dataset["inventory"])

        # Items require NO cleaning (they are metadata)
        # Items stay as-is
        # dataset["items"] = dataset["items"]

        st.session_state["dataset"] = dataset
        st.success("Sample data loaded! Go to Data Explorer →")
        st.stop()

    st.write("---")
    
    # -------------------------------------
    # OPTION B — MANUAL UPLOAD
    # -------------------------------------
    st.write("### 📤 Option B — Upload Your Own Files")

    sales_file = st.file_uploader("Upload *sales* CSV", type=["csv"])
    items_file = st.file_uploader("Upload *items* CSV", type=["csv"])
    inventory_file = st.file_uploader("Upload *inventory* CSV", type=["csv"])


    # ----------------------------------------------------
    # UNIVERSAL FILE NORMALIZER (Fixes tuple issue 100%)
    # ----------------------------------------------------
    def normalize_uploaded_file(upload_obj):
        """Always return a valid UploadedFile object or None."""
        if upload_obj is None:
            return None

        # Case 1: UploadedFile (normal)
        from streamlit.runtime.uploaded_file_manager import UploadedFile
        if isinstance(upload_obj, UploadedFile):
            return upload_obj

        # Case 2: Streamlit sometimes wraps it inside a tuple → (UploadedFile,)
        if isinstance(upload_obj, tuple) and len(upload_obj) > 0:
            if isinstance(upload_obj[0], UploadedFile):
                return upload_obj[0]

        # Case 3: List
        if isinstance(upload_obj, list) and len(upload_obj) > 0:
            if isinstance(upload_obj[0], UploadedFile):
                return upload_obj[0]

        # Unknown format → reject safely
        return None


    # Normalize each uploaded file 100%
    sales_file = normalize_uploaded_file(sales_file)
    items_file = normalize_uploaded_file(items_file)
    inventory_file = normalize_uploaded_file(inventory_file)


    # ----------------------------------------------------
    # SAFE DATAFRAME LOADING
    # ----------------------------------------------------
    def safe_load_csv(upload_obj):
        """Always return ONLY the dataframe (never a tuple)."""
        if upload_obj is None:
            return None

        try:
            df, dtype = loader.load_any_csv(upload_obj)   # Unpack correctly
            return df                                     # Only return DataFrame
        except Exception as e:
            st.error(f"❌ CSV Load Error: {e}")
            return None



    df_sales = safe_load_csv(sales_file)
    df_items = safe_load_csv(items_file)
    df_inventory = safe_load_csv(inventory_file)

    
    # ----------------------------------------------------
    # SAVE DATA (FINAL SAFE VERSION)
    # ----------------------------------------------------
    if st.button("Save Uploaded Data"):

        if df_sales is None:
            st.error("❌ You must upload at least the sales file.")
            st.stop()

        try:
            # CLEAN SALES FIRST (critical)
            df_sales_clean = cleaner.clean_sales(df_sales)

            # KEEP ITEMS AND INVENTORY SEPARATE
            df_inventory_clean = cleaner.clean_inventory(df_inventory) if df_inventory is not None else None

            # STORE RAW AND CLEAN
            st.session_state["dataset"] = {
                "sales": df_sales_clean,
                "raw_sales": df_sales,
                "items": df_items,
                "inventory": df_inventory_clean,
            }

            st.success("✅ Data uploaded successfully! Continue to Data Explorer →")

        except Exception as e:
            st.error(f"❌ Failed to process uploaded data: {e}")


    



# ============================================================
# 2. UNIVERSAL DATA EXPLORER (Shows ALL uploaded datasets)
# ============================================================
elif page == "📊 Data Explorer":
    if "dataset" not in st.session_state:
        st.warning("⚠ Load data first.")
        st.stop()

    ds = st.session_state["dataset"]

    st.subheader("📊 Data Explorer")

    # -----------------------------------------------------------
    # 1) Detect which datasets exist (sales/items/inventory)
    # -----------------------------------------------------------
    available = {
        "Sales": ds.get("sales"),
        "Items": ds.get("items"),
        "Inventory": ds.get("inventory"),
    }

    dataset_name = st.selectbox(
        "Choose dataset to explore:",
        [k for k, v in available.items() if v is not None and not v.empty]
    )

    df = available[dataset_name].copy()

    # -----------------------------------------------------------
    # 2) Remove useless columns (all NaN, all zeros, duplicates)
    # -----------------------------------------------------------
    def remove_useless_columns(df):
        clean_cols = []
        for col in df.columns:
            series = df[col]

            # Conditions to drop a column:
            if series.nunique() <= 1 and (series.isna().all() or (series.fillna(0) == 0).all()):
                continue  # skip empty/meaningless column

            clean_cols.append(col)

        return df[clean_cols]

    df_clean = remove_useless_columns(df)

    # -----------------------------------------------------------
    # 3) If dataset is Sales → show item picker & chart
    # -----------------------------------------------------------
    if dataset_name == "Sales":

        


        # Item selector
        items = sorted(df_clean["item"].unique())
        chosen_item = st.selectbox("Choose an item", items)

        df_item = df_clean[df_clean["item"] == chosen_item].copy()

        st.write("### 🧹 Cleaned Sales Data")
        st.dataframe(df_item.tail(30))

        # Sales chart
        st.write("### 📈 Sales Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_item["date"], y=df_item["quantity"],
            mode="lines+markers", name=chosen_item
        ))
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.write("### 📌 Summary Statistics")
        st.dataframe(df_item[["date", "quantity"]].describe())

    else:
        # -----------------------------------------------------------
        # 4) Non-sales dataset → just show clean table & stats
        # -----------------------------------------------------------
        st.write(f"### 📄 {dataset_name} Dataset Preview")
        st.dataframe(df_clean)

        # Show stats only for numeric columns
        numeric_cols = df_clean.select_dtypes(include=["int", "float"])
        if not numeric_cols.empty:
            st.write("### 📌 Dataset Statistics")
            st.dataframe(numeric_cols.describe())
        else:
            st.info("ℹ No numeric columns available for statistics.")




# ============================================================
# 3. FORECASTING
# ============================================================
elif page == "📈 Forecasting":
    if "dataset" not in st.session_state:
        st.warning("⚠ Load data first.")
        st.stop()

    ds = st.session_state["dataset"]
    df_sales = ds["sales"]

    st.subheader("📈 30-Day Hybrid Forecasting (Prophet + XGBoost)")

    # --------------------------------------------------------
    # Load predictor from cache
    # --------------------------------------------------------
    @st.cache_resource
    def load_predictor():
        return MLPredictor()

    predictor = load_predictor()

    # --------------------------------------------------------
    # ITEM SELECTION
    # --------------------------------------------------------
    items = sorted(df_sales["item"].unique())
    item = st.selectbox("Choose an item", items, label_visibility="visible")

    df_item = df_sales[df_sales["item"] == item].copy()

    if df_item.empty:
        st.error("❌ No data found for this item.")
        st.stop()

    # --------------------------------------------------------
    # TRAINING BUTTONS
    # --------------------------------------------------------
    st.write("### 🛠 Model Training")

    col1, col2 = st.columns(2)

    # Train Prophet model
    with col1:
        if st.button(f"Train Prophet for '{item}'"):
            with st.spinner("Training Prophet model..."):
                train_prophet_single(item, df_item)
                predictor._load_models()
            st.success(f"✅ Prophet model trained for {item}!")

    # Train ALL
    with col2:
        if st.button("Train ALL Models (Prophet + XGB)"):
            with st.spinner("Training models..."):
                train_prophet_all(df_sales)
                train_xgb_all(df_sales)
                predictor._load_models()
            st.success("✅ All Prophet + XGB models trained and loaded!")

    # --------------------------------------------------------
    # CHECK MODEL EXISTS
    # --------------------------------------------------------
    if not predictor.model_exists(item):
        st.warning("⚠ This item has no trained Prophet model yet. Please train it.")
        st.stop()

    # --------------------------------------------------------
    # RUN FORECAST
    # --------------------------------------------------------
    fc = predictor.forecast_item(item, df_item, days=30)

    if "error" in fc:
        st.error(fc["error"])
        st.stop()

    # --------------------------------------------------------
    # INSIGHT CARD
    # --------------------------------------------------------
    last_30 = df_item["quantity"].tail(30).mean().round(2)
    next_30 = fc["next_30_days"]
    trend_pct = ((next_30 - (last_30 * 30)) / (last_30 * 30)) * 100

    st.markdown("### 📊 Key Insights")
    st.info(f"""
    **📌 Expected Change (Next 30 Days):**
    - **{trend_pct:+.2f}%** compared to last month  
    - Average daily demand last 30 days: **{last_30} units**  
    - Forecast model weights → Prophet: `{fc["weights"]["prophet"]}`, XGB: `{fc["weights"]["xgb"]}`  
    """)

    # --------------------------------------------------------
    # FORECAST SUMMARY JSON
    # --------------------------------------------------------
    st.write("### 📌 Forecast Summary (Hybrid Model)")
    st.json({
        "Next Day": fc["next_day"],
        "Next 7 Days": fc["next_7_days"],
        "Next 30 Days": fc["next_30_days"],
        "Weights": fc["weights"],
    })

    # --------------------------------------------------------
    # DAILY FORECAST SERIES
    # --------------------------------------------------------
    future_dates = pd.date_range(
        start=df_item["date"].max() + pd.Timedelta(days=1),
        periods=30
    )

    daily_fc = fc["daily_forecast"]

    fc_df = pd.DataFrame({
        "date": future_dates,
        "forecast": daily_fc
    })

    # --------------------------------------------------------
    # Confidence intervals (from Prophet)
    # --------------------------------------------------------
    prophet_model = predictor.prophet_models[item]
    future_df = prophet_model.make_future_dataframe(periods=30)
    prophet_result = prophet_model.predict(future_df).tail(30)

    fc_df["lower"] = prophet_result["yhat_lower"]
    fc_df["upper"] = prophet_result["yhat_upper"]

    # --------------------------------------------------------
    # INVENTORY RISK ANALYSIS
    # --------------------------------------------------------
    st.markdown("### 🛑 Inventory Risk Analysis")

    last = df_item.sort_values("date").iloc[-1]

    ending_stock = last.get("ending_stock", 0)
    lead_time = last.get("avg_delay_days", 0)
    lift_factor = last.get("lift_factor", 1.0)

    next_7 = fc["next_7_days"]
    next_30 = fc["next_30_days"]

    # Volatility score
    volatility = df_item["quantity"].tail(14).std()

    # Calculate daily avg forecast
    daily_avg_fc = next_30 / 30

    # Depletion estimate
    if daily_avg_fc > 0:
        depletion_days = ending_stock / daily_avg_fc
    else:
        depletion_days = 999  # infinite

    # -------------------------------
    # Risk Scoring System (0 - 100)
    # -------------------------------
    risk = 0

    # High consumption risk
    if ending_stock < next_7:
        risk += 40
    elif ending_stock < next_7 * 1.2:
        risk += 25
    else:
        risk += 10

    # Lead time risk
    if lead_time > 10:
        risk += 25
    elif lead_time > 5:
        risk += 15
    else:
        risk += 5

    # Volatility risk
    if volatility > 20:
        risk += 20
    else:
        risk += 10

    # Lift factor risk (promotions or demand spikes)
    if lift_factor > 1.2:
        risk += 20

    risk = min(100, max(0, risk))

    # -------------------------------
    # Risk Label
    # -------------------------------
    if risk >= 75:
        risk_label = "🔴 HIGH RISK"
        risk_color = "danger"
    elif risk >= 45:
        risk_label = "🟠 MEDIUM RISK"
        risk_color = "warning"
    else:
        risk_label = "🟢 LOW RISK"
        risk_color = "success"

    # -------------------------------
    # DISPLAY RISK CARD
    # -------------------------------
    st.success(f"Inventory Risk Score: **{risk}/100** — {risk_label}")

    # -------------------------------
    # DEPLETION DATE
    # -------------------------------
    if depletion_days < 60:
        depletion_date = (df_item["date"].max() + pd.Timedelta(days=depletion_days)).date()
        st.warning(f"📉 Estimated Depletion Date: **{depletion_date}** ({depletion_days:.1f} days)")
    else:
        st.info("💼 Inventory is sufficient for the near future.")

    # -------------------------------
    # REORDER RECOMMENDATION
    # -------------------------------
    reorder_info = predictor.predict_reorder(item, df_item)

    prob = reorder_info["reorder_probability"]
    recommend = reorder_info["recommend_reorder"]

    if recommend:
        st.error(f"🚨 Reorder Recommended — Probability {prob:.2f}")
    else:
        st.info(f"Reorder Not Required — Probability {prob:.2f}")


    # --------------------------------------------------------
    # FORECAST CHART (Enhanced)
    # --------------------------------------------------------
    st.write("### 📈 Forecast Chart")

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=df_item["date"],
        y=df_item["quantity"],
        mode="lines",
        name="Actual",
        line=dict(color="blue")
    ))

    # Confidence Band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
        y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 150, 255, 0.1)',
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Range",
        hoverinfo="skip"
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fc_df["date"],
        y=fc_df["forecast"],
        mode="lines",
        name="Forecast",
        line=dict(color="orange")
    ))

    fig.update_layout(
        template="plotly_white",
        title=f"30-Day Forecast — {item}",
        xaxis_title="Date",
        yaxis_title="Quantity",
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, width="stretch")





# ============================================================
# 4.  REORDER INTELLIGENCE (AI-Driven + Enhanced UI)
# ============================================================
elif page == "🧮 Reorder Intelligence":

    if "dataset" not in st.session_state:
        st.warning("⚠ Load data first from the Start page.")
        st.stop()

    ds = st.session_state["dataset"]

    if "inventory" not in ds or ds["inventory"] is None:
        st.error("❌ Missing inventory dataset. Upload or include inventory_levels.csv")
        st.stop()

    df_sales = ds["sales"]
    inventory_df = ds["inventory"]

    st.subheader("📦 AI Reorder Intelligence")
    st.markdown("<span style='font-size:18px;'>Automatically analyzes <b>all items</b> using hybrid AI forecasting + AI risk scoring.</span>", unsafe_allow_html=True)
    st.markdown("---")

    @st.cache_resource
    def load_predictor():
        return MLPredictor()

    predictor = load_predictor()
    results = []

    # --------------------------------------------------------
    # PROCESS EACH ITEM (AI-driven analysis)
    # --------------------------------------------------------
    for item in inventory_df["item"].unique():

        df_item = df_sales[df_sales["item"] == item]
        if df_item.empty:
            continue

        df_clean = df_item.copy()

        # 1 — AI reorder probability from XGB
        pred = predictor.predict_reorder(item, df_clean)

        # 2 — AI next-7-day forecast
        fc = predictor.forecast_item(item, df_clean, days=7)
        avg_daily = fc["next_7_days"] / 7 if fc["next_7_days"] else 0

        # 3 — Current stock
        try:
            current_stock = float(inventory_df.loc[inventory_df["item"] == item, "current_stock"].iloc[0])
        except:
            current_stock = 0

        # 4 — Stockout estimate
        days_left = current_stock / avg_daily if avg_daily > 0 else 999

        # --------------------------------------------------------
        # AI RISK MODEL (Hybrid: XGB + Volatility + Demand Pressure)
        # --------------------------------------------------------
        xgb_prob = pred["reorder_probability"]

        # volatility normalized
        volatility = df_clean["quantity"].tail(14).std()
        vol_norm = min(1.0, volatility / (df_clean["quantity"].mean() + 1e-6))

        # demand pressure
        demand_pressure = (avg_daily * 7) / current_stock if current_stock > 0 else 1
        demand_norm = min(1, demand_pressure)

        # final AI risk score (0–100)
        ai_risk = int(
            (0.50 * xgb_prob * 100) +
            (0.25 * vol_norm * 100) +
            (0.25 * demand_norm * 100)
        )

        # Risk label (UI)
        if ai_risk >= 75:
            risk_label = "🔴 High"
        elif ai_risk >= 45:
            risk_label = "🟠 Medium"
        else:
            risk_label = "🟢 Low"

        # 5 — Reorder quantity recommendation (14-day target)
        suggested_qty = int(max(0, (avg_daily * 14) - current_stock))

        results.append({
            "Item": item,
            "Current Stock": int(current_stock),
            "Avg Daily Demand": round(avg_daily, 2),
            "Days Until Stockout": round(days_left, 1),

            "Stockout Date": (
                df_sales["date"].max() + pd.Timedelta(days=days_left)
            ).date() if days_left < 999 else "Sufficient",

            "Reorder Probability": round(xgb_prob, 4),
            "Recommend": "YES" if pred["recommend_reorder"] else "NO",

            "Risk Score": ai_risk,
            "Risk Level": risk_label,
            "Suggested Reorder Qty": suggested_qty
        })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("Risk Score", ascending=False)

    # --------------------------------------------------------
    # TOP CARDS — HIGHEST RISK ITEMS
    # --------------------------------------------------------
    st.write("### 🚨 Highest Risk Items")

    if not df_res.empty:
        top_items = df_res.head(3)
        colA, colB, colC = st.columns(3)
        cols = [colA, colB, colC]

        for col, (_, row) in zip(cols, top_items.iterrows()):
            item_name = row["Item"]
            risk_score = row["Risk Score"]      # ← REAL value
            recommend_flag = row["Recommend"]

            col.metric(
                label=item_name,
                value=f"{risk_score} / 100",
                delta=f"Reorder: {'YES' if recommend_flag=='YES' else 'NO'}"
            )

    st.markdown("---")

    # --------------------------------------------------------
    # MAIN TABLE
    # --------------------------------------------------------
    st.write("### 📊 Full Reorder Report (Ranked by Risk)")
    st.dataframe(df_res, use_container_width=True)

    # --------------------------------------------------------
    # HEATMAPS
    # --------------------------------------------------------
    st.write("### 🔥 Reorder Probability Heatmap")
    if not df_res.empty:
        heatmap_df1 = df_res.pivot_table(values="Reorder Probability", index="Item")
        st.dataframe(heatmap_df1.style.background_gradient(cmap="Reds"))

    st.write("### ⚡ Risk Score Heatmap")
    if not df_res.empty:
        heatmap_df2 = df_res.pivot_table(values="Risk Score", index="Item")
        st.dataframe(heatmap_df2.style.background_gradient(cmap="YlOrRd"))





# ============================================================
# 5. ANALYTICS DASHBOARD (ENHANCED)
# ============================================================
elif page == "🧠 Analytics":

    if "dataset" not in st.session_state:
        st.warning("⚠ Load data first.")
        st.stop()

    df = st.session_state["dataset"]["sales"].copy()
    st.subheader("🧠 Analytics Dashboard")
    st.markdown("A complete analytical breakdown of sales performance, seasonality, and item behavior.")

    # -----------------------------------------------------------
    # 1) KPI SUMMARY CARDS
    # -----------------------------------------------------------
    st.write("### 📌 Key Performance Indicators")

    total_sales = df["quantity"].sum()
    unique_items = df["item"].nunique()
    avg_daily = df.groupby("date")["quantity"].sum().mean().round(2)

    peak_day = df.groupby("date")["quantity"].sum().idxmax()
    peak_value = df.groupby("date")["quantity"].sum().max()

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Total Units Sold", f"{total_sales:,}")
    colB.metric("Unique Items", unique_items)
    colC.metric("Avg. Daily Volume", avg_daily)
    colD.metric("Peak Sales Day", f"{peak_value} units", delta=str(peak_day))

    st.markdown("---")

    # -----------------------------------------------------------
    # 2) MONTHLY SALES TREND
    # -----------------------------------------------------------
    st.write("### 📆 Monthly Sales Trend")

    df_month = df.copy()
    df_month["month"] = df_month["date"].dt.to_period("M")
    monthly = df_month.groupby("month")["quantity"].sum()

    fig_month = go.Figure()
    fig_month.add_trace(go.Scatter(
        x=monthly.index.astype(str),
        y=monthly.values,
        mode="lines+markers",
        name="Monthly Sales",
        line=dict(color="cyan")
    ))
    fig_month.update_layout(template="plotly_white")
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------
    # 3) ITEM CONTRIBUTION PIE CHART
    # -----------------------------------------------------------
    st.write("### 🥧 Item Contribution to Total Sales")

    item_totals = df.groupby("item")["quantity"].sum()

    fig_pie = go.Figure(
        go.Pie(
            labels=item_totals.index,
            values=item_totals.values,
            hole=0.4
        )
    )
    fig_pie.update_layout(template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------
    # 4) DEMAND VOLATILITY RANKING
    # -----------------------------------------------------------
    st.write("### ⚡ Demand Volatility (most unstable items)")

    vol_df = df.groupby("item")["quantity"].std().sort_values(ascending=False)
    vol_df = vol_df.rename("Volatility (Std Dev)")

    st.dataframe(vol_df.to_frame(), use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------
    # 5) ITEM-LEVEL DRILLDOWN
    # -----------------------------------------------------------
    st.write("### 🔍 Item-Level Analysis")

    items = sorted(df["item"].unique())
    selected_item = st.selectbox("Choose an item to explore:", items)

    item_df = df[df["item"] == selected_item].copy()

    col1, col2 = st.columns(2)

    # Month trend for item
    df_item_month = item_df.copy()
    df_item_month["month"] = df_item_month["date"].dt.to_period("M")
    item_month = df_item_month.groupby("month")["quantity"].sum()

    with col1:
        st.write(f"#### 📆 Monthly Trend — {selected_item}")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=item_month.index.astype(str),
            y=item_month.values,
            mode="lines+markers",
            line=dict(color="orange")
        ))
        fig1.update_layout(template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

    # Weekday pattern for item
    item_df["weekday"] = item_df["date"].dt.day_name()
    weekday_item = item_df.groupby("weekday")["quantity"].mean().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ])

    with col2:
        st.write(f"#### 📅 Weekday Pattern — {selected_item}")
        st.bar_chart(weekday_item)

    # Rolling volatility chart
    st.write(f"### 📈 Rolling Demand Volatility — {selected_item}")
    item_df_sorted = item_df.sort_values("date")
    item_df_sorted["rolling_std"] = item_df_sorted["quantity"].rolling(7).std()

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=item_df_sorted["date"],
        y=item_df_sorted["rolling_std"],
        mode="lines",
        line=dict(color="red"),
        name="7-Day Rolling Std"
    ))
    fig_vol.update_layout(template="plotly_white")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    
    # Def of Top Item and Weekly so we can made the chart and use them in AI insights and charts

    top_items = df.groupby("item")["quantity"].sum().sort_values(ascending=False)
    df_temp = df.copy()
    df_temp["weekday"] = df_temp["date"].dt.day_name()
    weekly = df_temp.groupby("weekday")["quantity"].mean().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ])
    
    # -----------------------------------------------------------
    # 5.5) AI-GENERATED INSIGHTS (Enhanced Version)
    # -----------------------------------------------------------

    st.write("### 🧠 AI Insights (Auto-Generated)")    

    try:
        # ----------------------------------------
        # PREPARE DATA FOR AI
        # ----------------------------------------
        # Format top items with percentages
        top_items_total = top_items.sum()
        top_items_formatted = {
            item: {
                "units": int(units),
                "percent": round((units / top_items_total) * 100, 2)
            }
            for item, units in top_items.head(5).items()
        }

        # Format volatility ranking
        volatility_formatted = {
            item: round(float(vol), 2)
            for item, vol in vol_df.head(5).items()
        }

        # Peak day formatted
        peak_day_str = str(peak_day) if hasattr(peak_day, "strftime") else str(peak_day)

        # Monthly trend clean (period index → string)
        monthly_dict = {str(k): int(v) for k, v in item_month.to_dict().items()}

        # Weekly performance (order preserved)
        weekly_dict = {day: round(float(val), 2) for day, val in weekly.items()}

        summary_payload = {
            "kpis": {
                "total_sales": int(total_sales),
                "unique_items": int(unique_items),
                "avg_daily_volume": float(avg_daily),
                "peak_sales_day": peak_day_str,
                "peak_sales_units": int(peak_value),
            },
            "top_items": top_items_formatted,
            "weekly_pattern": weekly_dict,
            "volatility_top": volatility_formatted,
            "selected_item": selected_item,
            "selected_item_weekday_pattern": weekday_item.to_dict(),
            "selected_item_monthly_trend": monthly_dict,
        }

        # ----------------------------------------
        # Clean and load into AI Agent
        # ----------------------------------------
        cleaner = DataCleaner()
        sales_clean = cleaner.clean_sales(df)
        inv_clean = cleaner.clean_inventory(st.session_state["dataset"].get("inventory"))
        ai_agent = InventoryAIAgent(sales_clean, inv_clean)

        # ----------------------------------------
        # Improved Prompt
        # ----------------------------------------
        prompt = f"""
    You are an expert inventory & sales analytics system.

    Using ONLY the structured data below, generate a **clear, professional, highly actionable insight report**.

    --- DATA SUMMARY (JSON) ---
    {json.dumps(summary_payload, indent=2)}
    --- END ---

    Your output MUST follow this structure:

    🔷 **1. Weekly & Monthly Demand Patterns**
    - Identify best/worst weekdays
    - Identify month-on-month direction (up/down/stable)
    - Comment on demand stability

    🔶 **2. Demand Spikes & Anomalies**
    - Identify unusual high/low days
    - Mention anomalies that require investigation

    🔷 **3. Key Revenue Drivers**
    - Highlight top-selling items
    - Mention revenue contribution %

    🔶 **4. High Volatility Items (Risk Alerts)**
    - List unstable items and why they matter
    - Suggest monitoring actions

    🔷 **5. Inventory Recommendations**
    Provide 3–5 practical, actionable steps:
    - Reorder suggestions
    - Stock balancing
    - Safety stock improvements
    - Forecast-driven decisions

    Style Guidelines:
    - Write in simple, business-friendly language
    - Avoid overly technical explanations
    - Keep it concise (8–12 bullets)
    - Use emojis (🔥 ⚠️ 📈 🟢) where appropriate
    """

        ai_text = ai_agent.generate_insights(prompt)

        # Pretty display
        st.markdown(f"""
    <div style="background-color:#1E2330; padding:18px; border-radius:10px; color:#E8E8E8;">
    {ai_text}
    </div>
    """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"⚠ AI Insights unavailable: {e}")


    # -----------------------------------------------------------
    # 6) CHARTS Of F TOP SELLING WEEKLY DAILY Qty
    # -----------------------------------------------------------
    #Top Item chart that were define earlier
    #Weekly def for chart

    pivot = df.pivot_table(
        index=df["date"].dt.date,
        columns="item",
        values="quantity",
        aggfunc="sum"
    )

    st.write("### 🔝 Top-Selling Items")
    st.bar_chart(top_items)

    st.write("### 📅 Weekly Sales Pattern")
    st.bar_chart(weekly)

    st.write("### 🔥 Daily Quantity Heatmap")
    st.dataframe(pivot.style.background_gradient(cmap="Blues"))




# ============================================================
# 6. AI ASSISTANT (Groq Llama - Enhanced UI/UX)
# ============================================================
elif page == "🤖 AI Assistant":

    if "dataset" not in st.session_state:
        st.warning("⚠ Load data first!")
        st.stop()

    ds = st.session_state["dataset"]

    if "inventory" not in ds or ds["inventory"] is None:
        st.error("❌ Missing inventory dataset.")
        st.stop()

    st.subheader("🤖 AI Inventory Assistant")
    st.markdown("""
    This assistant can help with:
    - 📈 Forecasting  
    - 📉 Inventory risk  
    - 🔁 Reorder recommendations  
    - 🛒 Stock levels  
    - 📊 Volatility & demand stability  
    - 🔍 Intelligent explanations  
    """)

    # ====================================================
    # CLEAN DATA
    # ====================================================
    sales_df = cleaner.clean_sales(ds["sales"])
    inventory_df = cleaner.clean_inventory(ds["inventory"])

    agent = InventoryAIAgent(sales_df, inventory_df)

    # ====================================================
    # PERSISTENT CHAT HISTORY
    # ====================================================
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    # ====================================================
    # USER INPUT
    # ====================================================
    st.write("### 💬 Ask the Assistant")
    user_input = st.text_input(
        "Ask anything (e.g., 'forecast apples 14 days', 'volatility juice', 'risk for bread'):",
        placeholder="Type your question here..."
    )

    # ====================================================
    # HANDLE USER MESSAGE
    # ====================================================
    if user_input:

        with st.spinner("🤖 Thinking..."):
            reply = agent.ask(user_input)

        # Save conversation
        st.session_state.ai_chat_history.append(("user", user_input))
        # Assistant structured reply (tool or plain chat)
        if isinstance(reply, dict) and reply.get("type") == "tool_result":

            print("🔍 DASHBOARD — Received tool_result:")
            print("   explanation length:", len(reply.get("explanation", "")))
            print("   chart exists:", reply.get("chart") is not None)
            print("   chart length:", len(reply["chart"]) if reply.get("chart") else 0)


            # Save assistant explanation as a normal chat bubble
            st.session_state.ai_chat_history.append(("assistant", reply["explanation"]))

            # Save chart in a separate chat entry
            chart_data = reply.get("chart", None)
            if chart_data is not None and isinstance(chart_data, str) and len(chart_data) > 0:
                st.session_state.ai_chat_history.append(("chart", chart_data))
                print("🔍 DASHBOARD — Saving chart to chat history (length:", len(chart_data), ")")
            else:
                print("⚠ DASHBOARD — Chart missing or empty, skipping display")


        else:
            # Normal LLM chat fallback (string)
            st.session_state.ai_chat_history.append(("assistant", reply))


    # ====================================================
    # DISPLAY CHAT HISTORY
    # ====================================================
    st.write("### 📝 Conversation")

    for role, text in st.session_state.ai_chat_history:
        # --- Handle chart messages cleanly ---
        if role == "chart":
            print("🖼️ CHAT LOOP — Rendering chart...")
            try:
                st.image(base64.b64decode(text), caption="📈 Forecast Chart")
                print("🖼️ SUCCESS — Chart rendered")
            except Exception as e:
                print("❌ ERROR — Chart failed to display:", e)
                st.warning(f"Chart could not be displayed: {e}")
            continue

        if role == "user":
            bubble_color = "#4A60DE"
            icon = "🧑‍💻"
        else:
            bubble_color = "#0B101C"
            icon = "🤖"

        st.markdown(
            f"""
            <div style="
                background:{bubble_color};
                padding:12px;
                border-radius:10px;
                margin-bottom:8px;
                border: 1px solid #3A3F4B;
            ">
                <b>{icon} {role.capitalize()}:</b><br>{text}
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")

    # ====================================================
    # QUICK INSIGHT PANELS (IF LAST AGENT REPLY CONTAINS KEYWORDS)
    # ====================================================
    if st.session_state.ai_chat_history:
        last_reply = st.session_state.ai_chat_history[-1][1]

        if any(x in last_reply.lower() for x in ["forecast", "volatility", "reorder", "risk"]):

            st.write("### 📌 Quick Insights Summary")

            cols = st.columns(4)

            # Forecast detection
            with cols[0]:
                if "day" in last_reply.lower():
                    st.metric("📈 Forecast Detected", "Yes", "")

            # Risk
            with cols[1]:
                if "risk" in last_reply.lower():
                    st.metric("⚠ Risk Mentioned", "Yes", "")

            # Reorder
            with cols[2]:
                if "reorder" in last_reply.lower():
                    st.metric("🔁 Reorder Advice", "Included", "")

            # Volatility
            with cols[3]:
                if "volatil" in last_reply.lower():
                    st.metric("📊 Volatility", "Analyzed", "")

            st.markdown(" ")

    # ====================================================
    # SUGGESTED PROMPTS
    # ====================================================
    st.write("### 💡 Suggested Questions")

    colA, colB = st.columns(2)

    with colA:
        if st.button("📈 Forecast Apples (14 days)"):
            st.session_state.ai_chat_history.append(("user", "forecast apples for 14 days"))
            st.rerun()

        if st.button("📊 Volatility of Juice"):
            st.session_state.ai_chat_history.append(("user", "show volatility for juice"))
            st.rerun()

    with colB:
        if st.button("⚠ Risk Level: Bread"):
            st.session_state.ai_chat_history.append(("user", "what is the risk level of bread"))
            st.rerun()

        if st.button("🔁 Should we reorder Eggs?"):
            st.session_state.ai_chat_history.append(("user", "should we reorder eggs"))
            st.rerun()

#
# ============================================================
# 7. EXPORT REPORTS (Enterprise Edition – Synced with Final Exporter)
# ============================================================
elif page == "📤 Export Reports":

    from src.report_exporter import ReportExporter

    # ---------------- SAFETY ----------------
    if "dataset" not in st.session_state:
        st.warning("⚠ Please load data first.")
        st.stop()

    ds = st.session_state["dataset"]
    df_sales = ds.get("sales")
    df_inventory = ds.get("inventory")

    if df_sales is None or df_sales.empty:
        st.error("❌ Sales dataset is empty.")
        st.stop()

    exporter = ReportExporter()

    # ---------------- PAGE HEADER ----------------
    st.title("📤 Export Reports")
    st.markdown("""
    Generate **enterprise-grade** reports in Excel or PDF:  
    - AI-cleaned data  
    - Auto-enrichment  
    - Hybrid fuzzy matching  
    - ML reorder predictions  
    - Advanced forecasting  
    """)

    # ============================================================
    #   REPORT TYPE SELECTION
    # ============================================================
    st.write("### 📁 Select Report Type")
    report_type = st.selectbox(
        "Choose a report:",
        [
            "Excel – All Cleaned Sales Data",
            "Excel – Inventory Table",
            "Excel – Demand Volatility Report",
            "PDF – Inventory Summary",
            "PDF – Forecast Report (Single Item)",
            "PDF – Reorder Risk Report",
        ]
    )

    # ============================================================
    #   PRE-ENRICH SALES ONCE (Very important!)
    # ============================================================
    enriched_sales = exporter.get_enriched_sales_df(df_sales)

    # ============================================================
    #   FORECAST OPTIONS
    # ============================================================
    if report_type == "PDF – Forecast Report (Single Item)":
        items = sorted(enriched_sales["item"].dropna().unique())
        selected_item = st.selectbox("Select Item:", items)
        forecast_days = st.slider("Forecast Days:", 7, 120, 30)

    st.markdown("---")

    # ============================================================
    #   PREVIEW PANEL
    # ============================================================
    st.write("### 👀 Preview")

    if report_type == "Excel – All Cleaned Sales Data":
        st.dataframe(enriched_sales.head(25))
        st.caption("ℹ️ Fully enriched dataset (exact content of Excel export).")

    elif report_type == "Excel – Inventory Table":
        st.dataframe(df_inventory.head(20))

    elif report_type == "Excel – Demand Volatility Report":
        st.info("📉 Includes Std Dev, Coefficient of Variation, Volatility Classes, and Trend Signals.")

    elif report_type == "PDF – Inventory Summary":
        preview_cols = [
            c for c in ["item", "current_stock", "reorder_point",
                        "recommended_reorder_qty", "status"]
            if c in df_inventory.columns
        ]
        st.dataframe(df_inventory[preview_cols].head(10))

    elif report_type == "PDF – Reorder Risk Report":
        preview_cols = [
            c for c in ["item", "current_stock", "reorder_point",
                        "recommended_reorder_qty", "status"]
            if c in df_inventory.columns
        ]
        st.info("⚠ AI-powered reorder urgency analysis.")
        st.dataframe(df_inventory[preview_cols].head(10))

    elif report_type == "PDF – Forecast Report (Single Item)":
        st.info(f"""
            📈 Includes:
            • {forecast_days}-day forecast  
            • Trend analysis  
            • ML-enhanced stability metrics  
            • Chart + Top-10 forecast table  
        """)
        st.dataframe(
            enriched_sales[enriched_sales["item"] == selected_item]
            .sort_values("date", ascending=False)
            .head(10)
        )

    st.markdown("---")

    # ============================================================
    #   GENERATE & DOWNLOAD SECTION
    # ============================================================
    st.write("### 📥 Generate Report")

    if st.button("📦 Generate & Download"):
        with st.spinner("📄 Generating your report..."):

            # ---------------- Excel: Enriched Sales ----------------
            if report_type == "Excel – All Cleaned Sales Data":
                file_bytes, filename = exporter.export_excel_bytes(
                    enriched_sales, filename="sales_export.xlsx"
                )

            # ---------------- Excel: Inventory (no enrichment) -------------
            elif report_type == "Excel – Inventory Table":
                file_bytes, filename = exporter.export_excel_bytes(
                    df_inventory, filename="inventory_export.xlsx"
                )

            # ---------------- Excel: Volatility --------------------
            elif report_type == "Excel – Demand Volatility Report":
                file_bytes, filename = exporter.export_volatility_excel_bytes(
                    enriched_sales
                )

            # ---------------- PDF: Inventory Summary --------------
            elif report_type == "PDF – Inventory Summary":
                file_bytes, filename = exporter.export_inventory_pdf_bytes(
                    df_inventory
                )

            # ---------------- PDF: Forecast (Single Item) ----------
            elif report_type == "PDF – Forecast Report (Single Item)":
                file_bytes, filename = exporter.export_forecast_pdf_bytes(
                    enriched_sales, df_inventory, selected_item, forecast_days
                )

            # ---------------- PDF: Reorder Risk --------------------
            elif report_type == "PDF – Reorder Risk Report":
                file_bytes, filename = exporter.export_reorder_pdf_bytes(
                    enriched_sales, df_inventory
                )

            else:
                st.error("❌ Unknown report type.")
                st.stop()

        # ------------- DOWNLOAD BUTTON -----------------
        st.download_button(
            label="⬇ Download Report",
            data=file_bytes,
            file_name=filename,
            type="primary",
            mime="application/octet-stream",
        )

        st.success("✅ Report generated successfully!")


# ============================================================
# 8. ABOUT (Humble, Refined & Human Edition — with Signature)
# ============================================================
elif page == "ℹ About":

    st.title("ℹ About This System")
    st.markdown("""
### **AI Inventory Management & Forecasting Dashboard**

This project was built to make inventory and demand forecasting **simpler, more intuitive, and more reliable**.  
Instead of juggling messy spreadsheets or guessing trends, this system brings everything together in one clear, automated workflow.

It’s not meant to be perfect — it's meant to be *helpful*.  
Something practical. Something that saves time.  
Something that grows and improves with each version.

---

## 🌱 **What This System Can Do**

### **1. Smarter Data Handling**
- Reads flexible CSV formats (no strict template required)  
- Automatically cleans, merges, and standardizes data  
- Handles missing values + messy item names  
- Designed to keep working even if your files aren’t ideal  

---

## 📈 **AI Forecasting (Prophet + XGBoost Hybrid)**
- Generates multi-day demand forecasts  
- Detects trends, spikes, anomalies  
- Captures weekly patterns  
- Produces clean, readable forecast charts  

The goal is simple: give you clarity about tomorrow.

---

## 🤖 **AI Assistant (Groq Llama-3)**
Your built-in AI teammate can:
- Answer questions in natural language  
- Run forecasts instantly  
- Explain patterns in simple terms  
- Highlight risks or reorder urgency  

It’s meant to feel supportive, not technical.

---

## 📊 **Analytics & Insights**
- Demand volatility  
- Low / Medium / High SKU classification  
- Inventory health indicators  
- Reorder urgency detection  
- Outlier discovery  

Insights that guide you — not overwhelm you.

---

## 📤 **Excel & PDF Reports**
- Clean enriched sales exports  
- Inventory summaries  
- Volatility analysis  
- Level-3 Forecast PDF with:
  - Trend analysis  
  - Confidence scoring  
  - Weekly pattern detection  
  - Peak demand days  
  - Anomaly checks  
  - Forecast charts  

Designed to look good when shared with managers or teammates.

---

## 🧩 **Tech Stack Behind the System**
- **Streamlit** — interactive dashboard  
- **Pandas / NumPy** — data cleaning & prep  
- **XGBoost + Prophet** — hybrid forecasting  
- **Groq Llama-3** — conversational AI  
- **Plotly / Matplotlib** — charts & visuals  
- **FPDF2 / OpenPyXL** — PDF & Excel exporting  

A mix of reliable tools built for clarity and performance.

---

## 🙏 **Why This Exists**
This system reflects:
- A desire to learn  
- A focus on simplicity  
- A belief that useful tools should be easy  
- A commitment to improving with every version  

It’s built with care — and built to grow.

---

## ✍️ **Created By**
**Salman Abdulhafiz Aljbrty**  
GitHub: [github.com/sal117](https://github.com/sal117)  
Email: **slmanaljbrty77@gmail.com**

I hope this tool gives anyone — beginner or expert — a clearer way to understand their data  
and a smoother inventory workflow.

---
    """)
