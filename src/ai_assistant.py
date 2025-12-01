# ============================================================
# ai_assistant.py 
# Advanced Inventory AI Agent (Hybrid LLM + ML + Analytics)
# ============================================================

import os
import json
import re
import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import streamlit as st
from groq import Groq

from src.agent_tools import AgentTools
from src.ml.ml_predictor import MLPredictor



# Load Groq API key safely (Streamlit Cloud or local fallback)
load_dotenv()

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")



class InventoryAIAgent:
    """
    SUPERCHARGED INVENTORY AI AGENT
    --------------------------------
    Features:
    - Hybrid pipeline: Prophet + XGB blended forecasts
    - Multi-item forecasting
    - Stockout prediction + reorder planning
    - Supply-chain explanation using LLM
    - Visualization: Demand curves, forecast curves (PNG Base64)
    - Trend detection
    - Spike/Anomaly detection
    - Memory of last item mentioned
    """

    def __init__(self, df_sales, inventory_df):
        if not GROQ_API_KEY:
            st.error("❌ GROQ_API_KEY is missing. Please add it in Streamlit Secrets.")
            raise ValueError("GROQ_API_KEY missing.")
            
        self.client = Groq(api_key=GROQ_API_KEY)

        # ML + classical logic tools
        self.tools = AgentTools(df_sales, inventory_df)
        self.predictor = MLPredictor()

        # Conversation memory
        self.memory = []
        self.last_item_mentioned = None      # store last used item
        self.last_forecast = {}              # store last forecast DF
        self.last_inventory_lookup = None    # optional

    # ========================================================
    # JSON SAFE SERIALIZER
    # ========================================================
    def _json_safe(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._json_safe(i) for i in obj]

        if isinstance(obj, np.generic):
            return obj.item()

        return obj

    # ========================================================
    # INTERNAL: LLM CALL (Groq)
    # ========================================================
    def _llm(self, messages):
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=350,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM ERROR] {str(e)}"
        
    def generate_insights(self, text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert analytics system. Provide clear, insightful, "
                    "actionable insights based only on the provided dataset summary."
                )
            },
            {"role": "user", "content": text}
        ]
        return self._llm(messages)
    

    # ========================================================
    # ITEM EXTRACTION (supports multi-item + memory fallback)
    # ========================================================
    def _extract_items(self, text):
        text = text.lower()
        found = []

        tokens = re.split(r"[ ,]+", text)

        for token in tokens:
            m = self.tools._match_item_name(token)
            if m:
                found.append(m)

        # last item fallback
        if not found and self.last_item_mentioned:
            found = [self.last_item_mentioned]

        return list(set(found))

    # ========================================================
    # INTENT DETECTION
    # ========================================================
    def _detect_intent(self, text):
        t = text.lower()

        # Forecast
        if "forecast" in t or "predict" in t or "next" in t:
            days_match = re.search(r"for\s+(\d+)\s*days", t) \
                      or re.search(r"(\d+)\s*day", t) \
                      or re.search(r"(\d+)\s*days", t)
            days = int(days_match.group(1)) if days_match else 30
            return "forecast_item", {"items": self._extract_items(t), "days": days}

        # Reorder / Restock
        if any(x in t for x in ["reorder", "restock", "order more", "stockout"]):
            return "get_reorder_report", {"items": self._extract_items(t)}

        # Inventory levels
        if any(x in t for x in ["inventory", "stock levels", "what do we have"]):
            return "get_inventory_levels", {}

        # Clean
        if "clean" in t:
            return "clean_sales_data", {"items": self._extract_items(t)}

        # Risk report
        if "risk" in t or "status" in t:
            return "get_full_risk_report", {}
        
        # Volatility / Stability / Fluctuations
        if any(x in t for x in ["volatility", "stable", "stability", "fluctuation", "variation"]):
            return "get_item_volatility", {"items": self._extract_items(t)}


        # No intent → fallback to LLM general chat
        return None, None

    # ========================================================
    # MEMORY COMPRESSION
    # ========================================================
    def _compress_memory(self):
        if len(self.memory) > 12:
            self.memory = self.memory[-8:]

    # ========================================================
    # VISUALIZATION (Matplotlib → Base64 PNG)
    # ========================================================
    def _plot_forecast_chart(self, df_hist, df_fc, item_name):
        """
        df_hist → historical df
        df_fc → forecast df (prophet + xgb blended)
        """

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

    # ========================================================
    # HIGH-LEVEL ASK METHOD
    # ========================================================
    def ask(self, user_input):

        self.memory.append({"role": "user", "content": user_input})
        self._compress_memory()

        tool_name, params = self._detect_intent(user_input)

        # If a tool was detected → execute it
        if tool_name:
            return self._run_tool(tool_name, params, user_input)

        # Else → general chat
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful inventory & supply-chain assistant. "
                    "Be concise and practical."
                )
            }
        ] + self.memory

        reply = self._llm(messages)
        self.memory.append({"role": "assistant", "content": reply})
        return reply
    
# ========================================================
#  Tool Execution & Advanced Logic
# ========================================================

    # ========================================================
    # NORMALIZE MULTI-ITEM BEHAVIOR
    # ========================================================
    def _normalize_tool_params(self, tool_name, params):

        single_item_tools = ["forecast_item", "get_reorder_report",
                            "clean_sales_data", "get_item_volatility"]

        # ❗ Only treat as multi-item if **2 or more** items are provided
        if tool_name in single_item_tools and "items" in params:
            items = params["items"]

            # 🟢 SINGLE ITEM → return normal params (NOT multi-item)
            if len(items) == 1:
                return {"item_name": items[0], **{k: v for k, v in params.items() if k != "items"}}, False

            # 🟠 MULTI-ITEM MODE
            safe_params = {k: v for k, v in params.items()
                        if k not in ["items", "item_name"]}

            results = []
            for item in items:
                tool_fn = getattr(self.tools, tool_name)
                output = tool_fn(item_name=item, **safe_params)
                results.append({item: output})

            return results, True

        return params, False


    # ========================================================
    # EXECUTE TOOL (Single-item OR Multi-item)
    # ========================================================
    def _run_tool(self, tool_name, params, user_input):

        print("🚀 TOOL TRIGGERED:", tool_name, "| params:", params)

        normalized, handled = self._normalize_tool_params(tool_name, params)

        # ----------------------------------------------------
        # MULTI-ITEM CASE (2 or more items)
        # ----------------------------------------------------
        if handled:
            safe_result = self._json_safe(normalized)

            # Extract a chart if ANY item contains chart_data
            chart_b64 = None
            for entry in safe_result:
                if isinstance(entry, dict):
                    inner = list(entry.values())[0]
                    if isinstance(inner, dict) and inner.get("chart_data"):
                        chart_b64 = inner["chart_data"]
                        break

            # Build small summaries for LLM (safe)
            small_summary = []
            for entry in safe_result:
                item_name = list(entry.keys())[0]
                item_data = entry[item_name]
                small_summary.append({
                    "item": item_name,
                    "next_day": item_data.get("next_day"),
                    "next_7_days": item_data.get("next_7_days"),
                    "next_30_days": item_data.get("next_30_days"),
                })

            messages = [
                {"role": "system",
                "content": "You are an expert inventory assistant. "
                            "Explain results clearly in simple terms."},
                {"role": "assistant",
                "content": f"TOOL_RESULT:\n{json.dumps(small_summary, indent=2)}"},
                {"role": "user", "content": "Explain this for a beginner."}
            ]

            explanation = self._llm(messages)

            return {
                "type": "tool_result",
                "explanation": explanation,
                "chart": chart_b64
            }



        #
        # ----------------------------------------------------
        # Normal single item execution  (SAFE STRUCTURED OUTPUT)
        # ----------------------------------------------------
        tool_fn = getattr(self.tools, tool_name)
        result = tool_fn(**normalized)

        # SPECIAL HANDLING: volatility returns plain text
        if tool_name == "get_item_volatility":
            explanation = self._format_volatility_response(result)
            return {
                "type": "tool_result",
                "explanation": explanation,
                "chart": None
            }

        # Save last used item name
        if "item_name" in normalized:
            self.last_item_mentioned = normalized["item_name"]

        # Errors
        if isinstance(result, dict) and result.get("error"):
            explanation = self._explain_error(user_input, result)
            return {
                "type": "tool_result",
                "explanation": explanation,
                "chart": None
            }

        # Convert data safely (history + forecast)
        safe_result = self._json_safe(result)

        # CHART HANDLING — use the chart already generated by AgentTools
        chart_b64 = safe_result.get("chart_data", None)

        # DEBUG
        if chart_b64:
            print("🔍 DEBUG — Using chart from AgentTools (len:", len(chart_b64), ")")
        else:
            print("🔍 DEBUG — No chart found in AgentTools result")

        

        
        # Build a small summary for LLM (DO NOT include full forecast or chart)
        summary_for_llm = {
            "item_name": safe_result.get("item_name"),
            "next_day": safe_result.get("next_day"),
            "next_7_days": safe_result.get("next_7_days"),
            "next_30_days": safe_result.get("next_30_days"),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert inventory assistant. "
                    "Explain the results clearly in simple terms."
                )
            },
            {
                "role": "assistant",
                "content": f"RESULT:\n{json.dumps(summary_for_llm, indent=2)}"
            },
            {"role": "user", "content": "Explain this for a beginner."}
        ]

        

        explanation = self._llm(messages)

        # ----------------------------------------------
        # Return structured output to dashboard
        # ----------------------------------------------
        return {
            "type": "tool_result",
            "explanation": explanation,
            "chart": chart_b64
        }
        
    # ========================================================
    # VOLATILITY FORMATTING HELPER
    # ========================================================
    def _format_volatility_response(self, result):
        if isinstance(result, dict) and result.get("error"):
            return f"❌ {result['message']}"

        item = result["item_name"]

        return f"""
📊 **Volatility Analysis for {item}**

- **Standard Deviation:** {result['std_dev']:.2f}  
- **Average Daily Demand:** {result['mean_quantity']:.2f}  
- **Min Quantity:** {result['min_quantity']}  
- **Max Quantity:** {result['max_quantity']}  
- **Rolling {result['window']}-day Volatility (first 10 points):**  
  {result['rolling_std'][:10]}

**Interpretation:**  
Items with higher volatility have inconsistent demand. Consider monitoring them closely and avoid aggressive overstocking unless supported by trends.
        """


    # ========================================================
    # TREND DETECTION (Last 14 days)
    # ========================================================
    def _detect_trend(self, df):
        if len(df) < 7:
            return "Not enough data for trend analysis."

        last = df.tail(14)
        slope = np.polyfit(range(len(last)), last["quantity"], 1)[0]

        if slope > 0.5:
            return "Demand is increasing."
        elif slope < -0.5:
            return "Demand is decreasing."
        return "Demand is stable."

    # ========================================================
    # SPIKE / ANOMALY DETECTION
    # ========================================================
    def _detect_spikes(self, df):
        if len(df) < 10:
            return []

        q1 = df["quantity"].quantile(0.25)
        q3 = df["quantity"].quantile(0.75)
        iqr = q3 - q1

        threshold = q3 + 2.2 * iqr
        spikes = df[df["quantity"] > threshold]

        return spikes[["date", "quantity"]].to_dict(orient="records")

    
    # ========================================================
    # ERROR EXPLANATION (LLM)
    # ========================================================
    def _explain_error(self, user_input, error_dict):
        msg = json.dumps(error_dict, indent=2)

        messages = [
            {
                "role": "system",
                "content": "Explain errors politely and give steps to fix it."
            },
            {"role": "assistant", "content": f"TOOL_ERROR:\n{msg}"},
            {"role": "user", "content": "Help me fix it."}
        ]

        reply = self._llm(messages)
        self.memory.append({"role": "assistant", "content": reply})
        return reply
