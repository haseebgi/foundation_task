"""
STEP 8: FINAL DASHBOARD
---------------------------
Streamlit-based interactive dashboard jo poore pipeline ke results dikhata hai:
  - Har stock ka Risk Score (0-100) aur Risk Level
  - Contributing factors ki explanation
  - Price history chart
  - Sab stocks ka side-by-side comparison

Chalane ka tareeqa (terminal mein):
    streamlit run step8_dashboard.py

NOTE: Ye currently synthetic/sample data par chal raha hai (kyunki is
sandbox mein Yahoo Finance access nahi hai). Real data ke liye
step1_data_collection.py apne computer par chalayein, phir Step 2-7
dobara run karein - dashboard automatically real data use karega.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from step6_predict_and_explain import load_artifacts, predict_risk
from step7_stock_comparison import build_comparison_table

DATA_PATH = "data/all_stocks_labeled.csv"

st.set_page_config(page_title="Stock Risk Analysis Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


@st.cache_resource
def get_model_artifacts():
    return load_artifacts()


def risk_color(level: str) -> str:
    return {"Low Risk": "#2ecc71", "Medium Risk": "#f1c40f", "High Risk": "#e74c3c"}.get(level, "#95a5a6")


def main():
    st.title("📊 AI-Powered Stock Risk Analysis Dashboard")
    st.caption("Beginner investors ke liye decision-support tool — ye financial advice nahi hai.")

    df = load_data()
    model, scaler, feature_columns, model_name = get_model_artifacts()
    stock_list = sorted(df["Symbol"].unique())

    # ---------------- Sidebar ----------------
    st.sidebar.header("Stock Select Karein")
    selected_stock = st.sidebar.selectbox("Stock", stock_list)
    st.sidebar.markdown(f"**Model in use:** {model_name}")
    st.sidebar.info("Data source: Real historical data (Yahoo Finance ke zariye). "
                     "Naye stocks add karne ke liye step1_data_collection.py mein STOCK_LIST update karein.")

    tab1, tab2 = st.tabs(["🔍 Single Stock Analysis", "⚖️ Compare All Stocks"])

    # ---------------- TAB 1: Single stock ----------------
    with tab1:
        result = predict_risk(selected_stock, df, model, scaler, feature_columns)

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{result['risk_score']}/100")
        col2.markdown(
            f"<h3 style='color:{risk_color(result['risk_level_predicted'])}'>"
            f"{result['risk_level_predicted']}</h3>",
            unsafe_allow_html=True,
        )
        col3.metric("Model Confidence", f"{result['confidence_pct']}%")

        st.subheader("Contributing Factors")
        for factor in result["explanation"]:
            st.write(f"- {factor}")

        st.subheader(f"{selected_stock} — Price History")
        stock_data = df[df["Symbol"] == selected_stock].sort_values("Date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"], name="Close Price", line=dict(color="#3498db")))
        fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["ma_50"], name="50-day MA", line=dict(color="#e67e22", dash="dash")))
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Score Trend Over Time")
        fig2 = px.line(stock_data, x="Date", y="risk_score", title=None)
        fig2.update_layout(height=300, yaxis_title="Risk Score (0-100)")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- TAB 2: Comparison ----------------
    with tab2:
        comparison_df = build_comparison_table()

        st.subheader("Sab Stocks ka Risk Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        fig3 = px.bar(
            comparison_df, x="Stock", y="Risk Score", color="Risk Level",
            color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f1c40f", "High Risk": "#e74c3c"},
            text="Risk Score",
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

        st.caption("⚠️ Ye system estimated risk information deta hai, guaranteed predictions ya profits nahi. "
                   "Final investment decision hamesha user ka hota hai.")


if __name__ == "__main__":
    main()
