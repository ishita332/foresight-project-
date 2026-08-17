"""
FORESIGHT planning dashboard.

Run with:
    streamlit run app/dashboard.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FORESIGHT NorthBay Living", layout="wide")


@st.cache_data
def load_data():
    weekly = pd.read_csv("data/processed/weekly_features.csv", parse_dates=["date"])
    forecasts = pd.read_csv("data/processed/forecasts.csv", parse_dates=["date"])
    risk = pd.read_csv("data/processed/risk_scores.csv")
    sku_master = pd.read_csv("data/processed/sku_master.csv")
    return weekly, forecasts, risk, sku_master


st.title("FORESIGHT - Demand and Inventory Intelligence")
st.caption("NorthBay Living - Planning Dashboard")

try:
    weekly, forecasts, risk, sku_master = load_data()
except FileNotFoundError:
    st.error(
        "Processed data not found. Run the pipeline first:\n\n"
        "python src/pipeline.py\n"
        "python src/features.py\n"
        "python src/train_and_score.py"
    )
    st.stop()

if risk.empty:
    st.warning("No SKUs were scored. Check that train_and_score.py ran successfully.")
    st.stop()

st.sidebar.header("Filters")
categories = sorted(sku_master["category"].dropna().unique())
selected_cats = st.sidebar.multiselect("Category", categories, default=categories)
quadrants = sorted(risk["quadrant"].unique())
selected_quads = st.sidebar.multiselect("Risk quadrant", quadrants, default=quadrants)

filtered_risk = risk[risk["category"].isin(selected_cats) & risk["quadrant"].isin(selected_quads)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs tracked", len(risk))
c2.metric("At stockout risk", int((risk["quadrant"] == "Reorder now").sum()))
c3.metric("Overstocked", int((risk["quadrant"] == "Markdown / clear").sum()))
c4.metric("Total value at stake", f"Rs {risk['value_at_stake'].sum():,.0f}")

st.divider()

st.subheader("Decisioning view")
if filtered_risk.empty:
    st.info("No SKUs match the current filters.")
else:
    fig = px.scatter(
        filtered_risk, x="overstock_risk", y="stockout_risk", size="value_at_stake",
        color="quadrant", hover_data=["sku_id", "sku_name", "value_at_stake"],
        color_discrete_map={
            "Reorder now": "#d64545", "Markdown / clear": "#6a5acd",
            "Watch / volatile": "#e0a52c", "Healthy": "#3a9d55",
        },
        labels={"overstock_risk": "Overstock risk", "stockout_risk": "Stockout risk"},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Prioritised reorder / markdown list")
action_table = filtered_risk[filtered_risk["quadrant"] != "Healthy"].sort_values(
    "value_at_stake", ascending=False
)
if action_table.empty:
    st.success("No SKUs currently need action within these filters.")
else:
    st.dataframe(
        action_table[["sku_id", "sku_name", "category", "quadrant", "recommended_action",
                       "stockout_risk", "overstock_risk", "value_at_stake",
                       "on_hand_units", "on_order_units"]],
        use_container_width=True, hide_index=True,
    )

st.divider()

st.subheader("Forecast vs actual - single SKU")
sku_options = filtered_risk["sku_id"].tolist() if not filtered_risk.empty else risk["sku_id"].tolist()
if sku_options:
    chosen_sku = st.selectbox("Select a SKU", sku_options)

    hist = weekly[weekly["sku_id"] == chosen_sku].sort_values("date")
    fcast = forecasts[forecasts["sku_id"] == chosen_sku].sort_values("date")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist["date"], y=hist["units_sold"], mode="lines",
                               name="Actual demand", line=dict(color="#1f2937")))
    fig2.add_trace(go.Scatter(x=fcast["date"], y=fcast["forecast"], mode="lines",
                               name="Forecast", line=dict(color="#6a5acd", dash="dot")))
    fig2.update_layout(height=400, xaxis_title="Week", yaxis_title="Units")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No SKUs to display for the current filters.")
