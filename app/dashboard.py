"""
FORESIGHT — Demand & Inventory Intelligence Dashboard

Run:
    streamlit run app/dashboard.py
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FORESIGHT — NorthBay Living",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    weekly = pd.read_csv(
        "data/processed/weekly_features.csv",
        parse_dates=["date"]
    )

    forecasts = pd.read_csv(
        "data/processed/forecasts.csv",
        parse_dates=["date"]
    )

    risk = pd.read_csv(
        "data/processed/risk_scores.csv"
    )

    sku_master = pd.read_csv(
        "data/processed/sku_master.csv"
    )

    return weekly, forecasts, risk, sku_master


# ============================================================
# LOAD
# ============================================================

try:

    weekly, forecasts, risk, sku_master = load_data()

except FileNotFoundError:

    st.error(
        """
        Processed data not found.

        Run:

        `python src/pipeline.py --input data/raw/online_retail_II.csv`

        `python src/features.py`

        `python src/train_and_score.py`
        """
    )

    st.stop()


if risk.empty:

    st.warning(
        "No risk scores found. Run train_and_score.py first."
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

for col in [
    "stockout_risk",
    "overstock_risk",
    "value_at_stake",
    "on_hand_units",
    "on_order_units"
]:

    if col in risk.columns:

        risk[col] = pd.to_numeric(
            risk[col],
            errors="coerce"
        ).fillna(0)


# ============================================================
# MERGE CATEGORY FROM SKU MASTER
# ============================================================

if "category" not in risk.columns:

    category_map = (
        sku_master[
            ["sku_id", "category"]
        ]
        .drop_duplicates("sku_id")
    )

    risk = risk.merge(
        category_map,
        on="sku_id",
        how="left"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📦 FORESIGHT — Demand & Inventory Intelligence"
)

st.caption(
    "NorthBay Living · Planning Dashboard"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎯 Filters")

st.sidebar.caption(
    "Control what appears in the decisioning view."
)


# ============================================================
# CATEGORY OPTIONS
# ============================================================

categories = sorted(
    sku_master[
        "category"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[
        lambda x: x != ""
    ]
    .unique()
)


selected_categories = st.sidebar.multiselect(

    "📂 Category",

    options=categories,

    default=categories,

    placeholder="Choose category..."
)


# ============================================================
# QUADRANTS
# ============================================================

quadrants = [
    "Reorder now",
    "Watch / volatile",
    "Healthy",
    "Markdown / clear"
]


available_quadrants = [
    q
    for q in quadrants
    if q in risk["quadrant"].unique()
]


selected_quadrants = st.sidebar.multiselect(

    "⚠️ Risk quadrant",

    options=available_quadrants,

    default=available_quadrants,

    placeholder="Choose quadrant..."
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_risk = risk.copy()


if selected_categories:

    filtered_risk = filtered_risk[
        filtered_risk[
            "category"
        ]
        .astype(str)
        .str.strip()
        .isin(selected_categories)
    ]


if selected_quadrants:

    filtered_risk = filtered_risk[
        filtered_risk[
            "quadrant"
        ]
        .isin(selected_quadrants)
    ]


# ============================================================
# SIDEBAR SUMMARY
# ============================================================

st.sidebar.divider()

st.sidebar.metric(
    "SKUs shown",
    f"{len(filtered_risk):,}"
)

st.sidebar.metric(
    "Total SKUs",
    f"{len(risk):,}"
)


# ============================================================
# KPI ROW
# ============================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "SKUs tracked",
    f"{len(risk):,}"
)


c2.metric(
    "At stockout risk",
    f"{(
        risk['quadrant'] == 'Reorder now'
    ).sum():,}"
)


c3.metric(
    "Overstocked",
    f"{(
        risk['quadrant'] == 'Markdown / clear'
    ).sum():,}"
)


c4.metric(
    "Total value at stake",
    f"£{risk['value_at_stake'].sum():,.0f}"
)


st.divider()


# ============================================================
# DECISIONING VIEW
# ============================================================

st.subheader(
    "Decisioning view"
)

st.caption(
    "Every SKU placed on a stockout vs overstock grid "
    "(bubble size = value at stake)"
)


if filtered_risk.empty:

    st.info(
        "No SKUs match the selected filters."
    )

else:

    # ========================================================
    # FIGURE
    # ========================================================

    fig = go.Figure()


    # ========================================================
    # COLORS
    # ========================================================

    colors = {

        "Reorder now":
            "#D95F5F",

        "Watch / volatile":
            "#D9A441",

        "Healthy":
            "#55A878",

        "Markdown / clear":
            "#7568C4"
    }


    # ========================================================
    # GLOBAL BUBBLE SCALE
    # ========================================================

    values = (
        filtered_risk[
            "value_at_stake"
        ]
        .fillna(0)
        .clip(lower=0)
    )


    size_base = (
        values
        .pow(0.5)
    )


    max_size = (
        size_base.max()
        if len(size_base)
        else 1
    )


    # ========================================================
    # ADD QUADRANT TRACES
    # ========================================================

    for quadrant in quadrants:

        df = filtered_risk[
            filtered_risk[
                "quadrant"
            ] == quadrant
        ].copy()


        if df.empty:

            continue


        # ----------------------------------------------------
        # Bubble size
        # ----------------------------------------------------

        bubble = (
            df[
                "value_at_stake"
            ]
            .fillna(0)
            .clip(lower=0)
            .pow(0.5)
        )


        if max_size > 0:

            bubble = (
                7 +
                (
                    bubble /
                    max_size
                ) * 35
            )

        else:

            bubble = 10


        # ----------------------------------------------------
        # Hover text
        # ----------------------------------------------------

        hover = []


        for _, row in df.iterrows():

            hover.append(

                f"<b>{row.get('sku_name', row['sku_id'])}</b>"
                f"<br>SKU: {row['sku_id']}"
                f"<br>Category: {row.get('category', '-')}"
                f"<br><br>"
                f"Stockout risk: "
                f"{row['stockout_risk']:.2f}"
                f"<br>"
                f"Overstock risk: "
                f"{row['overstock_risk']:.2f}"
                f"<br>"
                f"Value at stake: "
                f"£{row['value_at_stake']:,.0f}"
                f"<br>"
                f"On hand: "
                f"{row.get('on_hand_units', 0):,.0f}"
                f"<br>"
                f"On order: "
                f"{row.get('on_order_units', 0):,.0f}"
                f"<br><br>"
                f"<b>{row.get('recommended_action', '')}</b>"
            )


        # ----------------------------------------------------
        # TRACE
        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=df[
                    "overstock_risk"
                ],

                y=df[
                    "stockout_risk"
                ],

                mode="markers",

                name=quadrant,

                text=hover,

                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),

                marker=dict(

                    size=bubble,

                    color=colors[
                        quadrant
                    ],

                    opacity=0.72,

                    line=dict(
                        color="white",
                        width=1
                    )
                )
            )
        )


    # ========================================================
    # BACKGROUNDS
    # ========================================================

    backgrounds = [

        (
            0, 0.5,
            0.5, 1,
            "rgba(214,69,69,0.10)"
        ),

        (
            0.5, 1,
            0.5, 1,
            "rgba(224,165,44,0.10)"
        ),

        (
            0, 0.5,
            0, 0.5,
            "rgba(58,157,85,0.10)"
        ),

        (
            0.5, 1,
            0, 0.5,
            "rgba(106,90,205,0.10)"
        )
    ]


    for x0, x1, y0, y1, fill in backgrounds:

        fig.add_shape(

            type="rect",

            x0=x0,
            x1=x1,

            y0=y0,
            y1=y1,

            fillcolor=fill,

            line=dict(
                width=0
            ),

            layer="below"
        )


    # ========================================================
    # DIVIDERS
    # ========================================================

    fig.add_hline(

        y=0.5,

        line_dash="dash",

        line_color="gray",

        line_width=1
    )


    fig.add_vline(

        x=0.5,

        line_dash="dash",

        line_color="gray",

        line_width=1
    )


    # ========================================================
    # QUADRANT LABELS
    # ========================================================

    labels = [

        (
            0.25,
            0.94,
            "<b>REORDER NOW</b><br>"
            "<span style='font-size:10px'>"
            "high stockout · low overstock"
            "</span>",
            "#D64545"
        ),

        (
            0.75,
            0.94,
            "<b>WATCH / VOLATILE</b><br>"
            "<span style='font-size:10px'>"
            "high on both — investigate"
            "</span>",
            "#D49A28"
        ),

        (
            0.25,
            0.08,
            "<b>HEALTHY</b><br>"
            "<span style='font-size:10px'>"
            "no action needed"
            "</span>",
            "#3A9D55"
        ),

        (
            0.75,
            0.08,
            "<b>MARKDOWN / CLEAR</b><br>"
            "<span style='font-size:10px'>"
            "high overstock · low stockout"
            "</span>",
            "#6A5ACD"
        )
    ]


    for x, y, text, color in labels:

        fig.add_annotation(

            x=x,
            y=y,

            text=text,

            showarrow=False,

            align="center",

            font=dict(
                color=color,
                size=14
            )
        )


    # ========================================================
    # AXES
    # ========================================================

    fig.update_xaxes(

        title="Overstock risk →",

        range=[
            0,
            1
        ],

        dtick=0.2,

        showgrid=True,

        gridcolor=(
            "rgba(150,150,150,0.18)"
        ),

        zeroline=False
    )


    fig.update_yaxes(

        title="Stockout risk →",

        range=[
            0,
            1
        ],

        dtick=0.2,

        showgrid=True,

        gridcolor=(
            "rgba(150,150,150,0.18)"
        ),

        zeroline=False
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        height=620,

        margin=dict(
            l=60,
            r=30,
            t=20,
            b=60
        ),

        showlegend=False,

        plot_bgcolor="white",

        paper_bgcolor="white",

        hovermode="closest"
    )


    # ========================================================
    # PLOTLY TOOLBAR
    # ========================================================

    st.plotly_chart(

        fig,

        use_container_width=True,

        config={

            "displayModeBar": True,

            "displaylogo": False,

            "scrollZoom": True,

            "doubleClick": "reset",

            "responsive": True
        }
    )


st.divider()


# ============================================================
# PRIORITISED ACTION LIST
# ============================================================

st.subheader(
    "Prioritised reorder / markdown list"
)


action_table = (
    filtered_risk[
        filtered_risk[
            "quadrant"
        ] != "Healthy"
    ]
    .sort_values(
        "value_at_stake",
        ascending=False
    )
)


if action_table.empty:

    st.success(
        "No SKUs currently need action "
        "within these filters."
    )

else:

    columns = [

        "sku_id",
        "sku_name",
        "category",
        "quadrant",
        "recommended_action",
        "stockout_risk",
        "overstock_risk",
        "value_at_stake",
        "on_hand_units",
        "on_order_units"
    ]


    columns = [
        c
        for c in columns
        if c in action_table.columns
    ]


    st.dataframe(

        action_table[
            columns
        ],

        use_container_width=True,

        hide_index=True
    )


st.divider()


# ============================================================
# FORECAST VS ACTUAL
# ============================================================

st.subheader(
    "Forecast vs actual — single SKU"
)


sku_options = (
    filtered_risk[
        "sku_id"
    ]
    .dropna()
    .tolist()
)


if not sku_options:

    sku_options = (
        risk[
            "sku_id"
        ]
        .dropna()
        .tolist()
    )


if sku_options:

    chosen_sku = st.selectbox(

        "Select a SKU",

        sku_options
    )


    # --------------------------------------------------------
    # ACTUAL
    # --------------------------------------------------------

    hist = (
        weekly[
            weekly["sku_id"]
            == chosen_sku
        ]
        .sort_values(
            "date"
        )
    )


    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    fcast = (
        forecasts[
            forecasts["sku_id"]
            == chosen_sku
        ]
        .sort_values(
            "date"
        )
    )


    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    fig2 = go.Figure()


    fig2.add_trace(

        go.Scatter(

            x=hist["date"],

            y=hist["units_sold"],

            mode="lines",

            name="Actual demand",

            line=dict(
                color="#1f2937",
                width=2
            )
        )
    )


    if not fcast.empty:

        fig2.add_trace(

            go.Scatter(

                x=fcast["date"],

                y=fcast["forecast"],

                mode="lines",

                name="Forecast",

                line=dict(
                    color="#6A5ACD",
                    width=2,
                    dash="dot"
                )
            )
        )


    fig2.update_layout(

        height=420,

        xaxis_title="Week",

        yaxis_title="Units",

        hovermode="x unified",

        plot_bgcolor="white"
    )


    st.plotly_chart(

        fig2,

        use_container_width=True,

        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True
        }
    )

else:

    st.info(
        "No SKUs available."
    )