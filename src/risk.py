"""
Stockout / Overstock Risk Scoring

Scores every SKU into one of four decisioning quadrants:

1. Reorder now
2. Watch / volatile
3. Healthy
4. Markdown / clear

The output is used by the Streamlit FORESIGHT dashboard.

Run directly:
    python src/risk.py

Or imported by:
    train_and_score.py
"""

import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

STOCKOUT_THRESHOLD = 0.50

OVERSTOCK_THRESHOLD = 2.50

FORWARD_WINDOW_WEEKS = 8


# ============================================================
# SCORE ONE SKU
# ============================================================

def score_sku(
    sku_id,
    forecast_df,
    latest_inventory,
    sku_master_row
):
    """
    Score one SKU.

    Parameters
    ----------
    sku_id : SKU identifier

    forecast_df : dataframe containing future forecast
                  for this SKU

    latest_inventory : latest inventory row for this SKU

    sku_master_row : master data row for this SKU

    Returns
    -------
    dict
    """

    # ========================================================
    # SAFELY READ INVENTORY
    # ========================================================

    on_hand = float(
        latest_inventory.get(
            "on_hand_units",
            0
        )
    )

    on_order = float(
        latest_inventory.get(
            "on_order_units",
            0
        )
    )

    lead_time_days = float(
        latest_inventory.get(
            "lead_time_days",
            7
        )
    )


    # ========================================================
    # CLEAN NEGATIVE VALUES
    # ========================================================

    on_hand = max(
        on_hand,
        0
    )

    on_order = max(
        on_order,
        0
    )

    lead_time_days = max(
        lead_time_days,
        1
    )


    # ========================================================
    # CLEAN FORECAST
    # ========================================================

    if forecast_df is None or forecast_df.empty:

        return None


    fcast = forecast_df.copy()


    if "forecast" not in fcast.columns:

        return None


    fcast["forecast"] = pd.to_numeric(
        fcast["forecast"],
        errors="coerce"
    ).fillna(0)


    # Demand cannot be negative
    fcast["forecast"] = (
        fcast["forecast"]
        .clip(lower=0)
    )


    forecast_values = (
        fcast["forecast"]
        .to_numpy()
    )


    if len(forecast_values) == 0:

        return None


    # ========================================================
    # LEAD TIME
    # ========================================================

    lead_time_weeks = max(
        lead_time_days / 7.0,
        1.0
    )


    # Number of forecast weeks to use
    lead_weeks = int(
        round(
            lead_time_weeks
        )
    )


    lead_weeks = max(
        lead_weeks,
        1
    )


    lead_weeks = min(
        lead_weeks,
        len(forecast_values)
    )


    # ========================================================
    # LEAD-TIME DEMAND
    # ========================================================

    lead_time_demand = float(
        forecast_values[
            :lead_weeks
        ].sum()
    )


    # ========================================================
    # FORWARD DEMAND
    # ========================================================

    forward_weeks = min(
        FORWARD_WINDOW_WEEKS,
        len(forecast_values)
    )


    forward_demand = float(
        forecast_values[
            :forward_weeks
        ].sum()
    )


    # ========================================================
    # AVAILABLE STOCK
    # ========================================================

    available_stock = (
        on_hand +
        on_order
    )


    # ========================================================
    # STOCK POSITION AT LEAD TIME
    # ========================================================

    projected_stock = (
        available_stock -
        lead_time_demand
    )


    # ========================================================
    # STOCKOUT RISK
    #
    # If available stock is much lower than expected
    # lead-time demand -> risk approaches 1.
    #
    # If available stock comfortably covers demand -> 0.
    # ========================================================

    if lead_time_demand <= 0:

        stockout_risk = 0.0

    else:

        stock_coverage_ratio = (
            available_stock /
            lead_time_demand
        )

        stockout_risk = (
            1.0 -
            stock_coverage_ratio
        )

        stockout_risk = np.clip(
            stockout_risk,
            0.0,
            1.0
        )


    stockout_risk = float(
        stockout_risk
    )


    # ========================================================
    # OVERSTOCK RISK
    #
    # Ratio:
    #
    # on-hand / 8-week demand
    #
    # 1.0 = approximately one planning window
    # 2.5 = very high inventory
    #
    # Risk scales from 0 to 1.
    # ========================================================

    if forward_demand <= 0:

        if on_hand > 0:

            overstock_risk = 1.0

        else:

            overstock_risk = 0.0

    else:

        inventory_multiple = (
            on_hand /
            forward_demand
        )

        overstock_risk = (
            inventory_multiple - 1.0
        ) / (
            OVERSTOCK_THRESHOLD - 1.0
        )

        overstock_risk = np.clip(
            overstock_risk,
            0.0,
            1.0
        )


    overstock_risk = float(
        overstock_risk
    )


    # ========================================================
    # QUADRANT CLASSIFICATION
    # ========================================================

    is_stockout_risk = (
        stockout_risk >=
        STOCKOUT_THRESHOLD
    )


    is_overstock_risk = (
        overstock_risk >=
        0.50
    )


    # --------------------------------------------------------
    # REORDER NOW
    # --------------------------------------------------------

    if (
        is_stockout_risk
        and not is_overstock_risk
    ):

        quadrant = "Reorder now"

        recommended_action = (
            "Raise a replenishment order "
            "before stock runs out."
        )


    # --------------------------------------------------------
    # MARKDOWN / CLEAR
    # --------------------------------------------------------

    elif (
        is_overstock_risk
        and not is_stockout_risk
    ):

        quadrant = "Markdown / clear"

        recommended_action = (
            "Promote or discount excess "
            "inventory to free up capital."
        )


    # --------------------------------------------------------
    # WATCH / VOLATILE
    # --------------------------------------------------------

    elif (
        is_stockout_risk
        and is_overstock_risk
    ):

        quadrant = "Watch / volatile"

        recommended_action = (
            "Investigate demand volatility "
            "and review inventory manually."
        )


    # --------------------------------------------------------
    # HEALTHY
    # --------------------------------------------------------

    else:

        quadrant = "Healthy"

        recommended_action = (
            "No restock needed; "
            "inventory position is healthy."
        )


    # ========================================================
    # VALUE AT RISK
    # ========================================================

    list_price = float(
        sku_master_row.get(
            "list_price",
            0
        )
    )


    unit_cost = float(
        sku_master_row.get(
            "unit_cost",
            0
        )
    )


    list_price = max(
        list_price,
        0
    )


    unit_cost = max(
        unit_cost,
        0
    )


    # ========================================================
    # SALES VALUE AT RISK
    # ========================================================

    shortage_units = max(
        lead_time_demand -
        available_stock,
        0
    )


    sales_at_risk_value = (
        shortage_units *
        list_price
    )


    # ========================================================
    # CAPITAL LOCKED IN EXCESS STOCK
    # ========================================================

    excess_units = max(
        on_hand -
        forward_demand,
        0
    )


    capital_locked_value = (
        excess_units *
        unit_cost
    )


    # ========================================================
    # TOTAL VALUE AT STAKE
    # ========================================================

    value_at_stake = max(
        sales_at_risk_value,
        capital_locked_value
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "sku_id": sku_id,

        "on_hand_units": round(
            on_hand,
            2
        ),

        "on_order_units": round(
            on_order,
            2
        ),

        "lead_time_days": round(
            lead_time_days,
            2
        ),

        "lead_time_demand": round(
            lead_time_demand,
            2
        ),

        "forward_demand": round(
            forward_demand,
            2
        ),

        "projected_stock_at_lead_time": round(
            projected_stock,
            2
        ),

        "stockout_risk": round(
            stockout_risk,
            3
        ),

        "overstock_risk": round(
            overstock_risk,
            3
        ),

        "quadrant": quadrant,

        "recommended_action": (
            recommended_action
        ),

        "sales_at_risk_value": round(
            sales_at_risk_value,
            2
        ),

        "capital_locked_value": round(
            capital_locked_value,
            2
        ),

        "value_at_stake": round(
            value_at_stake,
            2
        )
    }


# ============================================================
# SCORE ALL SKUS
# ============================================================

def score_all_skus(
    forecasts,
    inventory,
    sku_master
):
    """
    Score every SKU.

    forecasts:
        Dictionary:
        {
            sku_id: forecast_dataframe
        }

    inventory:
        inventory_snapshots.csv

    sku_master:
        sku_master.csv
    """

    # ========================================================
    # COPY DATA
    # ========================================================

    inventory = inventory.copy()

    sku_master = sku_master.copy()


    # ========================================================
    # CLEAN SKU IDS
    # ========================================================

    inventory["sku_id"] = (
        inventory["sku_id"]
        .astype(str)
    )

    sku_master["sku_id"] = (
        sku_master["sku_id"]
        .astype(str)
    )


    # ========================================================
    # LATEST INVENTORY
    # ========================================================

    inventory["date"] = pd.to_datetime(
        inventory["date"],
        errors="coerce"
    )


    latest_inventory = (

        inventory
        .sort_values("date")
        .groupby(
            "sku_id",
            as_index=False
        )
        .tail(1)
        .set_index("sku_id")
    )


    # ========================================================
    # MASTER INDEX
    # ========================================================

    master = (
        sku_master
        .drop_duplicates(
            "sku_id"
        )
        .set_index("sku_id")
    )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    total_forecast_skus = len(
        forecasts
    )

    skipped_inventory = 0
    skipped_master = 0
    skipped_forecast = 0

    rows = []


    # ========================================================
    # SCORE EACH SKU
    # ========================================================

    for sku_id, forecast_df in forecasts.items():

        sku_id = str(
            sku_id
        )


        # ----------------------------------------------------
        # Inventory missing
        # ----------------------------------------------------

        if sku_id not in latest_inventory.index:

            skipped_inventory += 1

            continue


        # ----------------------------------------------------
        # Master missing
        # ----------------------------------------------------

        if sku_id not in master.index:

            skipped_master += 1

            continue


        # ----------------------------------------------------
        # Forecast missing
        # ----------------------------------------------------

        if (
            forecast_df is None
            or forecast_df.empty
        ):

            skipped_forecast += 1

            continue


        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        result = score_sku(

            sku_id,

            forecast_df,

            latest_inventory.loc[
                sku_id
            ],

            master.loc[
                sku_id
            ]
        )


        if result is not None:

            rows.append(
                result
            )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if not rows:

        return pd.DataFrame()


    risk_table = pd.DataFrame(
        rows
    )


    # ========================================================
    # ADD SKU NAME + CATEGORY
    # ========================================================

    master_columns = [
        "sku_id"
    ]


    if "sku_name" in sku_master.columns:

        master_columns.append(
            "sku_name"
        )


    if "category" in sku_master.columns:

        master_columns.append(
            "category"
        )


    master_info = (
        sku_master[
            master_columns
        ]
        .drop_duplicates(
            "sku_id"
        )
    )


    risk_table = risk_table.merge(

        master_info,

        on="sku_id",

        how="left"
    )


    # ========================================================
    # SORT BY VALUE AT STAKE
    # ========================================================

    risk_table = (
        risk_table
        .sort_values(
            "value_at_stake",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "RISK SCORING DIAGNOSTICS"
    )

    print(
        "=" * 60
    )

    print(
        f"Forecast SKUs:       "
        f"{total_forecast_skus:,}"
    )

    print(
        f"SKUs scored:         "
        f"{len(risk_table):,}"
    )

    print(
        f"Skipped inventory:   "
        f"{skipped_inventory:,}"
    )

    print(
        f"Skipped master:      "
        f"{skipped_master:,}"
    )

    print(
        f"Skipped forecast:    "
        f"{skipped_forecast:,}"
    )


    # ========================================================
    # RISK RANGES
    # ========================================================

    print()

    print(
        "Stockout risk range:"
    )

    print(
        f"  "
        f"{risk_table['stockout_risk'].min():.3f}"
        f" → "
        f"{risk_table['stockout_risk'].max():.3f}"
    )


    print()

    print(
        "Overstock risk range:"
    )

    print(
        f"  "
        f"{risk_table['overstock_risk'].min():.3f}"
        f" → "
        f"{risk_table['overstock_risk'].max():.3f}"
    )


    # ========================================================
    # QUADRANT DISTRIBUTION
    # ========================================================

    print()

    print(
        "Quadrant distribution:"
    )

    print(
        risk_table[
            "quadrant"
        ].value_counts()
    )


    # ========================================================
    # TOTAL VALUE
    # ========================================================

    print()

    print(
        "Total value at stake: "
        f"£{risk_table['value_at_stake'].sum():,.2f}"
    )

    print(
        "=" * 60
    )

    print()


    return risk_table


# ============================================================
# MAIN
# ============================================================

def main():

    import joblib

    from forecast import forecast_forward


    # ========================================================
    # LOAD DATA
    # ========================================================

    weekly = pd.read_csv(
        "data/processed/weekly_features.csv",
        parse_dates=["date"]
    )


    inventory = pd.read_csv(
        "data/processed/inventory_snapshots.csv",
        parse_dates=["date"]
    )


    sku_master = pd.read_csv(
        "data/processed/sku_master.csv"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = joblib.load(
        "data/processed/model.pkl"
    )


    # ========================================================
    # FORECAST ALL SKUS
    # ========================================================

    print(
        "Generating forecasts..."
    )


    forecasts = {}


    sku_list = (
        weekly[
            "sku_id"
        ]
        .dropna()
        .unique()
    )


    for i, sku in enumerate(
        sku_list,
        start=1
    ):

        forecasts[
            str(sku)
        ] = forecast_forward(
            model,
            weekly,
            sku,
            horizon_weeks=FORWARD_WINDOW_WEEKS
        )


        # Progress every 500 SKUs
        if i % 500 == 0:

            print(
                f"Forecasted "
                f"{i:,} / "
                f"{len(sku_list):,} SKUs"
            )


    # ========================================================
    # SCORE
    # ========================================================

    risk_table = score_all_skus(

        forecasts,

        inventory,

        sku_master
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        "data/processed/risk_scores.csv"
    )


    risk_table.to_csv(
        output_path,
        index=False
    )


    print(
        f"Saved risk table to:"
        f" {output_path}"
    )


    print(
        f"Done. "
        f"{len(risk_table):,} SKUs scored."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()