"""
Stockout / overstock risk scoring.

Combines the forward forecast with the latest inventory position to score
every SKU into a quadrant (Reorder now / Markdown-clear / Watch-volatile /
Healthy) with a recommended action and the estimated currency value at stake.
Transparent, rule-based -- not a black box, per the brief's requirement.
"""

import pandas as pd

STOCKOUT_THRESHOLD = 0.5   # projected stock below this fraction of forecast lead-time demand -> at risk
OVERSTOCK_THRESHOLD = 2.5  # on-hand more than this multiple of forward-window demand -> overstocked
FORWARD_WINDOW_WEEKS = 8


def score_sku(sku_id: str, forecast_df: pd.DataFrame, latest_inventory: pd.Series,
              sku_master_row: pd.Series) -> dict:
    """Score a single SKU's stockout/overstock risk given its forecast and inventory."""
    lead_time_weeks = max(latest_inventory["lead_time_days"] / 7, 1)
    on_hand = latest_inventory["on_hand_units"]
    on_order = latest_inventory["on_order_units"]

    # Demand expected over the replenishment lead time
    lead_time_demand = forecast_df["forecast"].iloc[:int(round(lead_time_weeks))].sum()
    projected_stock_at_lead_time = on_hand + on_order - lead_time_demand
    stockout_ratio = 1 - (projected_stock_at_lead_time / max(lead_time_demand, 1))
    stockout_risk = min(max(stockout_ratio, 0), 1)

    # Demand expected over the forward planning window
    forward_demand = forecast_df["forecast"].iloc[:FORWARD_WINDOW_WEEKS].sum()
    overstock_ratio = on_hand / max(forward_demand, 1)
    overstock_risk = min(max((overstock_ratio - 1) / (OVERSTOCK_THRESHOLD - 1), 0), 1)

    is_stockout_risk = stockout_risk > STOCKOUT_THRESHOLD
    is_overstock_risk = overstock_risk > (1 / OVERSTOCK_THRESHOLD)

    if is_stockout_risk and not is_overstock_risk:
        quadrant, action = "Reorder now", "Raise a replenishment order before stock runs out."
    elif is_overstock_risk and not is_stockout_risk:
        quadrant, action = "Markdown / clear", "Promote or discount to free up capital."
    elif is_stockout_risk and is_overstock_risk:
        quadrant, action = "Watch / volatile", "Investigate -- demand is erratic; review manually."
    else:
        quadrant, action = "Healthy", "No action needed; leave as is."

    list_price = sku_master_row.get("list_price", 0)
    unit_cost = sku_master_row.get("unit_cost", 0)
    sales_at_risk_value = max(lead_time_demand - (on_hand + on_order), 0) * list_price
    capital_locked_value = max(on_hand - forward_demand, 0) * unit_cost

    return {
        "sku_id": sku_id,
        "on_hand_units": on_hand,
        "on_order_units": on_order,
        "lead_time_days": latest_inventory["lead_time_days"],
        "stockout_risk": round(stockout_risk, 3),
        "overstock_risk": round(overstock_risk, 3),
        "quadrant": quadrant,
        "recommended_action": action,
        "sales_at_risk_value": round(sales_at_risk_value, 2),
        "capital_locked_value": round(capital_locked_value, 2),
        "value_at_stake": round(max(sales_at_risk_value, capital_locked_value), 2),
    }


def score_all_skus(forecasts: dict, inventory: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """
    forecasts: {sku_id: forecast_df} as produced by forecast.forecast_forward()
    inventory: the inventory_snapshots table (uses the most recent date per SKU)
    sku_master: for list_price / unit_cost lookups
    """
    latest_inv = inventory.sort_values("date").groupby("sku_id").tail(1).set_index("sku_id")
    master = sku_master.set_index("sku_id")

    rows = []
    for sku_id, fdf in forecasts.items():
        if sku_id not in latest_inv.index or sku_id not in master.index or fdf.empty:
            continue
        rows.append(score_sku(sku_id, fdf, latest_inv.loc[sku_id], master.loc[sku_id]))

    return pd.DataFrame(rows).sort_values("value_at_stake", ascending=False)


if __name__ == "__main__":
    import joblib
    from forecast import forecast_forward

    weekly = pd.read_csv("data/processed/weekly_features.csv", parse_dates=["date"])
    inventory = pd.read_csv("data/processed/inventory_snapshots.csv", parse_dates=["date"])
    sku_master = pd.read_csv("data/processed/sku_master.csv")
    model = joblib.load("data/processed/model.pkl")

    forecasts = {sku: forecast_forward(model, weekly, sku) for sku in weekly["sku_id"].unique()}
    risk_table = score_all_skus(forecasts, inventory, sku_master)
    risk_table.to_csv("data/processed/risk_scores.csv", index=False)
    print(risk_table.head(20))
