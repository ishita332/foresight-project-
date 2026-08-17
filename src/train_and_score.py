"""
Trains the final forecasting model on all available history, generates a
forward forecast for every SKU, and produces the risk-scoring table.

Run after pipeline.py and features.py:
    python src/train_and_score.py
"""

import joblib
import pandas as pd

from forecast import train_final_model, forecast_forward, backtest
from risk import score_all_skus

HORIZON_WEEKS = 8


def main():
    weekly = pd.read_csv("data/processed/weekly_features.csv", parse_dates=["date"])
    inventory = pd.read_csv("data/processed/inventory_snapshots.csv", parse_dates=["date"])
    sku_master = pd.read_csv("data/processed/sku_master.csv")

    print("Running backtest (baseline vs model)...")
    bt = backtest(weekly)
    bt.to_csv("reports/backtest_results.csv", index=False)
    print(bt)
    print(f"\nMean WAPE -- baseline: {bt['wape_baseline'].mean():.3f} | model: {bt['wape_model'].mean():.3f}")

    print("\nTraining final model on full history...")
    model = train_final_model(weekly)
    joblib.dump(model, "data/processed/model.pkl")

    print("Forecasting forward for every SKU...")
    forecasts = {}
    all_forecast_rows = []
    for sku_id in weekly["sku_id"].unique():
        fdf = forecast_forward(model, weekly, sku_id, horizon_weeks=HORIZON_WEEKS)
        forecasts[sku_id] = fdf
        all_forecast_rows.append(fdf)
    forecast_table = pd.concat(all_forecast_rows, ignore_index=True)
    forecast_table.to_csv("data/processed/forecasts.csv", index=False)

    print("Scoring stockout / overstock risk...")
    risk_table = score_all_skus(forecasts, inventory, sku_master)
    risk_table = risk_table.merge(sku_master[["sku_id", "sku_name", "category"]], on="sku_id", how="left")
    risk_table.to_csv("data/processed/risk_scores.csv", index=False)

    print(f"\nDone. {len(risk_table)} SKUs scored.")
    print(risk_table["quadrant"].value_counts())
    print(f"\nTotal value at stake: {risk_table['value_at_stake'].sum():,.2f}")


if __name__ == "__main__":
    main()
