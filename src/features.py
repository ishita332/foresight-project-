"""
Feature engineering for the demand forecast model.

All features are computed using only information available at or before
the row's own date (shift-based), to avoid leakage. Run after pipeline.py.
"""

import pandas as pd


LAG_WEEKS = [1, 2, 4, 8]
ROLL_WINDOWS = [7, 28]


def build_weekly_panel(sales_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales to weekly SKU-level demand (the forecast grain)."""
    df = sales_daily.copy()
    df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time
    weekly = (
        df.groupby(["sku_id", "week_start"])
        .agg(units_sold=("units_sold", "sum"),
             revenue=("revenue", "sum"),
             promo_flag=("promo_flag", "max"))
        .reset_index()
        .rename(columns={"week_start": "date"})
    )
    return weekly


def add_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features. No future data touches any row."""
    weekly = weekly.sort_values(["sku_id", "date"]).copy()
    g = weekly.groupby("sku_id")["units_sold"]

    for lag in LAG_WEEKS:
        weekly[f"lag_{lag}"] = g.shift(lag)

    for win in ROLL_WINDOWS:
        # shift(1) first so the current week's own value never leaks into its own rolling stat
        weekly[f"roll_mean_{win}"] = (
            g.shift(1).rolling(win, min_periods=max(2, win // 4)).mean().reset_index(level=0, drop=True)
        )
        weekly[f"roll_std_{win}"] = (
            g.shift(1).rolling(win, min_periods=max(2, win // 4)).std().reset_index(level=0, drop=True)
        )

    weekly["week_of_year"] = weekly["date"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["date"].dt.month
    weekly["promo_flag_prior_week"] = weekly.groupby("sku_id")["promo_flag"].shift(1).fillna(0)

    return weekly


def make_model_table(sales_daily_path="data/processed/sales_daily.csv",
                      out_path="data/processed/weekly_features.csv") -> pd.DataFrame:
    sales_daily = pd.read_csv(sales_daily_path, parse_dates=["date"])
    weekly = build_weekly_panel(sales_daily)
    weekly = add_features(weekly)
    weekly.to_csv(out_path, index=False)
    return weekly


if __name__ == "__main__":
    df = make_model_table()
    print(f"Wrote weekly feature table: {df.shape}")
    print(df.head())
