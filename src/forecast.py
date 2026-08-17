"""
Demand forecasting: seasonal-naive baseline, LightGBM model, and honest
rolling-origin backtesting. The non-negotiable rule: report WAPE for both
the baseline and the model on the SAME backtest folds.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28",
    "week_of_year", "month", "promo_flag_prior_week",
]
TARGET_COL = "units_sold"


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error. Robust to low-volume SKUs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error -- positive means the model over-forecasts."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def seasonal_naive_predict(weekly: pd.DataFrame, season_lag_weeks: int = 52) -> pd.Series:
    """
    Baseline: predict this week's demand = same SKU's demand N weeks ago.
    Falls back to lag_4 (last month) where a full season of history isn't
    available yet -- common for a ~2-year retail dataset.
    """
    g = weekly.groupby("sku_id")["units_sold"]
    seasonal = g.shift(season_lag_weeks)
    fallback = g.shift(4)
    return seasonal.fillna(fallback)


def rolling_origin_folds(dates: pd.Series, n_folds: int = 4, horizon_weeks: int = 8):
    """
    Yield (train_end_date, test_start_date, test_end_date) tuples that
    walk forward through time -- never a random split for time series.
    """
    unique_dates = sorted(dates.unique())
    fold_step = max(len(unique_dates) // (n_folds + 1), horizon_weeks)
    folds = []
    for i in range(1, n_folds + 1):
        cut = min(i * fold_step, len(unique_dates) - horizon_weeks - 1)
        if cut <= 0:
            continue
        train_end = unique_dates[cut]
        test_start = unique_dates[cut + 1] if cut + 1 < len(unique_dates) else train_end
        test_end_idx = min(cut + horizon_weeks, len(unique_dates) - 1)
        test_end = unique_dates[test_end_idx]
        folds.append((train_end, test_start, test_end))
    return folds


def backtest(weekly: pd.DataFrame, n_folds: int = 4, horizon_weeks: int = 8) -> pd.DataFrame:
    """
    Rolling-origin backtest comparing the LightGBM model against the
    seasonal-naive baseline. Returns a per-fold results table.
    """
    weekly = weekly.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    weekly["naive_pred"] = seasonal_naive_predict(weekly)
    weekly = weekly.dropna(subset=["naive_pred"])

    folds = rolling_origin_folds(weekly["date"], n_folds=n_folds, horizon_weeks=horizon_weeks)
    results = []

    for fold_i, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train = weekly[weekly["date"] <= train_end]
        test = weekly[(weekly["date"] > train_end) & (weekly["date"] <= test_end)]
        if train.empty or test.empty:
            continue

        model = LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
        )
        model.fit(train[FEATURE_COLS], train[TARGET_COL])
        model_pred = model.predict(test[FEATURE_COLS]).clip(min=0)

        results.append({
            "fold": fold_i,
            "train_end": train_end, "test_start": test_start, "test_end": test_end,
            "n_test_rows": len(test),
            "wape_baseline": wape(test[TARGET_COL], test["naive_pred"]),
            "wape_model": wape(test[TARGET_COL], model_pred),
            "bias_baseline": bias(test[TARGET_COL], test["naive_pred"]),
            "bias_model": bias(test[TARGET_COL], model_pred),
        })

    return pd.DataFrame(results)


def train_final_model(weekly: pd.DataFrame) -> LGBMRegressor:
    """Train on all available history for use in production forecasting."""
    train = weekly.dropna(subset=FEATURE_COLS + [TARGET_COL])
    model = LGBMRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
    model.fit(train[FEATURE_COLS], train[TARGET_COL])
    return model


def forecast_forward(model, weekly: pd.DataFrame, sku_id: str, horizon_weeks: int = 8) -> pd.DataFrame:
    """
    Recursively forecast a single SKU forward `horizon_weeks`, updating
    lag/rolling features with each new prediction (proper multi-step forecast).
    """
    hist = weekly[weekly["sku_id"] == sku_id].sort_values("date").copy()
    if hist.empty:
        return pd.DataFrame()

    last_date = hist["date"].max()
    series = hist.set_index("date")["units_sold"].copy()
    preds = []

    for step in range(1, horizon_weeks + 1):
        next_date = last_date + pd.Timedelta(weeks=step)
        row = {
            "lag_1": series.iloc[-1] if len(series) >= 1 else np.nan,
            "lag_2": series.iloc[-2] if len(series) >= 2 else np.nan,
            "lag_4": series.iloc[-4] if len(series) >= 4 else np.nan,
            "lag_8": series.iloc[-8] if len(series) >= 8 else np.nan,
            "roll_mean_7": series.tail(7).mean(),
            "roll_mean_28": series.tail(28).mean(),
            "roll_std_7": series.tail(7).std(),
            "roll_std_28": series.tail(28).std(),
            "week_of_year": next_date.isocalendar()[1],
            "month": next_date.month,
            "promo_flag_prior_week": 0,
        }
        X = pd.DataFrame([row])[FEATURE_COLS]
        pred = max(float(model.predict(X)[0]), 0)
        preds.append({"sku_id": sku_id, "date": next_date, "forecast": pred})
        series.loc[next_date] = pred  # feed forward for next step's lags

    return pd.DataFrame(preds)


if __name__ == "__main__":
    weekly = pd.read_csv("data/processed/weekly_features.csv", parse_dates=["date"])
    bt = backtest(weekly)
    print(bt)
    print("\nMean WAPE -- baseline:", bt["wape_baseline"].mean(), "| model:", bt["wape_model"].mean())
    bt.to_csv("reports/backtest_results.csv", index=False)
