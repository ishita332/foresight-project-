# Project FORESIGHT — Demand & Inventory Intelligence

Zidio Development internship engagement, client: NorthBay Living.

**Live dashboard:** https://hg2j6qitd9e6a28fphxpmz.streamlit.app/
**Live scoring API:** https://foresight-project.onrender.com (try `/health` or `/score/{sku_id}`)

## Problem

NorthBay Living plans inventory on gut feel and spreadsheets, causing lost
sales from stockouts and locked-up cash from overstock. FORESIGHT turns
their sales history into a weekly SKU-level demand forecast and a
stockout/overstock early-warning system the ops team can act on.

## Data

Source data (Kaggle, real transactional + inventory data):
- `sku_master` — sku_id, sku_name, category, subcategory, unit_price, cost_price, brand
- `sales_transactions` — date, receipt_id, store_id, sku_id, customer_id, quantity, unit_price, total_value, channel, discount_pct, promo_id
- `inventory_snapshot` — store_id, sku_id, stock_on_hand, reorder_point, safety_stock, last_restock_date
- `promotions` — promo events used to build the calendar table

The pipeline aggregates across stores to SKU-day level, cleans missing/invalid
rows, and derives the four-table schema (`sales_daily`, `sku_master`,
`calendar`, `inventory_snapshots`) used by the rest of the project. Two
fields aren't present in the source data and are documented assumptions:
`lead_time_days` (assumed constant, 14 days) and `on_order_units` (assumed 0,
not tracked). See `reports/data_quality.md` for the full list, auto-generated
by the pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the four raw files (any of `.csv`/`.xlsx`/`.xls`) in `data/raw/`:
```
data/raw/sales_transactions.csv (or .xlsx)
data/raw/sku_master.csv
data/raw/inventory_snapshot.csv
data/raw/promotions.csv
```

## Run — end to end

```bash
python src/pipeline.py
python src/features.py
python src/train_and_score.py
```

This produces everything under `data/processed/`: cleaned tables, weekly
features, the trained model, forward forecasts, and risk scores.

## Run — dashboard

```bash
streamlit run app/dashboard.py
```

## Run — scoring service

```bash
uvicorn service.main:app --reload
```
Then: `GET http://127.0.0.1:8000/score/{sku_id}` or `POST /score/batch`
with `{"sku_ids": ["...", "..."]}`.

## Backtest result

Rolling-origin backtest (`src/forecast.py::backtest`), 4 folds, 8-week
horizon, comparing the LightGBM model against a seasonal-naive baseline
on the same held-out folds:

| Metric | Baseline (seasonal-naive) | Model (LightGBM) |
|---|---|---|
| Mean WAPE | 0.612 | **0.448** |

The model beats the baseline by roughly 27% relative WAPE reduction,
consistently across all 4 folds. Full per-fold results:
`reports/backtest_results.csv`.

## Result summary

- **4,481 SKUs** scored
- **364** flagged for reorder (stockout risk)
- **3,075** flagged for markdown/clear (overstock risk)
- **1,042** healthy, no action needed
- **~₹1.32 billion** total value at stake across all flagged SKUs

## Key assumptions

- Sales aggregated across all stores/channels to SKU-day level (no
  per-store forecast in scope).
- `promo_flag` derived from `promo_id` presence (or `discount_pct` > 0 if
  `promo_id` absent).
- `inventory_snapshot` is a single current snapshot (not a daily time
  series); aggregated by summing `stock_on_hand`/`reorder_point` across
  stores per SKU.
- `lead_time_days` not present in source data; assumed constant at 14
  days for all SKUs. `on_order_units` assumed 0 (not tracked).
- SKUs with fewer than 30 days of sales history excluded from modelling.

## Repository structure

```
foresight/
  data/raw/              # place the downloaded raw files here (gitignored)
  data/processed/         # pipeline outputs (large files gitignored)
  src/
    pipeline.py           # ingest + clean + build 4-table schema
    features.py           # weekly aggregation + lag/rolling features
    forecast.py            # baseline, model, rolling-origin backtest, WAPE
    risk.py                # stockout/overstock scoring
    train_and_score.py     # orchestrates model training + full scoring run
  app/dashboard.py         # Streamlit planning dashboard (live link above)
  service/main.py          # FastAPI scoring service
  reports/                  # data-quality report, backtest results, EDA memo, exec readout
```

## Limitations

- Inventory snapshot is a single point-in-time record, not a daily time
  series — daily stock movement between snapshots is not observed.
- Lead time is assumed constant across all SKUs (real per-SKU lead times
  weren't in the source data).
- Category taxonomy comes directly from the source `sku_master` file.
