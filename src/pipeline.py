"""
FORESIGHT data pipeline -- built for the real dataset:
    sku_master.xlsx           (sku_id, sku_name, category, subcategory, unit_price, cost_price, brand)
    sales_transactions.csv     (date, receipt_id, store_id, sku_id, customer_id, quantity,
                                 unit_price, total_value, channel, discount_pct, promo_id)
    inventory_snapshot.xlsx    (store_id, sku_id, stock_on_hand, reorder_point, safety_stock, last_restock_date)
    promotions.xlsx            (optional -- used for calendar promo events if present)

Cleans everything, aggregates across stores to SKU level, and produces
the four-table analysis-ready schema the rest of the project expects:
    sales_daily, sku_master, calendar, inventory_snapshots

Run end-to-end with:
    python src/pipeline.py

IMPORTANT: sales_transactions is a very large file. Re-save it as CSV
from Excel first (File > Save As > CSV) -- reading a 700MB+ .xlsx file
directly is extremely slow / may crash.

Expected input files, all inside data/raw/:
    sales_transactions.csv
    sku_master.xlsx
    inventory_snapshot.xlsx
    promotions.xlsx   (optional)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pipeline")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# No lead-time field exists in inventory_snapshot.xlsx -- assumed constant
# since real lead times weren't provided. Change this if your mentor gives
# you a real number, and update the README/data-quality note to match.
ASSUMED_LEAD_TIME_DAYS = 14


# --------------------------------------------------------------------------
# 1. INGEST
# --------------------------------------------------------------------------
def _find_file(raw_dir: Path, base_name: str) -> Path:
    """Find a file by base name, trying .csv, .xlsx, then .xls (case-insensitive)."""
    for ext in (".csv", ".xlsx", ".xls"):
        candidate = raw_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    # case-insensitive fallback: scan directory
    for f in raw_dir.iterdir():
        if f.stem.lower() == base_name.lower() and f.suffix.lower() in (".csv", ".xlsx", ".xls"):
            return f
    raise FileNotFoundError(f"Could not find {base_name}.csv, .xlsx or .xls in {raw_dir}")


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    elif path.suffix.lower() == ".xls":
        return pd.read_excel(path, engine="xlrd")
    raise ValueError(f"Unsupported file type: {path}")


def load_raw(raw_dir: Path):
    log.info(f"Loading raw files from {raw_dir}")

    sales_path = _find_file(raw_dir, "sales_transactions")
    if sales_path.suffix.lower() == ".csv":
        sales = pd.read_csv(sales_path)
    elif sales_path.suffix.lower() == ".xlsx":
        log.info("Reading sales_transactions.xlsx with the fast 'calamine' engine "
                  "(this is a large file -- this step can take a few minutes)...")
        sales = pd.read_excel(sales_path, engine="calamine")
    else:
        log.info("Reading sales_transactions.xls -- this is a large file, "
                  "this step can take a few minutes...")
        sales = pd.read_excel(sales_path, engine="xlrd")

    sku_master = _read_any(_find_file(raw_dir, "sku_master"))
    inventory = _read_any(_find_file(raw_dir, "inventory_snapshot"))

    try:
        promo_path = _find_file(raw_dir, "promotions")
        promotions = _read_any(promo_path)
    except FileNotFoundError:
        promotions = None

    log.info(f"sales_transactions: {sales.shape}, sku_master: {sku_master.shape}, "
             f"inventory_snapshot: {inventory.shape}")
    return sales, sku_master, inventory, promotions


# --------------------------------------------------------------------------
# 2. CLEAN
# --------------------------------------------------------------------------
def clean_sales(sales: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    issues = {}
    n0 = len(sales)

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce", dayfirst=True)
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce")
    sales["unit_price"] = pd.to_numeric(sales["unit_price"], errors="coerce")
    sales["total_value"] = pd.to_numeric(sales["total_value"], errors="coerce")

    before = len(sales)
    sales = sales.dropna(subset=["date", "sku_id", "quantity", "unit_price"])
    issues["dropped_missing_key_fields"] = before - len(sales)

    before = len(sales)
    sales = sales[sales["quantity"] > 0]
    issues["non_positive_quantity_removed"] = before - len(sales)

    before = len(sales)
    sales = sales[sales["unit_price"] > 0]
    issues["non_positive_price_removed"] = before - len(sales)

    before = len(sales)
    sales = sales.drop_duplicates(subset=["receipt_id", "sku_id"] if "receipt_id" in sales.columns else None)
    issues["duplicate_rows_removed"] = before - len(sales)

    # total_value should already be quantity * unit_price (net of discount);
    # trust it as revenue if present, else recompute
    if "total_value" in sales.columns:
        sales["revenue"] = sales["total_value"].fillna(sales["quantity"] * sales["unit_price"])
    else:
        sales["revenue"] = sales["quantity"] * sales["unit_price"]

    sales["promo_flag"] = 0
    if "promo_id" in sales.columns:
        sales["promo_flag"] = sales["promo_id"].notna().astype(int)
    elif "discount_pct" in sales.columns:
        sales["promo_flag"] = (pd.to_numeric(sales["discount_pct"], errors="coerce").fillna(0) > 0).astype(int)

    issues["rows_in"] = n0
    issues["rows_out"] = len(sales)
    log.info(f"Cleaned sales: {n0:,} -> {len(sales):,} rows. Issues: {issues}")
    return sales, issues


def clean_sku_master(sku_master: pd.DataFrame) -> pd.DataFrame:
    df = sku_master.copy()
    df = df.dropna(subset=["sku_id"]).drop_duplicates(subset=["sku_id"])
    rename_map = {"cost_price": "unit_cost", "unit_price": "list_price"}
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    for col in ("unit_cost", "list_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "subcategory" not in df.columns:
        df["subcategory"] = df.get("category", "OTHER")
    return df


def clean_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """
    inventory_snapshot.xlsx is a single current snapshot per store+SKU
    (not a daily time series). Aggregate to one row per SKU (summed
    across stores) representing the company's total current position.
    """
    df = inventory.copy()
    df = df.dropna(subset=["sku_id"])
    for col in ("stock_on_hand", "reorder_point", "safety_stock"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    agg = df.groupby("sku_id").agg(
        on_hand_units=("stock_on_hand", "sum"),
        reorder_point=("reorder_point", "sum"),
        safety_stock=("safety_stock", "sum") if "safety_stock" in df.columns else ("stock_on_hand", "sum"),
    ).reset_index()

    agg["on_order_units"] = 0  # not tracked in source data
    agg["lead_time_days"] = ASSUMED_LEAD_TIME_DAYS  # not present in source data -- documented assumption
    agg["date"] = pd.Timestamp.today().normalize()  # treated as a single "current" snapshot
    return agg


# --------------------------------------------------------------------------
# 3. BUILD SCHEMA TABLES
# --------------------------------------------------------------------------
def build_sales_daily(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across stores/customers to one row per SKU per day."""
    daily = (
        sales.groupby(["sku_id", "date"])
        .agg(units_sold=("quantity", "sum"),
             revenue=("revenue", "sum"),
             unit_price=("unit_price", "mean"),
             promo_flag=("promo_flag", "max"))
        .reset_index()
    )
    return daily


def build_calendar(min_date, max_date, promotions: pd.DataFrame | None) -> pd.DataFrame:
    dates = pd.date_range(min_date, max_date, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["week"] = cal["date"].dt.isocalendar().week
    cal["month"] = cal["date"].dt.month
    cal["season"] = cal["month"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                       3: "Spring", 4: "Spring", 5: "Spring",
                                       6: "Summer", 7: "Summer", 8: "Summer",
                                       9: "Autumn", 10: "Autumn", 11: "Autumn"})
    holiday_md = {(1, 1), (8, 15), (10, 2), (12, 25)}  # India-ish proxy; adjust if you know real ones
    cal["is_holiday"] = cal["date"].apply(lambda d: (d.month, d.day) in holiday_md).astype(int)
    cal["promo_event"] = None

    if promotions is not None and "start_date" in promotions.columns:
        promotions = promotions.copy()
        promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce")
        end_col = "end_date" if "end_date" in promotions.columns else "start_date"
        promotions[end_col] = pd.to_datetime(promotions[end_col], errors="coerce")
        name_col = "promo_name" if "promo_name" in promotions.columns else promotions.columns[0]
        for _, row in promotions.dropna(subset=["start_date"]).iterrows():
            mask = (cal["date"] >= row["start_date"]) & (cal["date"] <= row.get(end_col, row["start_date"]))
            cal.loc[mask, "promo_event"] = row.get(name_col, "Promo")

    return cal


# --------------------------------------------------------------------------
# 4. MERGE
# --------------------------------------------------------------------------
def merge_all(sales_daily, sku_master, calendar, inventory) -> pd.DataFrame:
    df = sales_daily.merge(sku_master, on="sku_id", how="left")
    df = df.merge(calendar, on="date", how="left")
    # inventory is a single current snapshot per SKU -- broadcast onto every date row
    inv_static = inventory.drop(columns=["date"])
    df = df.merge(inv_static, on="sku_id", how="left")
    return df


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main(raw_dir: str):
    raw_dir = Path(raw_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    sales, sku_master_raw, inventory_raw, promotions = load_raw(raw_dir)
    sales, issues = clean_sales(sales)
    sku_master = clean_sku_master(sku_master_raw)
    inventory = clean_inventory(inventory_raw)

    sales_daily = build_sales_daily(sales)
    calendar = build_calendar(sales_daily["date"].min(), sales_daily["date"].max(), promotions)

    # Keep only SKUs with a reasonable amount of history
    sku_counts = sales_daily.groupby("sku_id")["date"].nunique()
    keep_skus = sku_counts[sku_counts >= 30].index
    sales_daily = sales_daily[sales_daily["sku_id"].isin(keep_skus)]
    sku_master = sku_master[sku_master["sku_id"].isin(keep_skus)]
    inventory = inventory[inventory["sku_id"].isin(keep_skus)]
    issues["skus_dropped_insufficient_history"] = int(len(sku_counts) - len(keep_skus))
    issues["skus_kept"] = int(len(keep_skus))

    processed = merge_all(sales_daily, sku_master, calendar, inventory)

    sales_daily.to_csv(PROCESSED_DIR / "sales_daily.csv", index=False)
    sku_master.to_csv(PROCESSED_DIR / "sku_master.csv", index=False)
    calendar.to_csv(PROCESSED_DIR / "calendar.csv", index=False)
    inventory.to_csv(PROCESSED_DIR / "inventory_snapshots.csv", index=False)
    processed.to_csv(PROCESSED_DIR / "analysis_ready.csv", index=False)

    log.info(f"Wrote processed tables to {PROCESSED_DIR}/")
    log.info(f"analysis_ready.csv: {processed.shape}")

    with open("reports/data_quality.md", "w") as f:
        f.write("# Data Quality Report\n\n")
        f.write("Auto-generated by `src/pipeline.py`. Add your own commentary before submitting.\n\n")
        f.write("## Cleaning issues found and handled\n\n")
        for k, v in issues.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Key assumptions\n\n")
        f.write("- Sales aggregated across all stores/channels to SKU-day level (no per-store forecast in scope).\n")
        f.write("- `promo_flag` derived from `promo_id` presence (or `discount_pct` > 0 if `promo_id` absent).\n")
        f.write("- `inventory_snapshot` is a single current snapshot (not a daily time series); "
                 "aggregated by summing `stock_on_hand`/`reorder_point` across stores per SKU.\n")
        f.write(f"- `lead_time_days` is not present in the source data; assumed constant at "
                 f"{ASSUMED_LEAD_TIME_DAYS} days for all SKUs. `on_order_units` assumed 0 (not tracked).\n")
        f.write("- SKUs with fewer than 30 days of sales history excluded from modelling.\n")

    log.info("Wrote reports/data_quality.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw",
                         help="Folder containing sales_transactions.csv, sku_master.xlsx, "
                              "inventory_snapshot.xlsx, promotions.xlsx")
    args = parser.parse_args()
    main(args.input)