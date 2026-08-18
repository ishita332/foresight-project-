"""
Computes the EDA facts needed for the data-quality & insight memo:
top sellers, dead stock, seasonality, category performance.

Run with:
    python src/eda_summary.py
"""

import pandas as pd

sales = pd.read_csv("data/processed/sales_daily.csv", parse_dates=["date"])
sku_master = pd.read_csv("data/processed/sku_master.csv")

print("=" * 60)
print("TOP 10 SELLERS (by total revenue)")
print("=" * 60)
top = (sales.groupby("sku_id")["revenue"].sum()
       .sort_values(ascending=False).head(10)
       .reset_index().merge(sku_master[["sku_id", "sku_name", "category"]], on="sku_id", how="left"))
print(top.to_string(index=False))

print("\n" + "=" * 60)
print("DEAD STOCK (no sales in the last 60 days of data)")
print("=" * 60)
last_date = sales["date"].max()
cutoff = last_date - pd.Timedelta(days=60)
recent_sales = sales[sales["date"] >= cutoff].groupby("sku_id")["units_sold"].sum()
all_skus = set(sku_master["sku_id"])
sold_recently = set(recent_sales[recent_sales > 0].index)
dead = all_skus - sold_recently
print(f"Total SKUs: {len(all_skus)}")
print(f"Dead stock (0 units sold in last 60 days): {len(dead)} SKUs "
      f"({len(dead)/len(all_skus)*100:.1f}% of catalog)")

print("\n" + "=" * 60)
print("SEASONALITY - avg weekly demand by month")
print("=" * 60)
sales["month"] = sales["date"].dt.month_name()
monthly = sales.groupby("month")["units_sold"].sum().sort_values(ascending=False)
print(monthly.head(12).to_string())

print("\n" + "=" * 60)
print("CATEGORY PERFORMANCE (total revenue)")
print("=" * 60)
cat_rev = sales.merge(sku_master[["sku_id", "category"]], on="sku_id", how="left")
cat_summary = cat_rev.groupby("category")["revenue"].sum().sort_values(ascending=False)
print(cat_summary.to_string())

print("\n" + "=" * 60)
print(f"Date range: {sales['date'].min().date()} to {sales['date'].max().date()}")
print(f"Total SKUs with sales: {sales['sku_id'].nunique()}")
print(f"Total revenue: Rs {sales['revenue'].sum():,.0f}")
print("=" * 60)