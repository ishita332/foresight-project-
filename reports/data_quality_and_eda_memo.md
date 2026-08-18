# Data Quality & EDA Insight Memo

**Project:** FORESIGHT — Demand & Inventory Intelligence
**Client:** NorthBay Living
**Prepared by:** Data Scientist, Zidio Development internship engagement

---

## 1. Data quality — issues found and how they were handled

The raw `sales_transactions` file contained **9,972,038** line items. After
cleaning, **3,915,719** rows remained as the usable, analysis-ready demand
signal. The gap is explained by:

| Issue | Rows affected | How it was handled |
|---|---|---|
| Missing date, SKU, quantity, or price | 0 | Would be dropped (none found in this extract) |
| Non-positive quantity (returns/cancellations) | 0 | Would be excluded from demand (none found — see note below) |
| Non-positive price | 0 | Would be excluded (none found) |
| Duplicate transaction rows | 14,935 | Removed via `drop_duplicates` on receipt_id + sku_id |
| Rows aggregated away by daily/SKU rollup | ~6,041,384 | Multiple line items per SKU per day were summed into one `sales_daily` row (this is expected — many customers can buy the same SKU on the same day across different receipts/stores) |
| SKUs with <30 days of sales history | 519 of 5,000 SKUs (10.4%) | Excluded from modelling — insufficient history to learn a reliable weekly pattern (likely newly launched or discontinued products) |

**Note:** unlike a typical retail extract, this dataset had no negative
quantities or non-positive prices, and no missing key fields — the raw
export was already fairly clean at the row level. The main cleaning work
was deduplication and aggregation, not error-correction.

### Key assumptions made (documented, not hidden)

- Sales aggregated across all stores/channels to SKU-day level (no
  per-store forecast in current scope).
- `promo_flag` derived from `promo_id` presence (or `discount_pct` > 0
  where `promo_id` is absent).
- `inventory_snapshot` is a single current snapshot, not a daily time
  series — aggregated by summing `stock_on_hand`/`reorder_point` across
  stores per SKU.
- `lead_time_days` is not present in the source data; assumed constant at
  14 days for all SKUs. `on_order_units` assumed 0 (not tracked in source).

---

## 2. EDA insights

Computed directly from `sales_daily.csv` via `src/eda_summary.py`.

### Insight 1 — Revenue is heavily concentrated in a small number of SKUs

The top seller, **EliteHome Hair Care Pack of 6 (SKU04321)**, generated
**₹40.3 crore** in revenue over the period — nearly **4x** the next-best
SKU (QuickBite Mobile Accessories 2kg, ₹10.3 crore). The top 10 SKUs by
revenue span five different categories (Personal Care, Electronics &
Accessories, Home & Kitchen, Dairy & Bakery, Apparel & Footwear), so this
isn't a single-category effect.

**Business implication:** stockout risk on SKU04321 specifically carries
outsized revenue exposure. This SKU should get manual monitoring in
addition to the automated risk flag, and any reorder delay on it deserves
priority escalation.

### Insight 2 — No dead stock in the active catalog, but ~10% of SKUs lack enough history to forecast

Every one of the 5,000 SKUs sold at least one unit in the last 60 days of
the data — **0% dead stock** by that definition. This is a healthy sign
for the existing catalog. However, 519 SKUs (10.4%) had fewer than 30
days of sales history and were excluded from the forecasting model. These
are most likely recently launched products; they still carry stockout/
overstock risk but need a different approach (e.g. category-level
proxies) rather than a SKU-specific forecast, since there isn't enough
history yet to learn their individual demand pattern.

**Business implication:** the "no dead stock" finding is good news, but
it also means slow movers may simply not exist yet in this snapshot, or
turnover is fast enough that dead stock gets cleared before it's
detectable at a 60-day window — worth re-checking this metric on a longer
window (90–120 days) in the next refresh.

### Insight 3 — Demand is remarkably flat across the year — no strong seasonality detected

Average monthly demand ranges only from about 611,594 to 617,478 units
month to month — a spread of under 1%. There's no clear festive-season
spike, no summer/winter swing, and no single month standing out. This is
unusual for a home & lifestyle retailer, where seasonal categories
(décor, small appliances) typically show holiday-period lift.

**Business implication:** the forecasting model gains little from
calendar/seasonal features in this dataset as it stands — safety stock
levels likely don't need heavy seasonal buffering. It's worth confirming
with the merchandising team whether this flatness reflects real demand or
a data-collection artifact (e.g. store operating hours or promotional
calendar not fully captured in the extract), since it's an unusual
pattern for this category of retailer.

### Insight 4 — Category revenue is broadly spread, not dominated by one category

Revenue is distributed across Home & Kitchen, Personal Care, Electronics
& Accessories, Dairy & Bakery, and Apparel & Footwear without one category
dominating the top 10 sellers list. This suggests inventory risk (and the
₹1.32B value-at-stake figure from the risk scoring) is a portfolio-wide
issue, not concentrated in a single product line — the reorder/markdown
recommendations should be reviewed by multiple category owners, not just
one merchandiser.

---

## 3. Charts

See the live dashboard's **Decisioning view** for the labelled,
interactive version of the stockout-vs-overstock grid (Figure 6 in the
brief). Static category/seasonality charts can be regenerated any time
with `python src/eda_summary.py`.

---

*This memo satisfies deliverable D2 (data-quality & EDA insight memo) per
the FORESIGHT engagement brief, Section 09.*