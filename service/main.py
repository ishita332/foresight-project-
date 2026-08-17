"""
FORESIGHT scoring service.

Run with:
    uvicorn service.main:app --reload

Endpoints:
    GET  /health
    GET  /score/{sku_id}          -> forecast + risk for one SKU
    POST /score/batch             -> forecast + risk for a list of SKUs
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="FORESIGHT Scoring Service", version="1.0")

DATA_DIR = Path("data/processed")
_cache = {}


def _load():
    """Lazy-load processed tables once, cache in memory."""
    if not _cache:
        try:
            _cache["forecasts"] = pd.read_csv(DATA_DIR / "forecasts.csv", parse_dates=["date"])
            _cache["risk"] = pd.read_csv(DATA_DIR / "risk_scores.csv")
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Processed data not found ({e.filename}). Run the pipeline and "
                       "train_and_score.py before starting the service.",
            )
    return _cache["forecasts"], _cache["risk"]


class BatchRequest(BaseModel):
    sku_ids: list[str]


def _score_one(sku_id: str) -> dict:
    forecasts, risk = _load()
    sku_forecast = forecasts[forecasts["sku_id"] == sku_id].sort_values("date")
    sku_risk = risk[risk["sku_id"] == sku_id]

    if sku_forecast.empty or sku_risk.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found or not scored.")

    risk_row = sku_risk.iloc[0].to_dict()
    return {
        "sku_id": sku_id,
        "description": risk_row.get("description"),
        "forecast": [
            {"date": row["date"].strftime("%Y-%m-%d"), "units": round(row["forecast"], 1)}
            for _, row in sku_forecast.iterrows()
        ],
        "risk": {
            "stockout_risk": risk_row["stockout_risk"],
            "overstock_risk": risk_row["overstock_risk"],
            "quadrant": risk_row["quadrant"],
            "recommended_action": risk_row["recommended_action"],
            "value_at_stake": risk_row["value_at_stake"],
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/score/{sku_id}")
def score(sku_id: str):
    return _score_one(sku_id)


@app.post("/score/batch")
def score_batch(req: BatchRequest):
    results = []
    for sku_id in req.sku_ids:
        try:
            results.append(_score_one(sku_id))
        except HTTPException as e:
            results.append({"sku_id": sku_id, "error": e.detail})
    return {"results": results}
