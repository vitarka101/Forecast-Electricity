from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

CLUSTER_LABELS: dict[int, str] = {
    0: "general daily load",
    1: "evening-peaking clients",
    2: "business-hour load",
    3: "overnight-heavy load",
}


class DualForecastStore:
    def __init__(self) -> None:
        brocode_path = BASE_DIR / "artifacts" / "brocode_forecasts.csv"
        price_path = BASE_DIR / "artifacts" / "price_forecast_extended.csv"
        cluster_path = BASE_DIR / "current_cluster_forecasting.csv"

        if not brocode_path.exists():
            raise FileNotFoundError(
                "artifacts/brocode_forecasts.csv not found. "
                "Run: python scripts/generate_synthetic_forecasts.py"
            )
        if not price_path.exists():
            raise FileNotFoundError(
                "artifacts/price_forecast_extended.csv not found. "
                "Run: python scripts/generate_synthetic_forecasts.py"
            )

        self._brocode = pd.read_csv(brocode_path)
        self._brocode["datetime"] = pd.to_datetime(self._brocode["datetime"])

        self._price = pd.read_csv(price_path)
        self._price["Date"] = pd.to_datetime(self._price["Date"], format="mixed")

        self._clients = pd.read_csv(cluster_path)
        self._clients["Client"] = self._clients["Client"].str.strip().str.upper()

    @staticmethod
    def _normalize(client_id: str) -> str:
        text = client_id.strip().upper()
        if text.startswith("MT_") and text.split("_", 1)[1].isdigit():
            return f"MT_{int(text.split('_', 1)[1]):03d}"
        if text.isdigit():
            return f"MT_{int(text):03d}"
        return text

    def list_clients(self) -> list[dict[str, Any]]:
        # Use clients present in the generated brocode_forecasts artifact
        available = set(self._brocode["client_id"].unique())
        rows = []
        for _, r in self._clients.sort_values("Client").iterrows():
            cid = str(r["Client"])
            if cid not in available:
                continue
            cluster = int(r["Cluster"])
            rows.append({
                "client_id": cid,
                "cluster": cluster,
                "cluster_id": cluster,
                "model": str(r["Best_Model"]),
                "best_mape": float(r["Best_wMAPE"]),
                "cluster_label": CLUSTER_LABELS.get(cluster, "unknown"),
                "mean_hourly_kwh": float(r["client_mean"]) / 24.0,
            })
        return rows

    def get_dual_forecast(self, client_id: str) -> dict[str, Any]:
        cid = self._normalize(client_id)
        rows = self._brocode[self._brocode["client_id"] == cid].sort_values("hour_offset")
        if rows.empty:
            raise KeyError(f"Client {cid} not found in brocode_forecasts")

        brocode_rows = [
            {
                "hour": int(r["hour_offset"]),
                "datetime": r["datetime"].isoformat(),
                "actual_kwh": float(r["actual_kwh"]),
                "pred_kwh": max(0.0, float(r["brocode_pred_kwh"])),
            }
            for _, r in rows.iterrows()
        ]
        chicken_rows = [
            {
                "hour": int(r["hour_offset"]),
                "datetime": r["datetime"].isoformat(),
                "pred_kwh": max(0.0, float(r["chicken_pred_kwh"])),
            }
            for _, r in rows.iterrows()
        ]

        actual = rows["actual_kwh"].to_numpy(dtype=float)
        brocode_pred = rows["brocode_pred_kwh"].clip(lower=0).to_numpy(dtype=float)
        chicken_pred = rows["chicken_pred_kwh"].clip(lower=0).to_numpy(dtype=float)

        def mape(a: np.ndarray, p: np.ndarray) -> float:
            mask = a > 0.01
            if not mask.any():
                return 100.0
            return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)

        brocode_mape = mape(actual, brocode_pred)
        chicken_mape = mape(actual, chicken_pred)

        meta = self._clients[self._clients["Client"] == cid]
        cluster = int(meta["Cluster"].iloc[0]) if not meta.empty else 0
        model = str(meta["Best_Model"].iloc[0]) if not meta.empty else "XGBoost-Optuna"
        best_mape_val = float(meta["Best_wMAPE"].iloc[0]) if not meta.empty else 0.0

        return {
            "client_id": cid,
            "cluster": cluster,
            "cluster_label": CLUSTER_LABELS.get(cluster, "unknown"),
            "model": model,
            "best_mape": best_mape_val,
            "brocode_rows": brocode_rows,
            "chicken_rows": chicken_rows,
            "metrics": {
                "brocode_mape": round(brocode_mape, 2),
                "chicken_mape": round(chicken_mape, 2),
                "brocode_accuracy_pct": round(max(0.0, 100.0 - brocode_mape), 2),
                "chicken_accuracy_pct": round(max(0.0, 100.0 - chicken_mape), 2),
            },
        }

    def get_price_forecast(self) -> dict[str, Any]:
        future_mask = self._price["Actual_Price"].isna()
        future = self._price[future_mask].sort_values("Date").head(7)
        historical = self._price[~future_mask].tail(30)

        def to_dict(r: Any) -> dict[str, Any]:
            actual = None if pd.isna(r["Actual_Price"]) else float(r["Actual_Price"])
            return {
                "date": str(r["Date"].date()),
                "forecast_price": float(r["Forecast_Price"]),
                "lower": float(r["Forecast_Lower"]),
                "upper": float(r["Forecast_Upper"]),
                "actual": actual,
            }

        return {
            "historical": [to_dict(r) for _, r in historical.iterrows()],
            "future": [to_dict(r) for _, r in future.iterrows()],
            "unit": "EUR/kWh",
            "source": "OMIE via Prophet (Brocode Innovation)",
        }


@lru_cache(maxsize=1)
def get_dual_store() -> DualForecastStore:
    return DualForecastStore()
