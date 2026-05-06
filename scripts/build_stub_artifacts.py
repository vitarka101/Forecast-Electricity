#!/usr/bin/env python3
"""
Generate minimal stub artifacts so ForecastArtifactStore loads without error.
The new dual-forecast endpoints (brocode_forecasts.csv) are the real data source;
these stubs only exist to satisfy the legacy store's file-existence check.
"""
import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def build() -> None:
    # forecast_map.json — minimal valid structure
    forecast_map = {
        "metadata": {
            "generated_at": "stub",
            "client_count": 0,
            "cluster_count": 4,
            "winner": {"model_name": "XGBoost-Optuna", "forecast_accuracy_pct": 91.0},
        },
        "clients": {},
        "clusters": {},
        "system": {
            "forecast_hourly": [],
            "forecast_daily": {"total_kwh": 0.0, "source_rows": 0},
            "generated_at": "stub",
        },
    }
    (ARTIFACTS / "forecast_map.json").write_text(json.dumps(forecast_map), encoding="utf-8")
    print("Written forecast_map.json")

    # client_profiles.csv
    (ARTIFACTS / "client_profiles.csv").write_text(
        "client_id,cluster_id,mean_hourly_kwh\n", encoding="utf-8"
    )
    print("Written client_profiles.csv")

    # cluster_profiles.csv
    (ARTIFACTS / "cluster_profiles.csv").write_text(
        "cluster_id,label,client_count\n", encoding="utf-8"
    )
    print("Written cluster_profiles.csv")

    # model_comparison.json
    model_comparison = {
        "summary": "Brocode vs chicken_dinner comparison",
        "brocode_accuracy_pct": 91.0,
        "chicken_dinner_accuracy_pct": 71.64,
        "improvement_pp": 19.36,
    }
    (ARTIFACTS / "model_comparison.json").write_text(
        json.dumps(model_comparison), encoding="utf-8"
    )
    print("Written model_comparison.json")

    # history_aggregates.parquet (empty, optional)
    try:
        import pandas as pd
        pd.DataFrame(columns=["client_id", "week", "total_kwh"]).to_parquet(
            ARTIFACTS / "history_aggregates.parquet", index=False
        )
        print("Written history_aggregates.parquet")
    except Exception:
        pass


if __name__ == "__main__":
    build()
    print("Stub artifacts ready.")
