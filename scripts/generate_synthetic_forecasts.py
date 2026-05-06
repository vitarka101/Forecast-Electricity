#!/usr/bin/env python3
"""
Generate synthetic Brocode forecasts and extended price data.
Run once before starting the server:
    python scripts/generate_synthetic_forecasts.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ARTIFACTS = BASE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

MODEL_COL_MAP = {
    "Naïve (Lag-24h)": "Naïve (Lag-24h)",
    "XGBoost-Optuna": "XGBoost-Optuna",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}


def generate_brocode_forecasts() -> None:
    fcast = pd.read_csv(BASE / "Outputs_temp" / "client_1day_ahead_forecast.csv")
    fcast["datetime"] = pd.to_datetime(fcast["datetime"])

    best = pd.read_csv(BASE / "Outputs_temp" / "best_models_per_client.csv")
    curr = pd.read_csv(BASE / "current_cluster_forecasting.csv")
    curr["Client"] = curr["Client"].str.strip().str.upper()

    # Build lookup maps
    best_model_map = dict(zip(best["client_id"], best["model"]))
    cluster_map = dict(zip(curr["Client"], curr["Cluster"]))
    wmape_map = dict(zip(curr["Client"], curr["Best_wMAPE"]))
    curr_model_map = dict(zip(curr["Client"], curr["Best_Model"]))

    rows = []
    clients = sorted(fcast["client_id"].unique())
    available_cols = set(fcast.columns)

    for client_id in clients:
        rng = np.random.default_rng(seed=abs(hash(client_id)) % (2**31))
        client_rows = fcast[fcast["client_id"] == client_id].sort_values("datetime").reset_index(drop=True)

        # Pick best model column
        model = best_model_map.get(client_id, "XGBoost-Optuna")
        col = MODEL_COL_MAP.get(model, "XGBoost-Optuna")
        if col not in available_cols:
            col = "XGBoost-Optuna"

        actual = client_rows["actual_kwh"].to_numpy(dtype=float)
        brocode_pred = client_rows[col].to_numpy(dtype=float).clip(min=0)

        # Synthetic chicken_dinner: always worse than Brocode.
        # Compute Brocode MAPE first, then set chicken error to Brocode MAPE + 15-25pp extra.
        mask_a = actual > 0.01
        if mask_a.any():
            b_errors = np.abs(actual[mask_a] - brocode_pred[mask_a]) / actual[mask_a]
            brocode_mape = float(np.mean(b_errors))
        else:
            brocode_mape = 0.25

        # chicken error = brocode error + 15-25% absolute extra, per hour with sign variation
        extra_error = rng.uniform(0.15, 0.25, size=len(actual))
        error_sign = rng.choice([-1.0, 1.0], size=len(actual))
        # Compute per-hour chicken errors ensuring they sum to > brocode MAPE
        chicken_error_fraction = brocode_mape + extra_error
        chicken_pred = (actual * (1.0 + chicken_error_fraction * error_sign)).clip(min=0)

        cluster = int(cluster_map.get(client_id, 0))
        mape_val = float(wmape_map.get(client_id, 0.0))

        for h, row in client_rows.iterrows():
            rows.append({
                "client_id": client_id,
                "datetime": row["datetime"].isoformat(),
                "hour_offset": int(h),
                "actual_kwh": float(actual[h]),
                "brocode_pred_kwh": float(brocode_pred[h]),
                "chicken_pred_kwh": float(chicken_pred[h]),
                "brocode_model": model,
                "cluster": cluster,
                "best_mape": mape_val,
            })

    df = pd.DataFrame(rows)
    out_path = ARTIFACTS / "brocode_forecasts.csv"
    df.to_csv(out_path, index=False)
    print(f"Written {len(df)} rows for {df['client_id'].nunique()} clients → {out_path}")


def generate_price_forecast() -> None:
    price = pd.read_csv(BASE / "daily_price_forecast.csv")
    price["Date"] = pd.to_datetime(price["Date"])

    # Use May historical data as spring baseline
    may_hist = price[price["Date"].dt.month == 5]
    if len(may_hist) >= 5:
        base_price = float(may_hist["Forecast_Price"].mean())
        base_std = float(may_hist["Forecast_Price"].std())
    else:
        base_price = 0.041
        base_std = 0.004

    rng = np.random.default_rng(seed=42)
    future_rows = []
    for i in range(7):
        date = pd.Timestamp("2026-05-05") + pd.Timedelta(days=i)
        fp = float(np.clip(rng.normal(base_price, base_std * 0.5), 0.025, 0.075))
        future_rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Forecast_Price": round(fp, 6),
            "Forecast_Lower": round(fp * 0.85, 6),
            "Forecast_Upper": round(fp * 1.15, 6),
            "Actual_Price": None,
        })

    future_df = pd.DataFrame(future_rows)
    extended = pd.concat([price, future_df], ignore_index=True)
    out_path = ARTIFACTS / "price_forecast_extended.csv"
    extended.to_csv(out_path, index=False)
    print(f"Written {len(extended)} rows to {out_path} ({len(future_rows)} synthetic future days)")


if __name__ == "__main__":
    generate_brocode_forecasts()
    generate_price_forecast()
    print("Done.")
