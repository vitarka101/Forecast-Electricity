from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


BASELINE_MODELS = [
    {
        "model_name": "Cluster LightGBM",
        "forecast_accuracy_pct": 71.64,
        "mape_pct": 28.36,
        "rmse": None,
        "notes": "chicken_dinner cluster LightGBM weighted from reported cluster MAPEs.",
    },
    {
        "model_name": "Cluster Linear Regression",
        "forecast_accuracy_pct": None,
        "mape_pct": None,
        "rmse": None,
        "notes": "Baseline model retained as performance floor in chicken_dinner notebook/report.",
    },
]

NOTEBOOK_REFERENCE_MODELS = [
    {
        "model_name": "TFT",
        "forecast_accuracy_pct": 84.535,
        "mape_pct": 15.465,
        "rmse": 39.438,
        "notes": "Best notebook KPI, but only used as reference unless full deployable artifacts are generated.",
    },
    {
        "model_name": "XGBoost-Optuna",
        "forecast_accuracy_pct": 77.892,
        "mape_pct": 22.108,
        "rmse": 403.625,
        "notes": "Corrected all-client tabular model from energy_forecasting.ipynb.",
    },
    {
        "model_name": "LightGBM",
        "forecast_accuracy_pct": 71.018,
        "mape_pct": 28.982,
        "rmse": 399.440,
        "notes": "Corrected all-client LightGBM from energy_forecasting.ipynb.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build precomputed electricity forecast-map artifacts.")
    parser.add_argument("--data-path", default="LD2011_2014.txt")
    parser.add_argument("--omie-path", default="omie_marginal_price.csv")
    parser.add_argument("--routing-path", default="Outputs/routing_table.csv")
    parser.add_argument("--predictions-path", default="Outputs/ui_predictions.csv")
    parser.add_argument("--metrics-path", default="Outputs/model_metrics.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--client-limit", type=int, default=None, help="Use first N clients for smoke tests.")
    parser.add_argument("--validation-days", type=int, default=28)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly = load_hourly(Path(args.data_path), args.client_limit)
    price = load_omie(Path(args.omie_path))

    routing_path = Path(args.routing_path)
    predictions_path = Path(args.predictions_path)
    if routing_path.exists() and predictions_path.exists():
        artifacts = build_from_routing_outputs(
            hourly=hourly,
            routing_path=routing_path,
            predictions_path=predictions_path,
            metrics_path=Path(args.metrics_path),
            price=price,
        )
        (output_dir / "forecast_map.json").write_text(json.dumps(artifacts["forecast_map"], indent=2), encoding="utf-8")
        artifacts["client_profiles"].to_csv(output_dir / "client_profiles.csv", index=False)
        artifacts["cluster_profiles"].to_csv(output_dir / "cluster_profiles.csv", index=False)
        (output_dir / "model_comparison.json").write_text(json.dumps(artifacts["model_comparison"], indent=2), encoding="utf-8")
        artifacts["history"].to_parquet(output_dir / "history_aggregates.parquet", index=False)

        metadata = artifacts["forecast_map"]["metadata"]
        print(f"Built routed artifacts for {metadata['client_count']} clients")
        print(f"Clients with model output: {metadata['clients_with_model_output']}")
        print(f"Routing source: {routing_path}")
        print(f"Prediction source: {predictions_path}")
        print(f"Output: {output_dir.resolve()}")
        return

    clients = hourly.columns.tolist()
    cluster_map, client_profiles = build_client_profiles(hourly)

    validation_hours = max(args.validation_days * 24, 24)
    validation_hours = min(validation_hours, max(24, len(hourly) // 5))
    train = hourly.iloc[:-validation_hours]
    valid = hourly.iloc[-validation_hours:]

    candidates = evaluate_candidates(hourly, train, valid)
    winner = max(candidates, key=lambda item: item["forecast_accuracy_pct"])

    forecast_hourly = build_future_forecast(hourly, winner["model_key"])
    forecast_map = build_forecast_map(hourly, forecast_hourly, cluster_map, client_profiles, winner)
    cluster_profiles = build_cluster_profiles(hourly, cluster_map, client_profiles)
    history = build_history_aggregates(hourly, cluster_map)
    comparison = build_model_comparison(candidates, winner, len(clients), price)

    (output_dir / "forecast_map.json").write_text(json.dumps(forecast_map, indent=2), encoding="utf-8")
    client_profiles.to_csv(output_dir / "client_profiles.csv", index=False)
    cluster_profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    (output_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    history.to_parquet(output_dir / "history_aggregates.parquet", index=False)

    print(f"Built artifacts for {len(clients)} clients")
    print(f"Winner: {winner['model_name']} | accuracy={winner['forecast_accuracy_pct']:.2f}% | MAPE={winner['mape_pct']:.2f}%")
    print(f"Output: {output_dir.resolve()}")


def build_from_routing_outputs(
    hourly: pd.DataFrame,
    routing_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    price: pd.DataFrame,
) -> dict:
    routing = load_routing_table(routing_path)
    predictions = load_prediction_outputs(predictions_path)
    metrics = load_metrics(metrics_path)

    client_profiles = build_routed_client_profiles(hourly, routing, predictions)
    cluster_profiles = build_routed_cluster_profiles(hourly, routing, predictions)
    forecast_map = build_routed_forecast_map(hourly, routing, predictions, client_profiles, cluster_profiles)
    history = build_routed_history_aggregates(hourly, routing)
    model_comparison = build_routed_model_comparison(routing, predictions, metrics, price)

    return {
        "forecast_map": forecast_map,
        "client_profiles": client_profiles,
        "cluster_profiles": cluster_profiles,
        "history": history,
        "model_comparison": model_comparison,
    }


def load_routing_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"client_id", "cluster", "assigned_cluster_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df[["client_id", "cluster", "assigned_cluster_model"]].copy()
    out["client_id"] = out["client_id"].map(normalize_client_id)
    out["cluster"] = out["cluster"].astype(str)
    out["assigned_cluster_model"] = out["assigned_cluster_model"].astype(str).str.strip()
    return out.drop_duplicates(subset=["client_id"]).sort_values("client_id").reset_index(drop=True)


def load_prediction_outputs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "client_id", "datetime", "predicted_kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df.copy()
    out["client_id"] = out["client_id"].map(normalize_client_id)
    out["model"] = out["model"].astype(str).str.strip()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["predicted_kwh"] = pd.to_numeric(out["predicted_kwh"], errors="coerce").clip(lower=0)
    if "actual_kwh" in out.columns:
        out["actual_kwh"] = pd.to_numeric(out["actual_kwh"], errors="coerce")
    if "split" not in out.columns:
        out["split"] = "model_output"
    return out.dropna(subset=["datetime", "predicted_kwh"]).sort_values(["client_id", "model", "datetime"])


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "MAPE", "SMAPE", "R2", "Forecast_Accuracy_pct"])
    df = pd.read_csv(path)
    if "model" not in df.columns:
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "MAPE", "SMAPE", "R2", "Forecast_Accuracy_pct"])
    return df


def normalize_client_id(value: str | int | float) -> str:
    text = str(value).strip().upper()
    if text.startswith("MT_"):
        suffix = text.split("_", 1)[1]
        if suffix.isdigit():
            return f"MT_{int(suffix):03d}"
    if text.isdigit():
        return f"MT_{int(text):03d}"
    return text


def matching_model_rows(predictions: pd.DataFrame, client_id: str, assigned_model: str) -> pd.DataFrame:
    client_rows = predictions[predictions["client_id"].eq(client_id)]
    if client_rows.empty:
        return client_rows
    assigned_key = model_key(assigned_model)
    matched = client_rows[client_rows["model"].map(model_key).eq(assigned_key)]
    return matched.sort_values("datetime")


def model_key(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("tft"):
        return "tft"
    if text in {"xgboost-optuna", "xgb-optuna", "xgb_optuna"}:
        return "xgboost-optuna"
    if text == "xgboost":
        return "xgboost"
    if text == "lightgbm":
        return "lightgbm"
    if "lag-24" in text or "naïve" in text or "naive" in text:
        return "naive-lag-24h"
    if text.startswith("deepar"):
        return "deepar"
    return text


def build_routed_client_profiles(
    hourly: pd.DataFrame,
    routing: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for route in routing.itertuples(index=False):
        client_id = route.client_id
        pred_rows = matching_model_rows(predictions, client_id, route.assigned_cluster_model)
        profile = historical_profile(hourly, client_id)
        output_status = "ok" if not pred_rows.empty else "missing_model_output"
        rows.append(
            {
                "client_id": client_id,
                "cluster_id": str(route.cluster),
                "assigned_model": route.assigned_cluster_model,
                "output_status": output_status,
                "model_output_rows": int(len(pred_rows)),
                "model_output_start": pred_rows["datetime"].min().isoformat() if not pred_rows.empty else None,
                "model_output_end": pred_rows["datetime"].max().isoformat() if not pred_rows.empty else None,
                **profile,
            }
        )
    return pd.DataFrame(rows)


def historical_profile(hourly: pd.DataFrame, client_id: str) -> dict:
    if client_id not in hourly.columns:
        return {
            "mean_hourly_kwh": 0.0,
            "median_hourly_kwh": 0.0,
            "max_hourly_kwh": 0.0,
            "peak_hour": None,
            "recent_daily_avg_kwh": 0.0,
            "recent_weekly_kwh": 0.0,
            "volatility_ratio": 0.0,
        }
    series = hourly[client_id].astype(float)
    recent = series.tail(24 * 28)
    mean = float(series.mean())
    return {
        "mean_hourly_kwh": mean,
        "median_hourly_kwh": float(series.median()),
        "max_hourly_kwh": float(series.max()),
        "peak_hour": int(series.groupby(series.index.hour).mean().idxmax()),
        "recent_daily_avg_kwh": float(recent.resample("D").sum().mean()),
        "recent_weekly_kwh": float(series.tail(24 * 7).sum()),
        "volatility_ratio": float(recent.std() / mean) if mean else 0.0,
    }


def build_routed_cluster_profiles(
    hourly: pd.DataFrame,
    routing: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cluster_id, group in routing.groupby("cluster"):
        clients = sorted(group["client_id"].tolist())
        available_clients = [client_id for client_id in clients if client_id in hourly.columns]
        if available_clients:
            total_series = hourly[available_clients].sum(axis=1)
            mean_hourly = float(total_series.mean())
            peak_hour = int(total_series.groupby(total_series.index.hour).mean().idxmax())
        else:
            mean_hourly = 0.0
            peak_hour = None
        predicted_count = 0
        for route in group.itertuples(index=False):
            if not matching_model_rows(predictions, route.client_id, route.assigned_cluster_model).empty:
                predicted_count += 1
        rows.append(
            {
                "cluster_id": str(cluster_id),
                "label": f"Routing cluster {cluster_id}",
                "client_count": int(len(clients)),
                "predicted_client_count": int(predicted_count),
                "mean_hourly_kwh": mean_hourly,
                "peak_hour": peak_hour,
                "assigned_models": ", ".join(sorted(group["assigned_cluster_model"].unique())),
                "top_clients": ", ".join(available_clients[:5]),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id")


def build_routed_forecast_map(
    hourly: pd.DataFrame,
    routing: pd.DataFrame,
    predictions: pd.DataFrame,
    client_profiles: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
) -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    clients = {}
    predicted_frames = []

    for route in routing.itertuples(index=False):
        client_id = route.client_id
        pred_rows = matching_model_rows(predictions, client_id, route.assigned_cluster_model)
        profile = client_profiles[client_profiles["client_id"].eq(client_id)].iloc[0].to_dict()
        forecast_rows = prediction_rows_for_ui(pred_rows)
        daily = daily_from_prediction_rows(pred_rows)
        clients[client_id] = {
            "client_id": client_id,
            "cluster_id": str(route.cluster),
            "assigned_model": route.assigned_cluster_model,
            "output_status": "ok" if not pred_rows.empty else "missing_model_output",
            "profile": clean(profile),
            "forecast_hourly": forecast_rows,
            "forecast_daily": daily,
            "model_output_summary": {
                "source": "Outputs/ui_predictions.csv",
                "matched_rows": int(len(pred_rows)),
                "output_start": pred_rows["datetime"].min().isoformat() if not pred_rows.empty else None,
                "output_end": pred_rows["datetime"].max().isoformat() if not pred_rows.empty else None,
                "split": sorted(pred_rows["split"].dropna().astype(str).unique().tolist()) if not pred_rows.empty else [],
            },
            "historical_weekly": weekly_history(hourly, client_id),
        }
        if not pred_rows.empty:
            tmp = pred_rows[["datetime", "predicted_kwh"]].copy()
            tmp["client_id"] = client_id
            tmp["cluster_id"] = str(route.cluster)
            predicted_frames.append(tmp)

    prediction_long = pd.concat(predicted_frames, ignore_index=True) if predicted_frames else pd.DataFrame(
        columns=["datetime", "predicted_kwh", "client_id", "cluster_id"]
    )
    clusters = build_routed_cluster_forecasts(routing, cluster_profiles, prediction_long)
    system = build_routed_system_forecast(prediction_long, generated_at)

    return {
        "metadata": {
            "generated_at": generated_at,
            "source_data_start": hourly.index.min().isoformat(),
            "source_data_end": hourly.index.max().isoformat(),
            "prediction_source": "Outputs/ui_predictions.csv",
            "routing_source": "Outputs/routing_table.csv",
            "client_count": int(len(routing)),
            "clients_with_model_output": int(sum(client["output_status"] == "ok" for client in clients.values())),
            "cluster_count": int(routing["cluster"].nunique()),
            "routing_models": sorted(routing["assigned_cluster_model"].unique().tolist()),
            "note": "Client forecasts are selected from the model named in routing_table.assigned_cluster_model.",
        },
        "clients": clients,
        "clusters": clusters,
        "system": system,
    }


def prediction_rows_for_ui(pred_rows: pd.DataFrame, limit: int = 24) -> list[dict]:
    if pred_rows.empty:
        return []
    window = pred_rows.sort_values("datetime").tail(limit)
    rows = []
    for row in window.itertuples(index=False):
        item = {
            "datetime": row.datetime.isoformat(),
            "predicted_kwh": float(row.predicted_kwh),
        }
        if hasattr(row, "actual_kwh") and pd.notna(row.actual_kwh):
            item["actual_kwh"] = float(row.actual_kwh)
        if hasattr(row, "model"):
            item["model"] = row.model
        if hasattr(row, "split"):
            item["split"] = row.split
        rows.append(item)
    return rows


def daily_from_prediction_rows(pred_rows: pd.DataFrame) -> dict:
    if pred_rows.empty:
        return {"date": None, "total_kwh": 0.0, "avg_hourly_kwh": 0.0, "source_rows": 0}
    window = pred_rows.sort_values("datetime").tail(24)
    total = float(window["predicted_kwh"].sum())
    return {
        "date": str(window["datetime"].max().date()),
        "total_kwh": total,
        "avg_hourly_kwh": total / max(len(window), 1),
        "source_rows": int(len(window)),
    }


def weekly_history(hourly: pd.DataFrame, client_id: str) -> dict:
    if client_id not in hourly.columns:
        return {"last_7_days_kwh": 0.0, "previous_7_days_kwh": 0.0}
    series = hourly[client_id]
    return {
        "last_7_days_kwh": float(series.tail(24 * 7).sum()),
        "previous_7_days_kwh": float(series.tail(24 * 14).head(24 * 7).sum()),
    }


def build_routed_cluster_forecasts(
    routing: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    prediction_long: pd.DataFrame,
) -> dict:
    clusters = {}
    for cluster_id, group in routing.groupby("cluster"):
        cluster_id = str(cluster_id)
        pred = prediction_long[prediction_long["cluster_id"].eq(cluster_id)]
        if pred.empty:
            hourly_rows = []
            daily = {"date": None, "total_kwh": 0.0, "avg_hourly_kwh": 0.0, "source_rows": 0}
        else:
            summed = pred.groupby("datetime", as_index=False)["predicted_kwh"].sum().sort_values("datetime")
            hourly_rows = prediction_rows_for_ui(summed, limit=24)
            daily = daily_from_prediction_rows(summed)
        profile = cluster_profiles[cluster_profiles["cluster_id"].astype(str).eq(cluster_id)].iloc[0].to_dict()
        clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "clients": sorted(group["client_id"].tolist()),
            "profile": clean(profile),
            "forecast_hourly": hourly_rows,
            "forecast_daily": daily,
        }
    return clusters


def build_routed_system_forecast(prediction_long: pd.DataFrame, generated_at: str) -> dict:
    if prediction_long.empty:
        return {
            "generated_at": generated_at,
            "forecast_hourly": [],
            "forecast_daily": {"date": None, "total_kwh": 0.0, "avg_hourly_kwh": 0.0, "source_rows": 0},
        }
    summed = prediction_long.groupby("datetime", as_index=False)["predicted_kwh"].sum().sort_values("datetime")
    return {
        "generated_at": generated_at,
        "forecast_hourly": prediction_rows_for_ui(summed, limit=24),
        "forecast_daily": daily_from_prediction_rows(summed),
    }


def build_routed_history_aggregates(hourly: pd.DataFrame, routing: pd.DataFrame) -> pd.DataFrame:
    route_clients = [client_id for client_id in routing["client_id"].tolist() if client_id in hourly.columns]
    if not route_clients:
        return pd.DataFrame(columns=["entity_type", "entity_id", "grain", "datetime", "load_kwh"])
    filtered = hourly[route_clients]
    frames = [
        long_history(filtered.tail(24 * 30), "client", "hourly_recent"),
        long_history(filtered.resample("D").sum(), "client", "daily"),
        long_history(filtered.resample("W").sum(), "client", "weekly"),
    ]
    cluster_hourly = pd.DataFrame(
        {
            str(cluster_id): filtered[[client for client in group["client_id"] if client in filtered.columns]].sum(axis=1)
            for cluster_id, group in routing.groupby("cluster")
        },
        index=filtered.index,
    )
    frames.extend(
        [
            long_history(cluster_hourly.tail(24 * 30), "cluster", "hourly_recent"),
            long_history(cluster_hourly.resample("D").sum(), "cluster", "daily"),
            long_history(cluster_hourly.resample("W").sum(), "cluster", "weekly"),
        ]
    )
    return pd.concat(frames, ignore_index=True)


def build_routed_model_comparison(
    routing: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    price: pd.DataFrame,
) -> dict:
    model_rows = []
    if not metrics.empty:
        for row in metrics.to_dict(orient="records"):
            model_rows.append(
                {
                    "model_name": row.get("model"),
                    "forecast_accuracy_pct": none_or_float(row.get("Forecast_Accuracy_pct")),
                    "mape_pct": none_or_float(row.get("MAPE")),
                    "rmse": none_or_float(row.get("RMSE")),
                    "notes": "Model metric imported from Outputs/model_metrics.csv.",
                }
            )
    winner = max(
        model_rows,
        key=lambda row: row["forecast_accuracy_pct"] if row["forecast_accuracy_pct"] is not None else -1,
        default={"model_name": None, "forecast_accuracy_pct": None, "mape_pct": None, "rmse": None, "notes": None},
    )
    coverage = []
    for route in routing.itertuples(index=False):
        coverage.append(not matching_model_rows(predictions, route.client_id, route.assigned_cluster_model).empty)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "project": "chicken_dinner",
            "models": BASELINE_MODELS,
            "notes": "Baseline used per-cluster LightGBM models and a Streamlit/FastAPI demo.",
        },
        "new_project": {
            "project": "routing-table model-output runtime",
            "client_count": int(len(routing)),
            "clients_with_model_output": int(sum(coverage)),
            "winner": winner,
            "models": model_rows,
            "routing_model_counts": routing["assigned_cluster_model"].value_counts().to_dict(),
        },
        "changes": [
            "Client answers now use routing_table.csv to pick that client's assigned model.",
            "Forecast values are read from Outputs/ui_predictions.csv, not generated by a naive lag heuristic.",
            "Clients without matching model-output rows are marked missing_model_output instead of receiving fabricated forecasts.",
            "Cluster and system totals aggregate only routed clients with available model outputs.",
        ],
        "data_notes": {
            "prediction_rows": int(len(predictions)),
            "prediction_clients": int(predictions["client_id"].nunique()),
            "omie_price_rows": int(len(price)),
        },
    }


def none_or_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def load_hourly(path: Path, client_limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    header = pd.read_csv(path, sep=";", nrows=0).columns.tolist()
    timestamp_col = header[0]
    client_cols = header[1:]
    if client_limit:
        client_cols = client_cols[:client_limit]

    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        usecols=[timestamp_col, *client_cols],
        index_col=timestamp_col,
    )
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    df = df.sort_index()
    hourly = df.resample("h").sum() / 4.0
    hourly = hourly[hourly.index < "2015-01-01"].astype("float32")
    hourly.columns = [str(col).strip().upper() for col in hourly.columns]
    return hourly


def load_omie(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "price_eur_kwh"])
    df = pd.read_csv(path)
    if not {"DATE", "CONCEPT"}.issubset(df.columns):
        return pd.DataFrame(columns=["datetime", "price_eur_kwh"])
    hour_cols = [col for col in df.columns if col.startswith("H") and col[1:].isdigit() and 1 <= int(col[1:]) <= 24]
    melted = df[df["CONCEPT"].eq("PRICE_PT")].melt(
        id_vars=["DATE", "CONCEPT"],
        value_vars=hour_cols,
        var_name="hour",
        value_name="price_eur_mwh",
    )
    melted["DATE"] = pd.to_datetime(melted["DATE"])
    melted["hour"] = melted["hour"].str[1:].astype(int) - 1
    melted["datetime"] = melted.apply(lambda row: row["DATE"].replace(hour=int(row["hour"])), axis=1)
    melted["price_eur_kwh"] = pd.to_numeric(melted["price_eur_mwh"], errors="coerce") / 1000.0
    return melted[["datetime", "price_eur_kwh"]].dropna()


def evaluate_candidates(hourly: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame) -> list[dict]:
    profile = profile_prediction(train, valid.index)
    lag_24 = hourly.shift(24).loc[valid.index]
    lag_168 = hourly.shift(168).loc[valid.index]
    blended = blend_predictions(profile, lag_24, lag_168)

    candidates = [
        score_candidate("seasonal_profile", "Seasonal Profile Map", valid, profile, "Median by client, day-of-week, and hour."),
        score_candidate("lag_24", "Lag-24h Map", valid, lag_24, "Repeats the previous day's same hour."),
        score_candidate("lag_168", "Lag-168h Map", valid, lag_168, "Repeats the previous week's same hour."),
        score_candidate("blended_profile", "Blended Seasonal-Lag Map", valid, blended, "Weighted blend of weekly lag, daily lag, and seasonal profile."),
    ]
    return candidates


def score_candidate(model_key: str, model_name: str, actual: pd.DataFrame, pred: pd.DataFrame, notes: str) -> dict:
    aligned = pred.reindex(index=actual.index, columns=actual.columns).fillna(0).clip(lower=0)
    y_true = actual.to_numpy(dtype=float).ravel()
    y_pred = aligned.to_numpy(dtype=float).ravel()
    mask = y_true > 0.01
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else 100.0
    smape = float(np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9)) * 100.0)
    return {
        "model_key": model_key,
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "smape_pct": smape,
        "forecast_accuracy_pct": max(0.0, 100.0 - mape),
        "notes": notes,
    }


def profile_prediction(train: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    framed = train.copy()
    framed["dayofweek"] = framed.index.dayofweek
    framed["hour"] = framed.index.hour
    profile = framed.groupby(["dayofweek", "hour"])[train.columns].median()
    fallback = train.tail(24 * 28).median()
    rows = []
    for ts in target_index:
        key = (ts.dayofweek, ts.hour)
        rows.append(profile.loc[key] if key in profile.index else fallback)
    return pd.DataFrame(rows, index=target_index, columns=train.columns)


def blend_predictions(profile: pd.DataFrame, lag_24: pd.DataFrame, lag_168: pd.DataFrame) -> pd.DataFrame:
    lag_168_filled = lag_168.where(lag_168.notna(), lag_24).where(lambda frame: frame.notna(), profile)
    lag_24_filled = lag_24.where(lag_24.notna(), profile)
    return (0.55 * lag_168_filled + 0.25 * lag_24_filled + 0.20 * profile).clip(lower=0)


def build_future_forecast(hourly: pd.DataFrame, model_key: str) -> pd.DataFrame:
    future_index = pd.date_range(hourly.index.max() + pd.Timedelta(hours=1), periods=24, freq="h")
    profile = profile_prediction(hourly, future_index)
    lag_24 = pd.DataFrame(index=future_index, columns=hourly.columns, dtype=float)
    lag_168 = pd.DataFrame(index=future_index, columns=hourly.columns, dtype=float)
    for ts in future_index:
        lag_24.loc[ts] = hourly.loc[ts - pd.Timedelta(hours=24)] if ts - pd.Timedelta(hours=24) in hourly.index else np.nan
        lag_168.loc[ts] = hourly.loc[ts - pd.Timedelta(hours=168)] if ts - pd.Timedelta(hours=168) in hourly.index else np.nan

    models: dict[str, Callable[[], pd.DataFrame]] = {
        "seasonal_profile": lambda: profile,
        "lag_24": lambda: lag_24.where(lag_24.notna(), profile),
        "lag_168": lambda: lag_168.where(lag_168.notna(), profile),
        "blended_profile": lambda: blend_predictions(profile, lag_24, lag_168),
    }
    return models.get(model_key, models["blended_profile"])().astype(float).clip(lower=0)


def build_client_profiles(hourly: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    recent = hourly.tail(24 * 28)
    hourly_profile = hourly.groupby(hourly.index.hour).mean()
    mean_hourly = hourly.mean()
    peak_hour = hourly_profile.idxmax()
    recent_daily = recent.resample("D").sum()
    recent_weekly = hourly.tail(24 * 7).sum()
    volatility = recent.std() / recent.mean().replace(0, np.nan)

    night_share = hourly.loc[hourly.index.hour.isin(range(0, 6))].sum() / hourly.sum().replace(0, np.nan)
    business_share = hourly.loc[hourly.index.hour.isin(range(8, 19))].sum() / hourly.sum().replace(0, np.nan)
    weekend_share = hourly.loc[hourly.index.dayofweek >= 5].sum() / hourly.sum().replace(0, np.nan)

    cluster_map = assign_clusters(mean_hourly, peak_hour, night_share, business_share)
    labels = cluster_map.map(cluster_label)

    profiles = pd.DataFrame(
        {
            "client_id": hourly.columns,
            "cluster_id": [str(cluster_map[col]) for col in hourly.columns],
            "cluster_label": [labels[col] for col in hourly.columns],
            "mean_hourly_kwh": [float(mean_hourly[col]) for col in hourly.columns],
            "median_hourly_kwh": [float(hourly[col].median()) for col in hourly.columns],
            "max_hourly_kwh": [float(hourly[col].max()) for col in hourly.columns],
            "peak_hour": [int(peak_hour[col]) for col in hourly.columns],
            "recent_daily_avg_kwh": [float(recent_daily[col].mean()) for col in hourly.columns],
            "recent_weekly_kwh": [float(recent_weekly[col]) for col in hourly.columns],
            "volatility_ratio": [float(volatility[col]) if pd.notna(volatility[col]) else 0.0 for col in hourly.columns],
            "night_share": [float(night_share[col]) if pd.notna(night_share[col]) else 0.0 for col in hourly.columns],
            "business_hour_share": [float(business_share[col]) if pd.notna(business_share[col]) else 0.0 for col in hourly.columns],
            "weekend_share": [float(weekend_share[col]) if pd.notna(weekend_share[col]) else 0.0 for col in hourly.columns],
        }
    )
    return cluster_map.astype(str), profiles


def assign_clusters(
    mean_hourly: pd.Series,
    peak_hour: pd.Series,
    night_share: pd.Series,
    business_share: pd.Series,
) -> pd.Series:
    p95 = mean_hourly.quantile(0.95)
    p99 = mean_hourly.quantile(0.99)
    clusters = {}
    for client in mean_hourly.index:
        if mean_hourly[client] >= p99:
            clusters[client] = "5"
        elif mean_hourly[client] >= p95:
            clusters[client] = "4"
        elif business_share.get(client, 0) >= 0.52 and 8 <= int(peak_hour[client]) <= 18:
            clusters[client] = "2"
        elif night_share.get(client, 0) >= 0.30:
            clusters[client] = "3"
        elif int(peak_hour[client]) >= 18:
            clusters[client] = "1"
        else:
            clusters[client] = "0"
    return pd.Series(clusters)


def cluster_label(cluster_id: str) -> str:
    return {
        "0": "general daily load",
        "1": "evening-peaking clients",
        "2": "business-hour load",
        "3": "overnight-heavy load",
        "4": "large-load clients",
        "5": "industrial outliers",
    }.get(str(cluster_id), "unlabeled cluster")


def build_cluster_profiles(hourly: pd.DataFrame, cluster_map: pd.Series, client_profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, members in cluster_map.groupby(cluster_map).groups.items():
        member_list = sorted(members)
        subset = hourly[member_list]
        rows.append(
            {
                "cluster_id": str(cluster_id),
                "label": cluster_label(str(cluster_id)),
                "client_count": len(member_list),
                "mean_hourly_kwh": float(subset.sum(axis=1).mean()),
                "peak_hour": int(subset.sum(axis=1).groupby(subset.index.hour).mean().idxmax()),
                "top_clients": ", ".join(
                    client_profiles[client_profiles["client_id"].isin(member_list)]
                    .sort_values("mean_hourly_kwh", ascending=False)["client_id"]
                    .head(5)
                    .tolist()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id")


def build_forecast_map(
    hourly: pd.DataFrame,
    forecast_hourly: pd.DataFrame,
    cluster_map: pd.Series,
    client_profiles: pd.DataFrame,
    winner: dict,
) -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    clients = {}
    for client_id in forecast_hourly.columns:
        profile_row = client_profiles[client_profiles["client_id"].eq(client_id)].iloc[0].to_dict()
        hourly_rows = to_forecast_rows(forecast_hourly[[client_id]].rename(columns={client_id: "predicted_kwh"}))
        daily_total = float(forecast_hourly[client_id].sum())
        clients[client_id] = {
            "client_id": client_id,
            "cluster_id": str(cluster_map[client_id]),
            "profile": clean(profile_row),
            "forecast_hourly": hourly_rows,
            "forecast_daily": {
                "date": str(forecast_hourly.index[0].date()),
                "total_kwh": daily_total,
                "avg_hourly_kwh": daily_total / 24.0,
            },
            "historical_weekly": {
                "last_7_days_kwh": float(hourly[client_id].tail(24 * 7).sum()),
                "previous_7_days_kwh": float(hourly[client_id].tail(24 * 14).head(24 * 7).sum()),
            },
        }

    clusters = {}
    for cluster_id, members in cluster_map.groupby(cluster_map).groups.items():
        cluster_forecast = forecast_hourly[list(members)].sum(axis=1).to_frame("predicted_kwh")
        cluster_total = float(cluster_forecast["predicted_kwh"].sum())
        clusters[str(cluster_id)] = {
            "cluster_id": str(cluster_id),
            "clients": sorted(members),
            "profile": {
                "label": cluster_label(str(cluster_id)),
                "client_count": len(members),
                "mean_hourly_kwh": float(hourly[list(members)].sum(axis=1).mean()),
                "peak_hour": int(hourly[list(members)].sum(axis=1).groupby(hourly.index.hour).mean().idxmax()),
            },
            "forecast_hourly": to_forecast_rows(cluster_forecast),
            "forecast_daily": {
                "date": str(forecast_hourly.index[0].date()),
                "total_kwh": cluster_total,
                "avg_hourly_kwh": cluster_total / 24.0,
            },
        }

    system_forecast = forecast_hourly.sum(axis=1).to_frame("predicted_kwh")
    system_total = float(system_forecast["predicted_kwh"].sum())
    return {
        "metadata": {
            "generated_at": generated_at,
            "source_data_start": hourly.index.min().isoformat(),
            "source_data_end": hourly.index.max().isoformat(),
            "forecast_start": forecast_hourly.index.min().isoformat(),
            "forecast_end": forecast_hourly.index.max().isoformat(),
            "horizon_hours": 24,
            "client_count": len(forecast_hourly.columns),
            "cluster_count": len(clusters),
            "winner": clean(winner),
        },
        "clients": clients,
        "clusters": clusters,
        "system": {
            "generated_at": generated_at,
            "forecast_hourly": to_forecast_rows(system_forecast),
            "forecast_daily": {
                "date": str(forecast_hourly.index[0].date()),
                "total_kwh": system_total,
                "avg_hourly_kwh": system_total / 24.0,
            },
        },
    }


def to_forecast_rows(df: pd.DataFrame) -> list[dict]:
    return [
        {"datetime": idx.isoformat(), "predicted_kwh": float(row["predicted_kwh"])}
        for idx, row in df.iterrows()
    ]


def build_history_aggregates(hourly: pd.DataFrame, cluster_map: pd.Series) -> pd.DataFrame:
    frames = []
    recent_hourly = hourly.tail(24 * 30)
    frames.append(long_history(recent_hourly, "client", "hourly_recent"))
    frames.append(long_history(hourly.resample("D").sum(), "client", "daily"))
    frames.append(long_history(hourly.resample("W").sum(), "client", "weekly"))

    cluster_hourly = pd.DataFrame(
        {cluster_id: hourly[list(members)].sum(axis=1) for cluster_id, members in cluster_map.groupby(cluster_map).groups.items()},
        index=hourly.index,
    )
    frames.append(long_history(cluster_hourly.tail(24 * 30), "cluster", "hourly_recent"))
    frames.append(long_history(cluster_hourly.resample("D").sum(), "cluster", "daily"))
    frames.append(long_history(cluster_hourly.resample("W").sum(), "cluster", "weekly"))
    return pd.concat(frames, ignore_index=True)


def long_history(df: pd.DataFrame, entity_type: str, grain: str) -> pd.DataFrame:
    out = df.reset_index().melt(id_vars="datetime", var_name="entity_id", value_name="load_kwh")
    out["entity_type"] = entity_type
    out["grain"] = grain
    return out[["entity_type", "entity_id", "grain", "datetime", "load_kwh"]]


def build_model_comparison(candidates: list[dict], winner: dict, client_count: int, price: pd.DataFrame) -> dict:
    deployable_models = [
        {
            "model_name": item["model_name"],
            "forecast_accuracy_pct": item["forecast_accuracy_pct"],
            "mape_pct": item["mape_pct"],
            "rmse": item["rmse"],
            "notes": item["notes"],
        }
        for item in sorted(candidates, key=lambda row: row["forecast_accuracy_pct"], reverse=True)
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "project": "chicken_dinner",
            "models": BASELINE_MODELS,
            "notes": "Baseline used per-cluster LightGBM models and a Streamlit/FastAPI demo.",
        },
        "notebook_reference": {
            "project": "Brocode energy_forecasting.ipynb",
            "models": NOTEBOOK_REFERENCE_MODELS,
        },
        "new_project": {
            "project": "forecast-map runtime",
            "client_count": client_count,
            "winner": {
                "model_name": winner["model_name"],
                "forecast_accuracy_pct": winner["forecast_accuracy_pct"],
                "mape_pct": winner["mape_pct"],
                "rmse": winner["rmse"],
                "notes": winner["notes"],
            },
            "models": deployable_models,
        },
        "changes": [
            "Runtime no longer performs model inference; it serves a saved forecast map for fast client and cluster lookups.",
            "Every client is mapped to a deterministic behavior cluster and can be queried by chat.",
            "Client comparisons use retrieved forecasts and profiles, then optionally use Ollama for narration.",
            "Daily totals are derived from the 24-hour forecast, while weekly context comes from historical aggregates.",
            "The UI separates operational chat from the baseline-vs-new improvement review.",
        ],
        "data_notes": {
            "omie_price_rows": int(len(price)),
            "runtime_forecast_horizon_hours": 24,
        },
    }


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "model_key"}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value) if not isinstance(value, (dict, list, str)) else False:
        return None
    return value


if __name__ == "__main__":
    main()
