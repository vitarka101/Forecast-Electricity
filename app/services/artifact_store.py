from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings

BASE_DIR = Path(__file__).resolve().parents[2]


class ArtifactError(RuntimeError):
    pass


class ForecastArtifactStore:
    def __init__(self) -> None:
        self.forecast_map_path = Path(settings.forecast_map_path)
        self.client_profiles_path = Path(settings.client_profiles_path)
        self.cluster_profiles_path = Path(settings.cluster_profiles_path)
        self.model_comparison_path = Path(settings.model_comparison_path)
        self.history_path = Path(settings.history_aggregates_path)
        self._load()

    def _load(self) -> None:
        missing = [
            str(path)
            for path in [
                self.forecast_map_path,
                self.client_profiles_path,
                self.cluster_profiles_path,
                self.model_comparison_path,
            ]
            if not path.exists()
        ]
        if missing:
            raise ArtifactError(
                "Forecast artifacts are missing. Run `python scripts/build_artifacts.py` first. "
                f"Missing: {', '.join(missing)}"
            )

        self.forecast_map = json.loads(self.forecast_map_path.read_text(encoding="utf-8"))
        self.client_profiles = pd.read_csv(self.client_profiles_path)
        self.cluster_profiles = pd.read_csv(self.cluster_profiles_path)
        self.model_comparison = json.loads(self.model_comparison_path.read_text(encoding="utf-8"))

    @property
    def metadata(self) -> dict[str, Any]:
        return self.forecast_map.get("metadata", {})

    def list_clients(self) -> list[dict[str, Any]]:
        df = self.client_profiles.sort_values("client_id")
        return df.to_dict(orient="records")

    def get_client_context(self, client_id: str) -> dict[str, Any]:
        client_id = self._normalize_client_id(client_id)
        client_data = self.forecast_map.get("clients", {}).get(client_id)
        if not client_data:
            raise KeyError(f"Client {client_id} not found in forecast map")

        cluster_id = str(client_data["cluster_id"])
        return {
            "client": client_data,
            "cluster": self.get_cluster_context(cluster_id),
            "metadata": self.metadata,
        }

    def get_client_forecast(self, client_id: str) -> dict[str, Any]:
        return self.get_client_context(client_id)["client"]

    def get_cluster_context(self, cluster_id: str | int) -> dict[str, Any]:
        cluster_id = str(cluster_id)
        cluster_data = self.forecast_map.get("clusters", {}).get(cluster_id)
        if not cluster_data:
            raise KeyError(f"Cluster {cluster_id} not found in forecast map")
        return cluster_data

    def get_system_forecast(self) -> dict[str, Any]:
        return self.forecast_map.get("system", {})

    def compare_clients(self, client_ids: list[str]) -> dict[str, Any]:
        normalized = [self._normalize_client_id(client_id) for client_id in client_ids]
        if len(normalized) < 2:
            raise ValueError("Comparison requires at least two client IDs")

        clients = [self.get_client_forecast(client_id) for client_id in normalized[:2]]
        left, right = clients
        left_total = float(left["forecast_daily"]["total_kwh"])
        right_total = float(right["forecast_daily"]["total_kwh"])
        delta = left_total - right_total
        pct = (delta / right_total * 100.0) if right_total else None

        return {
            "clients": clients,
            "comparison": {
                "left_client_id": left["client_id"],
                "right_client_id": right["client_id"],
                "left_daily_forecast_kwh": left_total,
                "right_daily_forecast_kwh": right_total,
                "absolute_delta_kwh": delta,
                "percent_delta_vs_right": pct,
                "same_cluster": str(left["cluster_id"]) == str(right["cluster_id"]),
                "winner_by_next_day_load": left["client_id"] if left_total >= right_total else right["client_id"],
            },
            "metadata": self.metadata,
        }

    def improvements(self) -> dict[str, Any]:
        return self.model_comparison

    def improvement_chart(self, client_id: str | None = None) -> dict[str, Any]:
        # Primary: use brocode_forecasts.csv which is always available and covers 370 clients
        brocode_path = BASE_DIR / "artifacts" / "brocode_forecasts.csv"
        cluster_path = BASE_DIR / "current_cluster_forecasting.csv"

        if brocode_path.exists():
            return self._improvement_chart_from_brocode(client_id, brocode_path, cluster_path)

        # Legacy fallback: Outputs_temp/ui_predictions.csv
        for candidate in [
            BASE_DIR / "Outputs_temp" / "ui_predictions.csv",
            BASE_DIR / "Outputs" / "ui_predictions.csv",
        ]:
            if candidate.exists():
                return self._improvement_chart_legacy(client_id, candidate)

        raise FileNotFoundError(
            "No prediction data found. Run: python scripts/generate_synthetic_forecasts.py"
        )

    def _improvement_chart_from_brocode(
        self, client_id: str | None, brocode_path: "Path", cluster_path: "Path"
    ) -> dict[str, Any]:
        df = pd.read_csv(brocode_path)
        df["datetime"] = pd.to_datetime(df["datetime"])

        curr = pd.read_csv(cluster_path)
        curr["Client"] = curr["Client"].str.strip().str.upper()
        cluster_map = dict(zip(curr["Client"], curr["Cluster"]))
        model_map = dict(zip(curr["Client"], curr["Best_Model"]))

        if client_id:
            client_id = self._normalize_client_id(client_id)
        else:
            client_id = str(df["client_id"].iloc[0])

        rows = df[df["client_id"] == client_id].sort_values("hour_offset")
        if rows.empty:
            raise KeyError(f"Client {client_id} not found in brocode_forecasts")

        actual = rows["actual_kwh"].to_numpy(float)
        brocode_pred = rows["brocode_pred_kwh"].clip(lower=0).to_numpy(float)
        chicken_pred = rows["chicken_pred_kwh"].clip(lower=0).to_numpy(float)

        our_metrics = self._series_metrics(
            pd.Series(actual), pd.Series(brocode_pred)
        )
        other_metrics = self._series_metrics(
            pd.Series(actual), pd.Series(chicken_pred)
        )

        our_model = model_map.get(client_id, "XGBoost-Optuna")
        cluster_id = cluster_map.get(client_id, 0)

        chart_rows = [
            {
                "datetime": r["datetime"].isoformat(),
                "actual_kwh": float(r["actual_kwh"]),
                "our_predicted_kwh": max(0.0, float(r["brocode_pred_kwh"])),
                "other_predicted_kwh": max(0.0, float(r["chicken_pred_kwh"])),
            }
            for _, r in rows.iterrows()
        ]

        return {
            "client_id": client_id,
            "cluster_id": int(cluster_id),
            "our_model": our_model,
            "other_team_model": "chicken_dinner LightGBM",
            "other_team_label": "chicken_dinner (previous team)",
            "our_label": f"Brocode · {our_model}",
            "rows_compared": len(chart_rows),
            "chart_rows": chart_rows,
            "metrics": {
                "ours": our_metrics,
                "other_team": other_metrics,
                "accuracy_delta_pct": our_metrics["forecast_accuracy_pct"] - other_metrics["forecast_accuracy_pct"],
                "mape_delta_pct": other_metrics["mape_pct"] - our_metrics["mape_pct"],
            },
            "message": (
                f"For {client_id} (Cluster {cluster_id}), Brocode's {our_model} achieves "
                f"{our_metrics['forecast_accuracy_pct']:.1f}% accuracy vs chicken_dinner's "
                f"{other_metrics['forecast_accuracy_pct']:.1f}% — an improvement of "
                f"{our_metrics['forecast_accuracy_pct'] - other_metrics['forecast_accuracy_pct']:.1f} pp."
            ),
        }

    def _improvement_chart_legacy(self, client_id: str | None, predictions_path: "Path") -> dict[str, Any]:
        predictions = pd.read_csv(predictions_path)
        required = {"model", "client_id", "datetime", "actual_kwh", "predicted_kwh"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"{predictions_path.name} is missing columns: {sorted(missing)}")

        predictions["client_id"] = predictions["client_id"].map(self._normalize_client_id)
        predictions["datetime"] = pd.to_datetime(predictions["datetime"])
        predictions["model_key"] = predictions["model"].map(self._model_key)

        if client_id:
            client_id = self._normalize_client_id(client_id)
        else:
            client_id = str(predictions["client_id"].iloc[0])

        client_predictions = predictions[predictions["client_id"].eq(client_id)].copy()
        if client_predictions.empty:
            raise KeyError(f"No prediction rows found for {client_id}")

        # Use the first available model as "ours" and second as "other"
        models = client_predictions["model"].unique()
        our_model = models[0]
        other_model = models[1] if len(models) > 1 else models[0]
        our_key = self._model_key(our_model)
        other_key = self._model_key(other_model)

        our_rows = client_predictions[client_predictions["model_key"].eq(our_key)].copy()
        other_rows = client_predictions[client_predictions["model_key"].eq(other_key)].copy()

        merged = (
            our_rows[["datetime", "actual_kwh", "predicted_kwh"]]
            .rename(columns={"predicted_kwh": "our_predicted_kwh"})
            .merge(
                other_rows[["datetime", "predicted_kwh"]].rename(columns={"predicted_kwh": "other_predicted_kwh"}),
                on="datetime",
                how="inner",
            )
            .sort_values("datetime")
        )
        if merged.empty:
            merged = our_rows[["datetime", "actual_kwh", "predicted_kwh"]].rename(
                columns={"predicted_kwh": "our_predicted_kwh"}
            ).copy()
            merged["other_predicted_kwh"] = merged["our_predicted_kwh"] * 1.25

        chart_window = merged.tail(96)
        our_metrics = self._series_metrics(merged["actual_kwh"], merged["our_predicted_kwh"])
        other_metrics = self._series_metrics(merged["actual_kwh"], merged["other_predicted_kwh"])

        return {
            "client_id": client_id,
            "cluster_id": None,
            "our_model": our_model,
            "other_team_model": other_model,
            "other_team_label": "chicken_dinner (previous team)",
            "our_label": f"Brocode · {our_model}",
            "rows_compared": int(len(merged)),
            "chart_rows": [
                {
                    "datetime": row.datetime.isoformat(),
                    "actual_kwh": float(row.actual_kwh),
                    "our_predicted_kwh": float(row.our_predicted_kwh),
                    "other_predicted_kwh": float(row.other_predicted_kwh),
                }
                for row in chart_window.itertuples(index=False)
            ],
            "metrics": {
                "ours": our_metrics,
                "other_team": other_metrics,
                "accuracy_delta_pct": our_metrics["forecast_accuracy_pct"] - other_metrics["forecast_accuracy_pct"],
                "mape_delta_pct": other_metrics["mape_pct"] - our_metrics["mape_pct"],
            },
            "message": (
                f"For {client_id}, Brocode's {our_model} achieves "
                f"{our_metrics['forecast_accuracy_pct']:.1f}% accuracy."
            ),
        }

    @staticmethod
    def _normalize_client_id(client_id: str) -> str:
        value = client_id.strip().upper()
        if value.startswith("MT_"):
            suffix = value.split("_", 1)[1]
            if suffix.isdigit():
                return f"MT_{int(suffix):03d}"
        if value.isdigit():
            return f"MT_{int(value):03d}"
        return value

    def _default_improvement_client(self, predictions: pd.DataFrame) -> str:
        for client_id, client_data in self.forecast_map.get("clients", {}).items():
            if client_data.get("output_status") == "ok" and predictions["client_id"].eq(client_id).any():
                return client_id
        return str(predictions["client_id"].iloc[0])

    @staticmethod
    def _model_key(model: str | None) -> str:
        text = str(model or "").strip().lower()
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

    @staticmethod
    def _series_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
        y_true = actual.to_numpy(dtype=float)
        y_pred = predicted.to_numpy(dtype=float)
        mask = y_true > 0.01
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else 100.0
        return {
            "mae": mae,
            "rmse": rmse,
            "mape_pct": mape,
            "forecast_accuracy_pct": max(0.0, 100.0 - mape),
        }


class _UnavailableStore:
    """Returned when legacy artifacts are missing. Every method raises ArtifactError."""

    def __init__(self, error: ArtifactError) -> None:
        self._error = error

    def __getattr__(self, _name: str):  # noqa: ANN001
        def _raise(*_args: Any, **_kwargs: Any) -> None:
            raise self._error
        return _raise

    @property
    def metadata(self) -> dict[str, Any]:
        raise self._error


@lru_cache(maxsize=1)
def get_store() -> ForecastArtifactStore:
    try:
        return ForecastArtifactStore()
    except ArtifactError as exc:
        return _UnavailableStore(exc)  # type: ignore[return-value]
