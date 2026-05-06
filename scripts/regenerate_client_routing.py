from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
import holidays as hol_lib
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_PATH = PROJECT_ROOT / "LD2011_2014.txt"
OMIE_PATH = PROJECT_ROOT / "omie_marginal_price.csv"


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    pt_holidays = hol_lib.Portugal(years=range(2011, 2015))
    hol_set = set(pd.to_datetime(list(pt_holidays.keys())).normalize())

    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["is_holiday"] = df["datetime"].dt.normalize().isin(hol_set).astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["season"] = df["month"].map({12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4})
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    dr = pd.date_range(df["datetime"].min().normalize(), df["datetime"].max().normalize(), freq="D")
    tmp = pd.DataFrame({"date": dr})
    tmp["is_hol"] = tmp["date"].isin(hol_set).astype(int)
    tmp["next_hol"] = tmp.loc[tmp["is_hol"] == 1, "date"].reindex(tmp.index).bfill()
    tmp["dth"] = (tmp["next_hol"] - tmp["date"]).dt.days.clip(0, 30)
    df["days_to_holiday"] = df["datetime"].dt.normalize().map(tmp.set_index("date")["dth"]).fillna(0)

    df["temp_x_hour"] = df["temperature"] * df["hour_sin"]
    df["temp_x_dow"] = df["temperature"] * df["dow_sin"]
    return df


def add_lags_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["client_id", "datetime"]).copy()
    grp = df.groupby("client_id")["load_kwh"]
    df["lag_24"] = grp.shift(24)
    df["lag_48"] = grp.shift(48)
    df["lag_168"] = grp.shift(168)
    df["lag_336"] = grp.shift(336)
    df["rolling_mean_24"] = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=6).mean())
    df["rolling_std_24"] = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=6).std())
    df["rolling_mean_168"] = grp.transform(lambda x: x.shift(1).rolling(168, min_periods=24).mean())
    df["rolling_max_24"] = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=6).max())
    df["rolling_min_24"] = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=6).min())
    return df


def load_long_data() -> pd.DataFrame:
    df_raw = pd.read_csv(DATA_PATH, sep=";", index_col=0, parse_dates=True, decimal=",")
    df_raw.index.name = "datetime"
    df_raw = df_raw.sort_index()

    hourly = df_raw.resample("h").sum() / 4
    hourly = hourly[hourly.index < "2015-01-01"]

    long = hourly.reset_index().melt(id_vars="datetime", var_name="client_id", value_name="load_kwh")
    long = long.sort_values(["client_id", "datetime"]).reset_index(drop=True)

    first_active = (
        long[long["load_kwh"] > 0]
        .groupby("client_id")["datetime"]
        .min()
        .reset_index()
        .rename(columns={"datetime": "first_active"})
    )
    long = long.merge(first_active, on="client_id", how="left")
    long = long[long["datetime"] >= long["first_active"]].drop(columns=["first_active"]).copy()

    t80 = sorted(long["datetime"].unique())[int(long["datetime"].nunique() * 0.8)]
    clip = (
        long[long["datetime"] <= t80]
        .groupby("client_id")["load_kwh"]
        .quantile([0.005, 0.995])
        .unstack()
        .reset_index()
    )
    clip.columns = ["client_id", "lo", "hi"]
    long = long.merge(clip, on="client_id", how="left")
    long["load_kwh"] = long["load_kwh"].clip(lower=long["lo"], upper=long["hi"])
    long = long.drop(columns=["lo", "hi"])

    cache_session = requests_cache.CachedSession(PROJECT_ROOT / ".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    responses = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 38.72,
            "longitude": -9.14,
            "start_date": "2011-01-01",
            "end_date": "2014-12-31",
            "hourly": ["temperature_2m", "relative_humidity_2m"],
            "timezone": "Europe/Lisbon",
        },
    )
    hw = responses[0].Hourly()
    weather_df = pd.DataFrame(
        {
            "datetime": pd.date_range(
                start=pd.to_datetime(hw.Time(), unit="s", utc=True).tz_convert("Europe/Lisbon").tz_localize(None),
                periods=hw.Variables(0).ValuesAsNumpy().shape[0],
                freq="h",
            ),
            "temperature": hw.Variables(0).ValuesAsNumpy(),
            "humidity": hw.Variables(1).ValuesAsNumpy(),
        }
    )
    long = long.merge(weather_df, on="datetime", how="left")

    try:
        df_omie = pd.read_csv(OMIE_PATH)
        df_omie["DATE"] = pd.to_datetime(df_omie["DATE"])
        melted = df_omie.melt(id_vars=["DATE", "CONCEPT"], value_vars=[f"H{i}" for i in range(1, 25)], var_name="h", value_name="v")
        melted["hour"] = melted["h"].str.replace("H", "").astype(int) - 1
        melted["datetime"] = melted.apply(lambda r: r["DATE"].replace(hour=r["hour"]), axis=1)
        omie = melted[melted["CONCEPT"] == "PRICE_PT"][["datetime", "v"]].rename(columns={"v": "price_eur_kwh"})
        omie["price_eur_kwh"] /= 1000
    except Exception:
        dates = pd.date_range("2011-01-01", "2014-12-31 23:00", freq="h")
        h = np.arange(len(dates))
        omie = pd.DataFrame(
            {
                "datetime": dates,
                "price_eur_kwh": (
                    0.05
                    + 0.01 * np.sin(2 * np.pi * h / 24)
                    + 0.005 * np.sin(2 * np.pi * h / 8760)
                    + np.random.default_rng(SEED).normal(0, 0.005, len(dates))
                ).clip(0.01),
            }
        )

    long = long.merge(omie, on="datetime", how="left")
    long["price_eur_kwh"] = long["price_eur_kwh"].ffill().bfill()
    return add_calendar(long)


def build_features(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    cutoff_test = long["datetime"].max() - pd.DateOffset(months=1)
    cutoff_val = cutoff_test - pd.DateOffset(months=1)
    df_train_raw = long[long["datetime"] < cutoff_val].copy()

    client_scale = (
        df_train_raw.groupby("client_id")["load_kwh"]
        .agg(client_mean="mean", client_std="std")
        .reset_index()
    )
    client_scale["client_std"] = client_scale["client_std"].replace(0, 1).fillna(1)
    client_scale["log_mean_scale"] = np.log1p(client_scale["client_mean"])
    long = long.merge(client_scale[["client_id", "log_mean_scale"]], on="client_id", how="left")

    long_with_lags = add_lags_rolling(long)
    df_train_tab = long_with_lags[long_with_lags["datetime"] < cutoff_val].dropna().copy()
    df_test_tab = long_with_lags[long_with_lags["datetime"] >= cutoff_test].dropna().copy()

    le = LabelEncoder()
    le.fit(long["client_id"])
    df_train_tab["client_ord"] = le.transform(df_train_tab["client_id"])
    df_test_tab["client_ord"] = le.transform(df_test_tab["client_id"])

    scale_cols = client_scale[["client_id", "client_mean", "client_std", "log_mean_scale"]]
    df_train_tab = df_train_tab.merge(scale_cols, on="client_id", how="left")
    df_test_tab = df_test_tab.merge(scale_cols, on="client_id", how="left")

    feature_cols = [
        "client_ord",
        "log_mean_scale",
        "temperature",
        "humidity",
        "price_eur_kwh",
        "hour",
        "dayofweek",
        "month",
        "year",
        "is_holiday",
        "is_weekend",
        "season",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "days_to_holiday",
        "temp_x_hour",
        "temp_x_dow",
        "lag_24",
        "lag_48",
        "lag_168",
        "lag_336",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
        "rolling_max_24",
        "rolling_min_24",
    ]
    feature_cols = [c for c in feature_cols if c in df_test_tab.columns]
    return df_train_raw, df_test_tab, feature_cols, client_scale


def build_predictions(df_test_tab: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    xgb_model = joblib.load(OUTPUT_DIR / "xgb.pkl")
    lgb_model = joblib.load(OUTPUT_DIR / "lgb.pkl")
    best_xgb = joblib.load(OUTPUT_DIR / "xgb_optuna.pkl")

    predictions_df_tab = df_test_tab[["datetime", "client_id", "load_kwh"]].copy()
    predictions_df_tab["Na\u00efve (Lag-24h)"] = df_test_tab["lag_24"].values
    predictions_df_tab["XGBoost"] = xgb_model.predict(df_test_tab[feature_cols])
    predictions_df_tab["LightGBM"] = lgb_model.predict(df_test_tab[feature_cols])
    predictions_df_tab["XGBoost-Optuna"] = best_xgb.predict(df_test_tab[feature_cols])
    return predictions_df_tab


def score_clients(predictions_df_tab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cols = ["Na\u00efve (Lag-24h)", "XGBoost", "LightGBM", "XGBoost-Optuna"]
    rows = []
    for client_id, client_data in predictions_df_tab.groupby("client_id", sort=True):
        y_true = client_data["load_kwh"].to_numpy(dtype=float)
        positive_mask = y_true > 0.01
        for model_name in model_cols:
            y_pred = client_data[model_name].to_numpy(dtype=float)
            valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if valid_mask.sum() == 0:
                continue
            yt = y_true[valid_mask]
            yp = y_pred[valid_mask]
            pos = positive_mask[valid_mask]
            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            mape = np.mean(np.abs((yt[pos] - yp[pos]) / yt[pos])) * 100 if pos.sum() else np.nan
            smape = np.mean(2 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-8)) * 100
            rows.append(
                {
                    "client_id": client_id,
                    "model": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "SMAPE": smape,
                    "Forecast_Accuracy_pct": max(0.0, 100.0 - mape) if np.isfinite(mape) else np.nan,
                    "n_test_rows": int(valid_mask.sum()),
                }
            )

    client_model_metrics = pd.DataFrame(rows).sort_values(["client_id", "MAE", "model"]).reset_index(drop=True)
    best_models_per_client = client_model_metrics.groupby("client_id", as_index=False).first()
    return client_model_metrics, best_models_per_client


def build_routing(df_train_raw: pd.DataFrame, client_model_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_base = df_train_raw.copy()
    cluster_base["is_night"] = cluster_base["hour"].isin(list(range(0, 7)) + [22, 23])
    cluster_features = (
        cluster_base.groupby("client_id")
        .agg(
            client_mean=("load_kwh", "mean"),
            client_std=("load_kwh", "std"),
            client_median=("load_kwh", "median"),
            client_max=("load_kwh", "max"),
            zero_share=("load_kwh", lambda s: (s <= 0.01).mean()),
            day_mean=("load_kwh", lambda s: s[cluster_base.loc[s.index, "is_night"].eq(False)].mean()),
            night_mean=("load_kwh", lambda s: s[cluster_base.loc[s.index, "is_night"]].mean()),
            weekday_mean=("load_kwh", lambda s: s[cluster_base.loc[s.index, "is_weekend"].eq(0)].mean()),
            weekend_mean=("load_kwh", lambda s: s[cluster_base.loc[s.index, "is_weekend"].eq(1)].mean()),
        )
        .reset_index()
    )
    cluster_features["client_std"] = cluster_features["client_std"].fillna(0)
    cluster_features["log_mean_scale"] = np.log1p(cluster_features["client_mean"])
    cluster_features["cv"] = cluster_features["client_std"] / (cluster_features["client_mean"] + 1e-5)
    cluster_features["peak_to_mean"] = cluster_features["client_max"] / (cluster_features["client_mean"] + 1e-5)
    cluster_features["night_day_ratio"] = cluster_features["night_mean"] / (cluster_features["day_mean"] + 1e-5)
    cluster_features["weekend_weekday_ratio"] = cluster_features["weekend_mean"] / (cluster_features["weekday_mean"] + 1e-5)

    cluster_feature_cols = [
        "log_mean_scale",
        "cv",
        "peak_to_mean",
        "zero_share",
        "night_day_ratio",
        "weekend_weekday_ratio",
    ]
    cluster_features[cluster_feature_cols] = cluster_features[cluster_feature_cols].replace([np.inf, -np.inf], np.nan)
    cluster_features[cluster_feature_cols] = cluster_features[cluster_feature_cols].fillna(cluster_features[cluster_feature_cols].median())

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(cluster_features[cluster_feature_cols])
    kmeans = KMeans(n_clusters=min(4, len(cluster_features)), random_state=SEED, n_init=10)
    cluster_features["cluster"] = kmeans.fit_predict(x_scaled)

    cluster_model_scores = (
        client_model_metrics.merge(cluster_features[["client_id", "cluster"]], on="client_id", how="inner")
        .groupby(["cluster", "model"], as_index=False)
        .agg(
            avg_mae=("MAE", "mean"),
            median_mae=("MAE", "median"),
            avg_accuracy=("Forecast_Accuracy_pct", "mean"),
            n_clients=("client_id", "nunique"),
        )
    )
    cluster_assignments = (
        cluster_model_scores.sort_values(["cluster", "avg_mae", "model"])
        .groupby("cluster", as_index=False)
        .first()
        .rename(columns={"model": "assigned_cluster_model"})
    )
    routing_table = cluster_features[["client_id", "cluster"]].merge(
        cluster_assignments[["cluster", "assigned_cluster_model"]],
        on="cluster",
        how="left",
    )
    return cluster_features, cluster_model_scores, routing_table


def main() -> None:
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading and featurizing data...")
    long = load_long_data()
    df_train_raw, df_test_tab, feature_cols, _ = build_features(long)
    print(f"Clients in test table: {df_test_tab['client_id'].nunique():,}")
    print(f"Test rows: {len(df_test_tab):,}")

    print("Loading saved models and scoring clients...")
    predictions_df_tab = build_predictions(df_test_tab, feature_cols)
    client_model_metrics, best_models_per_client = score_clients(predictions_df_tab)

    print("Clustering clients and assigning cluster models...")
    cluster_features, cluster_model_scores, routing_table = build_routing(df_train_raw, client_model_metrics)

    predictions_df_tab.rename(columns={"load_kwh": "actual_kwh"}).to_csv(OUTPUT_DIR / "client_all_models_history.csv", index=False)
    predictions_df_tab.groupby("client_id").tail(24).reset_index(drop=True).rename(columns={"load_kwh": "actual_kwh"}).to_csv(
        OUTPUT_DIR / "client_1day_ahead_forecast.csv",
        index=False,
    )
    client_model_metrics.to_csv(OUTPUT_DIR / "client_model_metrics.csv", index=False)
    best_models_per_client.to_csv(OUTPUT_DIR / "best_models_per_client.csv", index=False)
    cluster_features.to_csv(OUTPUT_DIR / "client_cluster_features.csv", index=False)
    cluster_model_scores.to_csv(OUTPUT_DIR / "cluster_model_scores.csv", index=False)
    routing_table.to_csv(OUTPUT_DIR / "routing_table.csv", index=False)

    print("\nBest model counts:")
    print(best_models_per_client["model"].value_counts().to_string())
    print("\nCluster routing:")
    print(
        cluster_model_scores.sort_values(["cluster", "avg_mae"])
        .groupby("cluster", as_index=False)
        .first()[["cluster", "model", "avg_mae", "n_clients"]]
        .to_string(index=False)
    )
    print("\nSaved updated routing artifacts to outputs/")


if __name__ == "__main__":
    main()
