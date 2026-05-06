from __future__ import annotations

import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import xgboost as xgb
from retry_requests import retry
import holidays as hol_lib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
CHICKEN_ROOT = REPO_ROOT / "chicken_dinner" / "project A"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_DAY = pd.Timestamp("2014-12-31")
TRAIN_END = pd.Timestamp("2014-01-01")
FINAL_START = TARGET_DAY
FINAL_END = TARGET_DAY + pd.Timedelta(hours=23)


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
    df["rolling_std_168"] = grp.transform(lambda x: x.shift(1).rolling(168, min_periods=24).std())
    df["rolling_max_24"] = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=6).max())
    df["rolling_min_24"] = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=6).min())
    df["rolling_mean_same_hour_7d"] = grp.transform(lambda x: x.shift(24).rolling(7, min_periods=2).mean())
    return df


def load_long_data() -> pd.DataFrame:
    df_raw = pd.read_csv(PROJECT_ROOT / "LD2011_2014.txt", sep=";", index_col=0, parse_dates=True, decimal=",")
    df_raw.index.name = "datetime"
    hourly = df_raw.sort_index().resample("h").sum() / 4
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

    train_for_clip = long[long["datetime"] < TRAIN_END]
    clip = train_for_clip.groupby("client_id")["load_kwh"].quantile([0.005, 0.995]).unstack().reset_index()
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
        df_omie = pd.read_csv(PROJECT_ROOT / "omie_marginal_price.csv")
        df_omie["DATE"] = pd.to_datetime(df_omie["DATE"])
        melted = df_omie.melt(id_vars=["DATE", "CONCEPT"], value_vars=[f"H{i}" for i in range(1, 25)], var_name="h", value_name="v")
        melted["hour"] = melted["h"].str.replace("H", "").astype(int) - 1
        melted["datetime"] = melted.apply(lambda r: r["DATE"].replace(hour=r["hour"]), axis=1)
        omie = melted[melted["CONCEPT"] == "PRICE_PT"][["datetime", "v"]].rename(columns={"v": "price_eur_kwh"})
        omie["price_eur_kwh"] /= 1000
    except Exception:
        dates = pd.date_range("2011-01-01", "2014-12-31 23:00", freq="h")
        omie = pd.DataFrame({"datetime": dates, "price_eur_kwh": 0.05})

    long = long.merge(omie, on="datetime", how="left")
    long["temperature"] = long["temperature"].ffill().bfill()
    long["humidity"] = long["humidity"].ffill().bfill()
    long["price_eur_kwh"] = long["price_eur_kwh"].ffill().bfill()
    return add_calendar(long)


def make_model_frames(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    train_raw = long[long["datetime"] < TRAIN_END].copy()
    pre_forecast_raw = long[long["datetime"] < FINAL_START].copy()
    train_scale = train_raw.groupby("client_id")["load_kwh"].agg(client_mean="mean", client_std="std").reset_index()
    fallback_scale = pre_forecast_raw.groupby("client_id")["load_kwh"].agg(client_mean="mean", client_std="std").reset_index()
    scale = fallback_scale.merge(train_scale, on="client_id", how="left", suffixes=("_fallback", ""))
    scale["client_mean"] = scale["client_mean"].fillna(scale["client_mean_fallback"])
    scale["client_std"] = scale["client_std"].fillna(scale["client_std_fallback"])
    scale = scale[["client_id", "client_mean", "client_std"]]
    scale["client_std"] = scale["client_std"].replace(0, 1).fillna(1)
    scale["log_mean_scale"] = np.log1p(scale["client_mean"])

    long = long.merge(scale[["client_id", "log_mean_scale"]], on="client_id", how="left")
    with_lags = add_lags_rolling(long).dropna().copy()

    le = LabelEncoder()
    le.fit(long["client_id"])
    with_lags["client_ord"] = le.transform(with_lags["client_id"])
    with_lags = with_lags.merge(scale[["client_id", "client_mean", "client_std", "log_mean_scale"]], on="client_id", how="left")

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
        "rolling_std_168",
        "rolling_max_24",
        "rolling_min_24",
        "rolling_mean_same_hour_7d",
    ]
    feature_cols = [c for c in feature_cols if c in with_lags.columns]

    train_df = with_lags[with_lags["datetime"] < TRAIN_END].copy()
    val_df = with_lags[(with_lags["datetime"] >= TRAIN_END) & (with_lags["datetime"] < FINAL_START)].copy()
    final_df = with_lags[(with_lags["datetime"] >= FINAL_START) & (with_lags["datetime"] <= FINAL_END)].copy()
    return train_df, val_df, final_df, feature_cols, scale


def fit_models(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    x_train = train_df[feature_cols]
    y_train = train_df["load_kwh"]

    models: dict[str, object] = {}
    models["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=127,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_child_samples=30,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )
    models["LightGBM"].fit(x_train, y_train)

    models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=450,
        max_depth=8,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.03,
        reg_lambda=1.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="mae",
    )
    models["XGBoost"].fit(x_train, y_train, verbose=False)
    return models


def add_candidate_predictions(df: pd.DataFrame, models: dict[str, object], feature_cols: list[str]) -> pd.DataFrame:
    out = df[["datetime", "client_id", "load_kwh"]].copy()
    out["Naive_24h"] = df["lag_24"].values
    out["Naive_168h"] = df["lag_168"].values
    out["Naive_Blend"] = 0.70 * df["lag_24"].values + 0.30 * df["lag_168"].values
    for name, model in models.items():
        out[name] = np.clip(model.predict(df[feature_cols]), 0, None)
    out["LGB_XGB_Blend"] = 0.55 * out["LightGBM"] + 0.45 * out["XGBoost"]
    return out


def choose_client_models(val_pred: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows = []
    for client_id, g in val_pred.groupby("client_id"):
        y = g["load_kwh"].to_numpy(float)
        for model in candidate_cols:
            p = g[model].to_numpy(float)
            rows.append({"client_id": client_id, "model": model, "val_mae": mean_absolute_error(y, p)})
    metrics = pd.DataFrame(rows)
    selected = metrics.sort_values(["client_id", "val_mae", "model"]).groupby("client_id", as_index=False).first()
    return selected


def apply_routing(final_pred: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    routed = final_pred.merge(selected[["client_id", "model", "val_mae"]], on="client_id", how="left")
    routed["model"] = routed["model"].fillna("Naive_Blend")
    routed["Routed_Best"] = routed.apply(lambda r: r[r["model"]], axis=1)
    return routed


def chicken_predictions_for_final_day() -> pd.DataFrame:
    import lightgbm as native_lgb

    cluster_mapping = pd.read_csv(CHICKEN_ROOT / "artifacts" / "cluster_mapping.csv")
    clip_bounds = joblib.load(CHICKEN_ROOT / "artifacts" / "clip_bounds.pkl")
    pt_holidays = joblib.load(CHICKEN_ROOT / "artifacts" / "pt_holidays.pkl")
    long_data = pd.read_parquet(CHICKEN_ROOT / "artifacts" / "hourly_clean.parquet")
    long_data["datetime"] = pd.to_datetime(long_data["datetime"])

    models = {}
    for path in (CHICKEN_ROOT / "artifacts").glob("model_cluster_*.txt"):
        cid = int(path.stem.replace("model_cluster_", ""))
        models[cid] = native_lgb.Booster(model_file=str(path))

    feature_cols = ["hour", "dayofweek", "month", "is_holiday", "lag_24", "lag_168", "temperature", "humidity"]
    rows = []
    for client_id in sorted(cluster_mapping["client_id"].unique()):
        mapping = cluster_mapping[cluster_mapping["client_id"] == client_id]
        cluster = int(mapping["cluster"].iloc[0])
        if cluster not in models:
            continue

        df = long_data[long_data["client_id"] == client_id].copy()
        bounds = clip_bounds[clip_bounds["client_id"] == client_id]
        if not bounds.empty:
            df["load_kwh"] = df["load_kwh"].clip(bounds["lo"].iloc[0], bounds["hi"].iloc[0])
        df = df.sort_values("datetime")
        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month
        df["is_holiday"] = df["datetime"].dt.normalize().isin(pd.to_datetime(list(pt_holidays.keys()))).astype(int)
        df["lag_24"] = df["load_kwh"].shift(24)
        df["lag_168"] = df["load_kwh"].shift(168)
        df = df.dropna(subset=["lag_24", "lag_168"])
        day = df[(df["datetime"] >= FINAL_START) & (df["datetime"] <= FINAL_END)].copy()
        if day.empty:
            continue
        day["Chicken_Dinner"] = np.clip(models[cluster].predict(day[feature_cols].values), 0, None)
        rows.append(day[["datetime", "client_id", "Chicken_Dinner"]])
    return pd.concat(rows, ignore_index=True)


def summarize(name: str, y: pd.Series, p: pd.Series) -> dict[str, float | str]:
    y_arr = y.to_numpy(float)
    p_arr = p.to_numpy(float)
    valid = np.isfinite(y_arr) & np.isfinite(p_arr)
    y_arr = y_arr[valid]
    p_arr = p_arr[valid]
    mask = y_arr > 0.01
    mape = np.mean(np.abs((y_arr[mask] - p_arr[mask]) / y_arr[mask])) * 100 if mask.sum() else np.nan
    return {
        "model": name,
        "MAE": mean_absolute_error(y_arr, p_arr),
        "RMSE": np.sqrt(mean_squared_error(y_arr, p_arr)),
        "MAPE": mape,
        "Forecast_Accuracy_pct": max(0.0, 100.0 - mape),
        "n_rows": int(valid.sum()),
    }


def main() -> None:
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading data and building 2011-2013 train / 2014 validation / 2014-12-31 final frames...")
    long = load_long_data()
    train_df, val_df, final_df, feature_cols, _ = make_model_frames(long)
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,} | Final rows: {len(final_df):,}")
    print(f"Final clients: {final_df['client_id'].nunique():,} | Final hours: {final_df['datetime'].nunique():,}")

    print("Training final 2011-2013 models...")
    models = fit_models(train_df, feature_cols)
    for name, model in models.items():
        joblib.dump(model, OUTPUT_DIR / f"final_2014_{name.lower()}.pkl")

    print("Scoring 2014 validation period for client-level routing...")
    val_pred = add_candidate_predictions(val_df, models, feature_cols)
    candidate_cols = ["Naive_24h", "Naive_168h", "Naive_Blend", "LightGBM", "XGBoost", "LGB_XGB_Blend"]
    selected = choose_client_models(val_pred, candidate_cols)

    print("Forecasting 2014-12-31 and comparing with chicken_dinner...")
    final_pred = add_candidate_predictions(final_df, models, feature_cols)
    final_routed = apply_routing(final_pred, selected)
    chicken = chicken_predictions_for_final_day()

    final = final_routed.merge(chicken, on=["datetime", "client_id"], how="left")
    final = final.rename(columns={"load_kwh": "actual_kwh", "model": "selected_model"})

    out_cols = [
        "datetime",
        "client_id",
        "actual_kwh",
        "Naive_24h",
        "Naive_168h",
        "Naive_Blend",
        "LightGBM",
        "XGBoost",
        "LGB_XGB_Blend",
        "selected_model",
        "Routed_Best",
        "Chicken_Dinner",
    ]
    final = final[out_cols].sort_values(["client_id", "datetime"]).reset_index(drop=True)

    model_cols = ["Naive_24h", "Naive_168h", "Naive_Blend", "LightGBM", "XGBoost", "LGB_XGB_Blend", "Routed_Best", "Chicken_Dinner"]
    summary = pd.DataFrame([summarize(c, final["actual_kwh"], final[c]) for c in model_cols if final[c].notna().any()])

    selected_counts = selected["model"].value_counts().rename_axis("model").reset_index(name="n_clients")
    per_client = []
    for client_id, g in final.groupby("client_id"):
        row = {"client_id": client_id}
        for c in model_cols:
            valid = g[["actual_kwh", c]].dropna()
            if len(valid) > 0:
                row[f"{c}_MAE"] = mean_absolute_error(valid["actual_kwh"], valid[c])
        per_client.append(row)
    per_client_df = pd.DataFrame(per_client)

    final.to_csv(OUTPUT_DIR / "final_2014_12_31_hourly_all_models.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "final_2014_12_31_model_comparison.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "final_2014_client_selected_models.csv", index=False)
    selected_counts.to_csv(OUTPUT_DIR / "final_2014_selected_model_counts.csv", index=False)
    per_client_df.to_csv(OUTPUT_DIR / "final_2014_12_31_per_client_mae.csv", index=False)

    print("\nFinal 2014-12-31 model comparison:")
    print(summary.sort_values("MAE").to_string(index=False))
    if {"Routed_Best", "Chicken_Dinner"}.issubset(set(summary["model"])):
        our_mae = summary.loc[summary["model"].eq("Routed_Best"), "MAE"].iloc[0]
        chicken_mae = summary.loc[summary["model"].eq("Chicken_Dinner"), "MAE"].iloc[0]
        delta = chicken_mae - our_mae
        print(f"\nRouted_Best vs Chicken_Dinner MAE delta: {delta:.4f} ({delta / chicken_mae * 100:.2f}% better if positive)")
    print("\nSelected model counts from 2014 validation routing:")
    print(selected_counts.to_string(index=False))
    print("\nSaved final CSVs to outputs/")


if __name__ == "__main__":
    main()
