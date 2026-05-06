# BroCode Electricity Load Forecasting

Columbia University | Team BroCode | May 2026

An end-to-end electricity load forecasting system for 370 Portuguese electricity clients from the UCI Electricity Load Diagrams 2011-2014 dataset. The project combines classical time-series models, deep learning, client clustering, model routing, OMIE price forecasting, and a deployed FastAPI dashboard with natural-language exploration.

## Live Demo

- Live app: https://brocode-electricity-forecasting-648786197436.us-central1.run.app
- Team comparison page: https://brocode-electricity-forecasting-648786197436.us-central1.run.app/improvements

## Project Summary

Electricity demand is highly heterogeneous across clients. A small residential client, a medium commercial client, and a large industrial client can have very different load scales, volatility, seasonality, and weather sensitivity. Because of this, the project does not recommend a single global model for every customer.

The final strategy is a client-aware, cluster-based model-routing system:

- Forecast hourly electricity load for each of 370 clients.
- Enrich raw load data with weather, holidays, calendar features, lag features, rolling statistics, and OMIE market prices.
- Train and compare Prophet, SARIMA/SARIMAX, LSTM, and DeepAR.
- Group clients by load scale, volatility, and model-performance behavior.
- Route each client to the best-performing model based on validation/test error.
- Serve precomputed forecasts through a deployed dashboard and API.

## Business Value

Accurate electricity forecasting helps utilities, energy traders, and grid operators:

- balance supply and demand more reliably;
- reduce over-procurement and under-procurement risk;
- identify high-error clients that need custom treatment;
- compare model behavior across customer segments;
- combine demand forecasts with OMIE prices for future EUR impact analysis.

The dashboard turns the modeling work into an operational interface where non-technical users can inspect client-level forecasts, compare BroCode results with the previous team baseline, and view daily electricity price forecasts.

## Dataset

Base dataset:

- UCI Electricity Load Diagrams 2011-2014
- 370 Portuguese electricity clients
- Raw quarter-hourly readings from 2011-01-01 through 2014-12-31
- 140,256 raw timestamp rows across 370 client columns
- 10,484,118 processed long-format client-time rows after reshaping and cleaning

Important local data note:

- `LD2011_2014.txt` is intentionally ignored by Git because it is about 678 MB and exceeds GitHub's 100 MB file limit.
- Keep the file locally in the project root when rebuilding the full notebook pipeline.
- The deployed app uses generated artifacts and does not require pushing the raw dataset to GitHub.

## External Data

The project enriches electricity demand with external signals:

| Source | Features | Purpose |
| --- | --- | --- |
| Open-Meteo archive API | temperature, relative humidity | capture weather-sensitive demand |
| OMIE electricity market data | hourly/daily electricity prices in EUR/kWh | support price-aware planning |
| Portuguese holiday calendar | `is_holiday`, days-to-holiday style features | capture abnormal demand days |
| Calendar decomposition | hour, day of week, month, season, weekend | capture recurring temporal structure |
| Lag and rolling windows | 24h, 48h, 168h, rolling mean/std/min/max | capture autoregressive load behavior |

## Preprocessing Pipeline

The technical pipeline follows a strict leakage-aware workflow:

1. Load `LD2011_2014.txt`.
2. Parse timestamps and clean numeric values.
3. Convert the wide client matrix into long client-time format.
4. Aggregate quarter-hourly readings into hourly load.
5. Remove inactive zero-padding periods before each client's first non-zero reading.
6. Clip outliers using training-period thresholds.
7. Merge Open-Meteo weather features.
8. Merge OMIE price data and fill missing values.
9. Add Portuguese holiday and calendar features.
10. Add lag and rolling statistics with shifted windows.
11. Apply per-client normalization using training data only.
12. Split chronologically into train, validation, and test sets.

The chronological split prevents future information from leaking into model training:

| Split | Date Range | Rows |
| --- | --- | ---: |
| Train | 2011-01-01 to 2014-10-30 | 9,933,188 |
| Validation | 2014-10-30 to 2014-11-30 | 275,280 |
| Test | 2014-11-30 to 2014-12-31 | 275,650 |

Per-client normalization is essential because the largest and smallest clients differ by orders of magnitude in average load. Client-level statistics are computed on the training period only and then applied consistently to validation/test data.

## Models

Four core forecasting models were benchmarked across all 370 clients:

| Model | Type | Notes |
| --- | --- | --- |
| Prophet | decomposable statistical model | strong for stable trend/seasonality patterns |
| SARIMA/SARIMAX | seasonal statistical time-series model | strongest average single-model result in the technical documentation |
| LSTM | recurrent neural network | trained with sliding windows over normalized load |
| DeepAR | probabilistic deep-learning model | provides sample-based quantile forecasts such as Q10/Q50/Q90 |

The notebook also prepared a Temporal Fusion Transformer workflow with PyTorch Forecasting, but the full comparable TFT training run was not completed because of compute constraints. The deployed comparison layer also includes generated routed artifacts used by the dashboard for fast client-level lookup.

## Evaluation Metrics

The project evaluates forecasts with:

- RMSE: penalizes large errors.
- MAE: average absolute error in kWh.
- MAPE/wMAPE: percentage-style error used for model comparison and routing.

Lower is better for all reported error metrics.

## Model Results

The formatted technical documentation reports the following average model performance across 370 clients:

| Rank | Model | RMSE | MAE | MAPE |
| ---: | --- | ---: | ---: | ---: |
| 1 | SARIMA | 50.01 | 40.58 | 14.18% |
| 2 | Prophet | 88.00 | 76.99 | 16.19% |
| 3 | DeepAR | 96.95 | 76.50 | 22.27% |
| 4 | LSTM | 101.42 | 82.45 | 23.34% |

The main finding is that no single model is best for every client. The routed ensemble substantially improves aggregate performance by assigning each client to the model that works best for its load profile.

| Strategy | Average Error | Relative Improvement |
| --- | ---: | ---: |
| Single-model baseline | 18.99% | - |
| Cluster-based routed ensemble | 9.10% | 52.1% lower error |

The final presentation and comparison analysis also highlight a weighted cluster-level comparison against the previous team baseline, where BroCode's routed approach reduced weighted average MAPE from roughly 26.7% to roughly 7.5% across all 370 clients.

## Clustering and Model Routing

Client clustering is used to make model selection practical and explainable. Clients are grouped using load-profile features and model-performance features such as:

- log-transformed client mean load;
- log-transformed client load standard deviation;
- model-level error features;
- seasonality and volatility behavior.

The resulting routing table maps each client to:

- client ID;
- cluster ID;
- cluster label/profile;
- selected best model;
- best validation/test MAPE;
- average load statistics.

This table allows the deployed app to serve client forecasts from cached artifacts rather than running expensive inference at request time.

## Comparison With Previous Team

The previous team, `chicken_dinner`, used a cluster-based LightGBM workflow and an agentic AI layer. Their approach was computationally efficient and achieved strong system-level results, but BroCode extended the scope with richer features, more model families, probabilistic forecasting, and a deployed comparison dashboard.

| Dimension | Previous Team: chicken_dinner | Team BroCode |
| --- | --- | --- |
| Primary models | Linear Regression, LightGBM | Prophet, SARIMA, LSTM, DeepAR |
| Strategy | single model per cluster | client/cluster-aware model routing |
| External features | limited weather/calendar features | weather, OMIE price, holidays, lags, rolling stats |
| Uncertainty | point forecasts only | DeepAR quantile forecasts |
| Per-client baseline | about 26.7% MAPE | about 7.5%-9.1% routed error depending on comparison framing |
| Dashboard | previous FastAPI/n8n/Gemini workflow | deployed FastAPI UI with chat, dashboard, and team comparison |
| Price forecast | not included | OMIE daily price forecast with Prophet |

BroCode's main contribution is not just another model; it is a full model-selection and deployment workflow that makes client-level forecasting inspectable.

## OMIE Price Forecasting

The project includes a separate Prophet model for daily OMIE electricity prices:

- hourly OMIE data is aggregated to daily average price;
- Prophet forecasts future daily EUR/kWh prices;
- the deployed dashboard shows the next 7 days;
- uncertainty bands provide planning context.

This price forecast is a foundation for future financial-impact analysis, where forecast error reductions can be translated into approximate EUR procurement savings.

## Deployed Application

The app is a FastAPI service with a browser UI.

Main user-facing views:

- Chat: natural-language queries for clients, clusters, comparisons, and system summaries.
- Dashboard: select any client and view next-day hourly forecasts.
- Compare Teams: BroCode forecast vs `chicken_dinner` baseline on the same chart.
- Price Forecast: 7-day OMIE electricity price forecast.
- API docs: interactive OpenAPI documentation.

Example chat queries:

- `MT_194 forecast next day`
- `MT_001 next day`
- `Compare MT_001 vs MT_199`
- `Show cluster 0`
- `Show price forecast`
- `How did we improve over chicken_dinner?`

## API Endpoints

Local examples:

```bash
curl http://localhost:8000/api/v1/health
curl "http://localhost:8000/api/v1/clients?source=brocode&limit=370"
curl http://localhost:8000/api/v1/clients/MT_194/dual-forecast
curl http://localhost:8000/api/v1/price-forecast
curl http://localhost:8000/api/v1/clusters/0
curl http://localhost:8000/api/v1/improvements
curl "http://localhost:8000/api/v1/improvements/chart?client_id=MT_194"
```

Live API examples:

```bash
curl https://brocode-electricity-forecasting-648786197436.us-central1.run.app/api/v1/health
curl https://brocode-electricity-forecasting-648786197436.us-central1.run.app/api/v1/clients/MT_194/dual-forecast
curl https://brocode-electricity-forecasting-648786197436.us-central1.run.app/api/v1/price-forecast
```

Agent query endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare MT_001 vs MT_370"}'
```

## Local Setup

Install dependencies:

```bash
uv sync
```

Without `uv`:

```bash
python3 -m pip install -r requirements.txt
```

Generate app artifacts:

```bash
python3 scripts/build_stub_artifacts.py
python3 scripts/generate_synthetic_forecasts.py
```

Run the app:

```bash
uvicorn app.main:app --reload --port 8000
```

Open:

- App: http://localhost:8000

## Docker

Build and run locally:

```bash
docker compose up --build
```

The Docker image generates required runtime artifacts during build:

- `artifacts/forecast_map.json`
- `artifacts/client_profiles.csv`
- `artifacts/cluster_profiles.csv`
- `artifacts/model_comparison.json`
- `artifacts/brocode_forecasts.csv`
- `artifacts/price_forecast_extended.csv`

## Google Cloud Run Deployment

The repository includes:

- `Dockerfile`
- `cloudbuild.yaml`
- generated artifact scripts for container startup/build support

The live service is deployed at:

https://brocode-electricity-forecasting-648786197436.us-central1.run.app

## Project Layout

```text
.
├── app
│   ├── api                  FastAPI routes
│   ├── core                 application settings
│   ├── prompts              LLM/router prompt
│   ├── services             artifact stores, router, Ollama adapter
│   └── static               browser UI
├── artifacts                generated runtime artifacts
├── outputs                  final notebook/model outputs
├── Outputs_temp             dashboard/model comparison outputs
├── scripts
│   ├── build_artifacts.py
│   ├── build_stub_artifacts.py
│   ├── final_2014_day_benchmark.py
│   ├── generate_synthetic_forecasts.py
│   └── regenerate_client_routing.py
├── energy_forecasting.ipynb
├── Energy_Forecasting_Final.pptx
├── Electricity_Load_Forecasting_Technical_Documentation_Formatted.docx
├── Dockerfile
├── cloudbuild.yaml
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Source Deliverables

The README is based on the final notebook, technical documentation, and final presentation:

- `energy_forecasting.ipynb`
- `Energy_Forecasting_Final.pptx`
- `Electricity_Load_Forecasting_Technical_Documentation_Formatted.docx`

## Team

| Member | UNI | Role |
| --- | --- | --- |
| Ayush Kumar | ak5486 | ML Engineer and Pipeline Architect |
| Aditi Mittal | am6845 | Data Processing and Feature Engineering |
| Shweta Smriti Tripathi | sst2166 | Model Evaluation and Presentation |

## Future Work

- Run the full Temporal Fusion Transformer workflow on GPU.
- Add rolling monthly retraining and drift detection.
- Quantify EUR savings by combining load forecast improvements with OMIE prices.
- Add XGBoost/Optuna as a formal fifth candidate model in the documented model-selection pipeline.
- Extend calendar features with school calendars, industrial shutdowns, and extreme-weather flags.
- Schedule nightly forecast generation so dashboard queries always read fresh cached forecasts.
