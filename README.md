# Brocode · Electricity Forecasting UI

**Columbia University — Team Brocode**

A full-stack electricity forecasting app for 370 Portuguese clients (UCI Electricity Load Dataset 2011–2014). Features a **Chat** window for natural language queries and a **Dashboard** with client-level dual-team forecast comparison, 7-day electricity price forecasting, and a complete write-up of our pipeline.

## Quick Start (Recommended)

```bash
# 1. Install dependencies
uv sync
# — or without uv —
pip3 install -r requirements.txt

# 2. Generate synthetic forecast data (one-time, ~30 seconds)
python3 scripts/generate_synthetic_forecasts.py

# 3. (Optional) Build legacy artifacts for /improvements page
python3 scripts/build_artifacts.py

# 4. Run the server
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**

## Features

**Chat tab** — ask anything:
- `MT_194 forecast next day` → hourly dual-team chart (Brocode vs chicken_dinner), cluster + model shown
- `Show price forecast` → 7-day electricity price bar chart (OMIE/Prophet)
- `Compare MT_001 vs MT_199` → side-by-side comparison

**Dashboard tab**:
- Dropdown for all 370 clients
- Next-day hourly forecast chart: Brocode (blue) vs chicken_dinner (orange dashed) vs Actual (dark)
- Accuracy metrics grid
- Brocode Innovation: 7-day daily price forecast
- "What We've Implemented" — 8-card write-up of our full pipeline

## New API Endpoints

```bash
# Dual-team hourly forecast for any client
curl http://localhost:8000/api/v1/clients/MT_194/dual-forecast

# 7-day electricity price forecast
curl http://localhost:8000/api/v1/price-forecast

# All 370 Brocode clients
curl "http://localhost:8000/api/v1/clients?source=brocode&limit=370"
```

## What It Does (Legacy endpoints)

Ask questions like:

- `Tell me about MT_001`
- `Compare MT_001 vs MT_370`
- `Show cluster 2`
- `What is the system forecast?`
- `How did we improve over chicken_dinner?`

If Ollama is enabled, the app asks the local model to narrate the retrieved facts. If Ollama is unavailable, deterministic summaries are returned.

## Project Layout

```text
.
├── app
│   ├── api              FastAPI routes
│   ├── core             settings
│   ├── prompts          LLM routing prompt
│   ├── services         artifact store, router, Ollama client
│   └── static           browser UI
├── artifacts            generated forecast-map artifacts
├── scripts
│   └── build_artifacts.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Generate Artifacts

For a quick smoke test:

```bash
python3 scripts/build_artifacts.py --client-limit 10
```

For the full project:

```bash
python3 scripts/build_artifacts.py
```

Generated files:

- `artifacts/forecast_map.json`
- `artifacts/client_profiles.csv`
- `artifacts/cluster_profiles.csv`
- `artifacts/history_aggregates.parquet`
- `artifacts/model_comparison.json`

The build script validates multiple deployable forecast-map methods on a holdout period and chooses the highest-accuracy method as the winner. Daily totals are derived from the saved 24-hour forecast. Weekly values are historical context aggregates.

When `Outputs/routing_table.csv` and `Outputs/ui_predictions.csv` exist, the build script uses the routed model-output path instead:

- `Outputs/routing_table.csv`: `client_id`, `cluster`, `assigned_cluster_model`
- `Outputs/ui_predictions.csv`: `model`, `client_id`, `datetime`, `actual_kwh`, `predicted_kwh`, `split`
- `Outputs/model_metrics.csv`: model-level metrics for the improvements page

If a routed client does not have matching rows in `ui_predictions.csv`, the app marks that client as `missing_model_output` instead of fabricating a forecast.

## Run Locally

```bash
python3 -m pip install -r requirements.txt
python3 scripts/build_artifacts.py --client-limit 10
uvicorn app.main:app --reload
```

Open:

- chat UI: `http://localhost:8000`
- improvements page: `http://localhost:8000/improvements`
- API docs: `http://localhost:8000/docs`

## Docker Deployment

Create the environment file:

```bash
cp .env.example .env
```

Build artifacts before starting Docker:

```bash
python3 scripts/build_artifacts.py
```

Start the app:

```bash
docker compose up --build
```

Open `http://localhost:8000`.

## Ollama Setup

Install Ollama and pull a model:

```bash
ollama pull llama3.1:8b
```

Use this `.env` setup when Docker should call Ollama on the host:

```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

To run without Ollama:

```bash
LLM_PROVIDER=heuristic
LLM_MODEL=
```

## API Examples

```bash
curl http://localhost:8000/api/v1/clients/MT_001/context
curl http://localhost:8000/api/v1/clusters/2
curl http://localhost:8000/api/v1/improvements
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"client_ids":["MT_001","MT_370"]}'
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare MT_001 vs MT_370"}'
```

## Improvement Story

The chicken_dinner baseline used per-cluster LightGBM models and a minimal Streamlit/FastAPI interface. This version improves the final product by:

- serving a saved forecast map instead of doing slow runtime inference
- adding chat-first client, cluster, system, and comparison workflows
- mapping every client to a behavior cluster
- adding Ollama narration with deterministic fallback
- adding a separate improvements page
- documenting artifact generation and Docker deployment

The notebook results are preserved as reference evidence, while the deployed app uses the best validated forecast-map artifact generated by the offline script.
