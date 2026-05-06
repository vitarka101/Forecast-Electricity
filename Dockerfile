FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY current_cluster_forecasting.csv ./
COPY daily_price_forecast.csv ./
COPY Outputs_temp ./Outputs_temp

# Generate all artifacts at build time (no pre-existing artifacts needed)
RUN mkdir -p artifacts && \
    python scripts/build_stub_artifacts.py && \
    python scripts/generate_synthetic_forecasts.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
