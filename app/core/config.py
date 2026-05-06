from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Settings:
    def __init__(self) -> None:
        _load_dotenv(BASE_DIR / ".env")
        self.app_name = os.getenv("APP_NAME", "electricity-forecast-map")
        self.api_v1_prefix = os.getenv("API_V1_PREFIX", "/api/v1")
        self.environment = os.getenv("ENVIRONMENT", "dev")

        self.artifacts_dir = _resolve_path_env("ARTIFACTS_DIR", BASE_DIR / "artifacts")
        self.forecast_map_path = _resolve_path_env("FORECAST_MAP_PATH", BASE_DIR / "artifacts" / "forecast_map.json")
        self.client_profiles_path = _resolve_path_env("CLIENT_PROFILES_PATH", BASE_DIR / "artifacts" / "client_profiles.csv")
        self.cluster_profiles_path = _resolve_path_env("CLUSTER_PROFILES_PATH", BASE_DIR / "artifacts" / "cluster_profiles.csv")
        self.history_aggregates_path = _resolve_path_env("HISTORY_AGGREGATES_PATH", BASE_DIR / "artifacts" / "history_aggregates.parquet")
        self.model_comparison_path = _resolve_path_env("MODEL_COMPARISON_PATH", BASE_DIR / "artifacts" / "model_comparison.json")

        self.llm_provider = os.getenv("LLM_PROVIDER", "heuristic")
        self.llm_model = os.getenv("LLM_MODEL") or None
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))


def _resolve_path_env(key: str, default: Path) -> str:
    value = os.getenv(key)
    if not value:
        return str(default)
    path = Path(value)
    if path.is_absolute() and len(path.parts) > 1 and path.parts[1] == "app" and BASE_DIR != Path("/app"):
        return str(BASE_DIR.joinpath(*path.parts[2:]))
    return str(path)


settings = Settings()
Path(settings.artifacts_dir).mkdir(parents=True, exist_ok=True)
