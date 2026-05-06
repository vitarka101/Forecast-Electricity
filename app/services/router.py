from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.services.artifact_store import ForecastArtifactStore
from app.services.llm.ollama import OllamaClient


Intent = Literal["client", "compare", "cluster", "system", "improvements"]


@dataclass
class RouterDecision:
    intent: Intent
    client_ids: list[str]
    cluster_id: str | None = None


class AgentRouterService:
    CLIENT_RE = re.compile(r"\b(?:MT[_\s-]?)?(\d{1,3})\b", re.IGNORECASE)
    EXPLICIT_CLIENT_RE = re.compile(r"\bMT[_\s-]?(\d{1,3})\b", re.IGNORECASE)
    CLUSTER_RE = re.compile(r"\bcluster\s+(\d+)\b", re.IGNORECASE)

    def __init__(self, store: ForecastArtifactStore):
        self.store = store
        self.ollama = OllamaClient()
        self.system_prompt = self._load_prompt()

    def handle_query(self, query: str) -> dict[str, Any]:
        decision = self._decide(query)

        if decision.intent == "compare":
            payload = self.store.compare_clients(decision.client_ids)
            summary = self._narrate_or_template(query, payload, self._compare_summary(payload))
            return {"intent": "compare", "summary": summary, "payload": payload}

        if decision.intent == "cluster":
            payload = self.store.get_cluster_context(decision.cluster_id or "0")
            summary = self._narrate_or_template(query, payload, self._cluster_summary(payload))
            return {"intent": "cluster", "summary": summary, "payload": payload}

        if decision.intent == "system":
            payload = self.store.get_system_forecast()
            summary = self._narrate_or_template(query, payload, self._system_summary(payload))
            return {"intent": "system", "summary": summary, "payload": payload}

        if decision.intent == "improvements":
            payload = self.store.improvements()
            summary = self._narrate_or_template(query, payload, self._improvement_summary(payload))
            return {"intent": "improvements", "summary": summary, "payload": payload}

        if not decision.client_ids:
            raise ValueError("Please include a client ID such as MT_001 or ask for a cluster/system summary.")
        payload = self.store.get_client_context(decision.client_ids[0])
        summary = self._narrate_or_template(query, payload, self._client_summary(payload))
        return {"intent": "client", "summary": summary, "payload": payload}

    def _decide(self, query: str) -> RouterDecision:
        llm_decision = self._ollama_decision(query)
        if llm_decision:
            return llm_decision

        lowered = query.lower()
        explicit_clients = [f"MT_{int(match):03d}" for match in self.EXPLICIT_CLIENT_RE.findall(query)]
        cluster_match = self.CLUSTER_RE.search(query)

        if "improve" in lowered or "baseline" in lowered or "chicken" in lowered or "comparison page" in lowered:
            return RouterDecision(intent="improvements", client_ids=[])
        if "system" in lowered or "all clients" in lowered or "total load" in lowered:
            return RouterDecision(intent="system", client_ids=[])
        if cluster_match:
            return RouterDecision(intent="cluster", client_ids=[], cluster_id=cluster_match.group(1))
        if "compare" in lowered or " vs " in lowered or "versus" in lowered:
            clients = explicit_clients or [f"MT_{int(match):03d}" for match in self.CLIENT_RE.findall(query)]
            return RouterDecision(intent="compare", client_ids=list(dict.fromkeys(clients)))
        if explicit_clients:
            return RouterDecision(intent="client", client_ids=[explicit_clients[0]])
        return RouterDecision(intent="client", client_ids=[])

    def _ollama_decision(self, query: str) -> RouterDecision | None:
        response = self.ollama.chat_json(
            self.system_prompt,
            (
                "Return JSON with keys: intent, client_ids, cluster_id. "
                "intent must be one of client, compare, cluster, system, improvements.\n"
                f"Query: {query}"
            ),
        )
        if not response:
            return None
        try:
            intent = response.get("intent")
            if intent not in {"client", "compare", "cluster", "system", "improvements"}:
                return None
            clients = [
                ForecastArtifactStore._normalize_client_id(str(client_id))
                for client_id in response.get("client_ids", [])
            ]
            cluster_id = response.get("cluster_id")
            return RouterDecision(intent=intent, client_ids=clients, cluster_id=str(cluster_id) if cluster_id else None)
        except Exception:
            return None

    def _narrate_or_template(self, query: str, payload: dict[str, Any], fallback: str) -> str:
        return self.ollama.narrate(payload, query) or fallback

    @staticmethod
    def _client_summary(payload: dict[str, Any]) -> str:
        client = payload["client"]
        profile = client["profile"]
        daily = client["forecast_daily"]
        cluster = payload["cluster"]
        if client.get("output_status") != "ok":
            return (
                f"{client['client_id']} is mapped to cluster {client['cluster_id']} and assigned to "
                f"{client.get('assigned_model', 'unknown model')}, but no matching model-output rows were found "
                "in Outputs/ui_predictions.csv."
            )
        return (
            f"{client['client_id']} is mapped to cluster {client['cluster_id']} "
            f"({cluster['profile']['label']}) and uses {client.get('assigned_model', 'the routed model')}. "
            f"Its selected 24-row model-output total is "
            f"{daily['total_kwh']:.2f} kWh, with an average hourly load of "
            f"{profile['mean_hourly_kwh']:.2f} kWh and peak historical hour {profile['peak_hour']}."
        )

    @staticmethod
    def _compare_summary(payload: dict[str, Any]) -> str:
        comp = payload["comparison"]
        pct = comp["percent_delta_vs_right"]
        pct_text = "not available" if pct is None else f"{pct:+.2f}%"
        return (
            f"{comp['left_client_id']} is forecast at {comp['left_daily_forecast_kwh']:.2f} kWh for the next day, "
            f"while {comp['right_client_id']} is forecast at {comp['right_daily_forecast_kwh']:.2f} kWh. "
            f"The delta is {comp['absolute_delta_kwh']:+.2f} kWh ({pct_text} vs the second client)."
        )

    @staticmethod
    def _cluster_summary(payload: dict[str, Any]) -> str:
        profile = payload["profile"]
        daily = payload["forecast_daily"]
        return (
            f"Cluster {payload['cluster_id']} contains {profile['client_count']} clients and is labeled "
            f"{profile['label']}. It has {profile.get('predicted_client_count', 0)} clients with available "
            f"model-output rows, and their selected 24-row aggregate is {daily['total_kwh']:.2f} kWh."
        )

    @staticmethod
    def _system_summary(payload: dict[str, Any]) -> str:
        daily = payload.get("forecast_daily", {})
        return f"The system-level next-day forecast is {daily.get('total_kwh', 0):.2f} kWh across all mapped clients."

    @staticmethod
    def _improvement_summary(payload: dict[str, Any]) -> str:
        winner = payload.get("new_project", {}).get("winner", {})
        return (
            f"The new project now routes each client through routing_table.csv and reads that client's assigned "
            f"model output from ui_predictions.csv. The top global metric model is "
            f"{winner.get('model_name', 'unknown')}."
        )

    @staticmethod
    def _load_prompt() -> str:
        path = Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
        return path.read_text(encoding="utf-8") if path.exists() else "Route electricity forecast questions."
