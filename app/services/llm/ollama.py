from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import settings


class OllamaClient:
    def __init__(self) -> None:
        self.enabled = settings.llm_provider.lower() == "ollama" and bool(settings.llm_model)
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.llm_model

    def chat_json(self, system: str, user: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            with httpx.Client(timeout=settings.ollama_timeout_seconds) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
            content = response.json().get("message", {}).get("content", "").strip()
            return json.loads(content) if content else None
        except Exception:
            return None

    def narrate(self, facts: dict[str, Any], question: str) -> str | None:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You explain electricity forecast data using only the provided JSON facts. "
                        "Be concise, cite client IDs, clusters, forecast totals, and comparison deltas. "
                        "Do not invent metrics or causes that are not in the facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nFacts:\n{json.dumps(facts, indent=2)[:12000]}",
                },
            ],
        }
        try:
            with httpx.Client(timeout=settings.ollama_timeout_seconds) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip() or None
        except Exception:
            return None
