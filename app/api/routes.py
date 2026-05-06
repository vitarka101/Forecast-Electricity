from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.artifact_store import ArtifactError, get_store
from app.services.dual_forecast_store import get_dual_store


router = APIRouter()


class CompareRequest(BaseModel):
    client_ids: list[str] = Field(..., min_length=2, max_length=2)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=2)


@router.get("/health")
def health() -> dict[str, Any]:
    try:
        store = get_store()
        return {"status": "ok", "metadata": store.metadata}
    except ArtifactError as exc:
        return {"status": "missing_artifacts", "detail": str(exc)}


@router.get("/clients")
def list_clients(
    q: Optional[str] = Query(default=None, description="Optional client ID search text"),
    limit: int = Query(default=100, ge=1, le=500),
    source: Optional[str] = Query(default=None, description="Use 'brocode' for full 370-client list"),
) -> dict[str, Any]:
    try:
        if source == "brocode":
            clients = get_dual_store().list_clients()
        else:
            clients = get_store().list_clients()
        if q:
            needle = q.strip().upper()
            clients = [client for client in clients if needle in str(client.get("client_id", "")).upper()]
        return {"clients": clients[:limit], "count": len(clients)}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/clients/{client_id}/context")
def client_context(client_id: str) -> dict[str, Any]:
    try:
        return get_store().get_client_context(client_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/clients/{client_id}/forecast")
def client_forecast(client_id: str) -> dict[str, Any]:
    try:
        return get_store().get_client_forecast(client_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/clients/{client_id}/dual-forecast")
def client_dual_forecast(client_id: str) -> dict[str, Any]:
    try:
        return get_dual_store().get_dual_forecast(client_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/price-forecast")
def price_forecast() -> dict[str, Any]:
    try:
        return get_dual_store().get_price_forecast()
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/clusters/{cluster_id}")
def cluster_context(cluster_id: str) -> dict[str, Any]:
    try:
        return get_store().get_cluster_context(cluster_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/system/forecast")
def system_forecast() -> dict[str, Any]:
    try:
        return get_store().get_system_forecast()
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/compare")
def compare_clients(request: CompareRequest) -> dict[str, Any]:
    try:
        return get_store().compare_clients(request.client_ids)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/agent/query")
def agent_query(request: AgentQueryRequest) -> dict[str, Any]:
    try:
        from app.services.router import AgentRouterService
        store = get_store()
        return AgentRouterService(store).handle_query(request.query)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/improvements")
def improvements() -> dict[str, Any]:
    try:
        return get_store().improvements()
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/improvements/chart")
def improvements_chart(client_id: Optional[str] = None) -> dict[str, Any]:
    try:
        return get_store().improvement_chart(client_id=client_id)
    except Exception as exc:
        raise _http_error(exc) from exc


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ArtifactError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))
