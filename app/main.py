from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


app = FastAPI(title=settings.app_name)
app.include_router(router, prefix=settings.api_v1_prefix)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/improvements")
def improvements_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "improvements.html")
