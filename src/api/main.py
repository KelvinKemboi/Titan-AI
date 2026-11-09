"""
Titan API Gateway. Run locally:
    uvicorn src.api.main:app --reload or python -m src.api.main
"""
from pathlib import Path

from dotenv import load_dotenv

# Must run before any src.* import: src/data/db.py (DATABASE_URL), src/api/auth.py
# (API_KEYS) and the Anthropic/Voyage clients all read the environment at import
# or first use. Without this, `uvicorn src.api.main:app` never sees .env and every
# /chat call dies with "Could not resolve authentication method".
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI

from src.api.config import settings
from src.api.routes import chat, company, compare, earnings, health, observability, rankings
from src.observability.metrics import PrometheusMiddleware, metrics_endpoint
import uvicorn

app = FastAPI(title="Titan API Gateway")
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics_endpoint, include_in_schema=False)

app.include_router(health.router)
app.include_router(rankings.router)
app.include_router(company.router)
app.include_router(compare.router)
app.include_router(chat.router)
app.include_router(earnings.router)
app.include_router(observability.router)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=settings.host, port=settings.port, reload=True)
