"""
Titan API Gateway. Run locally:
    uvicorn src.api.main:app --reload or python -m src.api.main
"""
from fastapi import FastAPI

from src.api.config import settings
from src.api.routes import health, rankings
import uvicorn

app = FastAPI(title="Titan API Gateway")

app.include_router(health.router)
app.include_router(rankings.router)

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=settings.host, port=settings.port, reload=True)
