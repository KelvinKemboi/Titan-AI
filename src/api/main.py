"""
Titan API Gateway. Run locally:
    uvicorn src.api.main:app --reload or python -m src.api.main
"""
from fastapi import FastAPI

from src.api.config import settings
from src.api.routes import chat, company, compare, earnings, health, observability, rankings
import uvicorn

app = FastAPI(title="Titan API Gateway")

app.include_router(health.router)
app.include_router(rankings.router)
app.include_router(company.router)
app.include_router(compare.router)
app.include_router(chat.router)
app.include_router(earnings.router)
app.include_router(observability.router)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=settings.host, port=settings.port, reload=True)
