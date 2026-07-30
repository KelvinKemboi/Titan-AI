import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """API Gateway config. Every value is sourced from the environment
    (with a local-dev default) — nothing here is hardcoded for deploys."""

    host: str = os.environ.get("API_HOST", "0.0.0.0")
    port: int = int(os.environ.get("API_PORT", "8000"))
    environment: str = os.environ.get("API_ENV", "development")
    log_level: str = os.environ.get("API_LOG_LEVEL", "info")


settings = Settings()
