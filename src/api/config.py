import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """API Gateway config. Every value is sourced from the environment
    (with a local-dev default) - nothing here is hardcoded for deploys."""

    host: str = os.environ.get("API_HOST", "0.0.0.0") # The host address for the API server, defaulting to "
    port: int = int(os.environ.get("API_PORT", "8000")) # The port number for the API server, defaulting to 8000
    environment: str = os.environ.get("API_ENV", "development") # The environment in which the API is running, defaulting to "development"
    log_level: str = os.environ.get("API_LOG_LEVEL", "info") # The log level for the API server, defaulting to "info"


settings = Settings() # Create an instance of the Settings dataclass-holds API configuration values from the environment variables
