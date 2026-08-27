"""
Redis cache for read-heavy endpoints: /rankings and
/company/{ticker}. Invalidated on scan completion (scanner_service.py's
run_scan_and_persist calls invalidate_scan_caches once results are
committed to Postgres)
"""
import logging
import os
from typing import Iterable, List, Optional

import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

# Backstop only
_DEFAULT_TTL_SECONDS = 3600

_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    """
    Lazily creates the shared Redis client
    """
    global _client
    if _client is None:
        _client = redis.Redis.from_url(
            REDIS_URL, decode_responses=True, socket_connect_timeout=1, socket_timeout=1,
        )
    return _client


def cache_get(key: str) -> Optional[str]:
    """
    Returns the cached string for `key`, or None on a cache miss OR any
    Redis failure
    """
    try:
        return _get_client().get(key)
    except redis.RedisError:
        logger.warning("Redis GET failed for key=%s; falling back to the DB", key, exc_info=True)
        return None


def cache_set(key: str, value: str, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
    """Best-effort cache write. Failures are logged and swallowed, never raised."""
    try:
        _get_client().setex(key, ttl, value)
    except redis.RedisError:
        logger.warning("Redis SET failed for key=%s; continuing without caching", key, exc_info=True)


def invalidate(keys: Iterable[str]) -> None:
    """Best-effort cache invalidation for `keys`. Failures are logged and swallowed."""
    keys = list(keys)
    if not keys:
        return
    try:
        _get_client().delete(*keys)
    except redis.RedisError:
        logger.warning("Redis DELETE failed for keys=%s", keys, exc_info=True)


_RANKINGS_KEY_PREFIX = "rankings:v1"
_COMPANY_KEY_PREFIX = "company:v1"

# Every `factor` /rankings accepts (see rankings.py's _FACTOR_COLUMNS) and the default "composite" sort 
# kept as its own list here to avoid a cache-layer -> route-layer dependency
RANKINGS_SORT_KEYS = ["composite", "value", "momentum", "quality", "solvency", "volatility"]


def rankings_cache_key(factor: Optional[str]) -> str:
    return f"{_RANKINGS_KEY_PREFIX}:{(factor or 'composite').strip().lower()}"


def company_cache_key(ticker: str) -> str:
    return f"{_COMPANY_KEY_PREFIX}:{ticker.strip().upper()}"


def invalidate_scan_caches(tickers: List[str]) -> None:
    """
    Call once a scan's results are committed to Postgres
    (scanner_service.run_scan_and_persist): invalidates every /rankings
    sort-order key and every /company/{ticker} key for a ticker this scan updated
    """
    keys = [rankings_cache_key(f) for f in RANKINGS_SORT_KEYS]
    keys += [company_cache_key(t) for t in tickers]
    invalidate(keys)
