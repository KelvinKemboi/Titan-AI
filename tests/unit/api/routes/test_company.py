import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import redis
from fastapi.testclient import TestClient

from src.api.deps import get_db
from src.api.main import app
from src.data import cache
from src.data.models import Company, FactorScore


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _company(ticker="AAPL"):
    return Company(ticker=ticker, name="Apple Inc.", sector="Technology", industry="Consumer Electronics")


def _factor_score(ticker="AAPL", scan_run_id=42):
    return FactorScore(
        ticker=ticker, scan_run_id=scan_run_id, value_score=75.0, momentum_score=100.0,
        quality_score=100.0, solvency_score=80.0, volatility_score=90.0, composite_score=90.75,
        rating="STRONG BUY", raw_metrics={}, computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _cached_detail_json(ticker="AAPL"):
    return json.dumps({
        "ticker": ticker, "name": "Apple Inc.", "sector": "Technology",
        "industry": "Consumer Electronics", "description": None, "scan_run_id": 42,
        "computed_at": "2026-01-01T00:00:00Z", "composite_score": 90.75, "value_score": 75.0,
        "momentum_score": 100.0, "quality_score": 100.0, "solvency_score": 80.0,
        "volatility_score": 90.0, "rating": "STRONG BUY", "raw_metrics": {},
    })


# a cache hit must skip the Postgres round-trip entirely
def test_cache_hit_avoids_the_db_round_trip(monkeypatch, client, db):
    monkeypatch.setattr(
        "src.api.routes.company.cache_get", MagicMock(return_value=_cached_detail_json())
    )
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.api.routes.company.cache_set", mock_cache_set)

    resp = client.get("/company/AAPL")

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"
    db.get.assert_not_called()
    mock_cache_set.assert_not_called()


# a cache miss falls through to Postgres, and the result is written back to the cache
def test_cache_miss_falls_through_to_db_and_populates_the_cache(monkeypatch, client, db):
    monkeypatch.setattr("src.api.routes.company.cache_get", MagicMock(return_value=None))
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.api.routes.company.cache_set", mock_cache_set)
    db.get.return_value = _company()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = _factor_score()

    resp = client.get("/company/AAPL")

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"
    mock_cache_set.assert_called_once()
    cache_key, cached_value = mock_cache_set.call_args.args
    assert cache_key == "company:v1:AAPL"
    assert json.loads(cached_value)["ticker"] == "AAPL"


# ticker normalization (lowercase, whitespace) must resolve to the same cache key
def test_ticker_is_normalized_in_the_cache_key(monkeypatch, client, db):
    mock_cache_get = MagicMock(return_value=None)
    monkeypatch.setattr("src.api.routes.company.cache_get", mock_cache_get)
    monkeypatch.setattr("src.api.routes.company.cache_set", MagicMock())
    db.get.return_value = _company()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    client.get("/company/aapl")

    mock_cache_get.assert_called_once_with("company:v1:AAPL")


# an unknown ticker is never cached - it might be a real ticker after a future scan
def test_unknown_ticker_404_is_not_cached(monkeypatch, client, db):
    monkeypatch.setattr("src.api.routes.company.cache_get", MagicMock(return_value=None))
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.api.routes.company.cache_set", mock_cache_set)
    db.get.return_value = None

    resp = client.get("/company/ZZZZNOTREAL")

    assert resp.status_code == 404
    mock_cache_set.assert_not_called()


# a Redis outage (not just a plain cache miss) must degrade to a correct DB-backed
# response, not a 500 - exercises the real cache module, not a mocked cache_get
def test_redis_unavailable_degrades_to_a_db_read(monkeypatch, client, db):
    broken_client = MagicMock()
    broken_client.get.side_effect = redis.exceptions.ConnectionError("refused")
    broken_client.setex.side_effect = redis.exceptions.ConnectionError("refused")
    monkeypatch.setattr(cache, "_get_client", lambda: broken_client)
    db.get.return_value = _company()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = _factor_score()

    resp = client.get("/company/AAPL")

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"
