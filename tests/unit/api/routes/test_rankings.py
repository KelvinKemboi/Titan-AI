import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import redis
from fastapi.testclient import TestClient

from src.api.deps import get_db
from src.api.main import app
from src.data import cache
from src.data.models import FactorScore


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _factor_score(ticker="AAPL", scan_run_id=42):
    return FactorScore(
        ticker=ticker, scan_run_id=scan_run_id, value_score=75.0, momentum_score=100.0,
        quality_score=100.0, solvency_score=80.0, volatility_score=90.0, composite_score=90.75,
        rating="STRONG BUY", raw_metrics={}, computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# a cache hit must skip the Postgres round-trip entirely
def test_cache_hit_avoids_the_db_round_trip(monkeypatch, client, db):
    cached_payload = json.dumps([
        {
            "ticker": "AAPL", "scan_run_id": 42, "computed_at": "2026-01-01T00:00:00Z",
            "composite_score": 90.75, "value_score": 75.0, "momentum_score": 100.0,
            "quality_score": 100.0, "solvency_score": 80.0, "volatility_score": 90.0,
            "rating": "STRONG BUY",
        }
    ])
    monkeypatch.setattr("src.api.routes.rankings.cache_get", MagicMock(return_value=cached_payload))
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.api.routes.rankings.cache_set", mock_cache_set)

    resp = client.get("/rankings")

    assert resp.status_code == 200
    assert resp.json()[0]["ticker"] == "AAPL"
    db.query.assert_not_called()
    mock_cache_set.assert_not_called()  # already cached - no need to re-write it


# a cache miss falls through to Postgres, and the result is written back to the cache
def test_cache_miss_falls_through_to_db_and_populates_the_cache(monkeypatch, client, db):
    monkeypatch.setattr("src.api.routes.rankings.cache_get", MagicMock(return_value=None))
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.api.routes.rankings.cache_set", mock_cache_set)
    db.query.return_value.scalar.return_value = 42
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [_factor_score()]

    resp = client.get("/rankings")

    assert resp.status_code == 200
    assert resp.json()[0]["ticker"] == "AAPL"
    mock_cache_set.assert_called_once()
    cache_key, cached_value = mock_cache_set.call_args.args
    assert cache_key == "rankings:v1:composite"
    assert json.loads(cached_value)[0]["ticker"] == "AAPL"


# the `factor` query param is part of the cache key - sorting by momentum must not
# reuse the composite-sort cache entry
def test_different_factors_use_different_cache_keys(monkeypatch, client, db):
    mock_cache_get = MagicMock(return_value=None)
    monkeypatch.setattr("src.api.routes.rankings.cache_get", mock_cache_get)
    monkeypatch.setattr("src.api.routes.rankings.cache_set", MagicMock())
    db.query.return_value.scalar.return_value = 42
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

    client.get("/rankings?factor=momentum")

    mock_cache_get.assert_called_once_with("rankings:v1:momentum")


# a Redis outage exercises the real cache module
def test_redis_unavailable_degrades_to_a_db_read(monkeypatch, client, db):
    broken_client = MagicMock()
    broken_client.get.side_effect = redis.exceptions.ConnectionError("refused")
    broken_client.setex.side_effect = redis.exceptions.ConnectionError("refused")
    monkeypatch.setattr(cache, "_get_client", lambda: broken_client)
    db.query.return_value.scalar.return_value = 42
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [_factor_score()]

    resp = client.get("/rankings")

    assert resp.status_code == 200
    assert resp.json()[0]["ticker"] == "AAPL"


def test_unknown_factor_is_a_400_before_touching_the_cache(monkeypatch, client, db):
    mock_cache_get = MagicMock()
    monkeypatch.setattr("src.api.routes.rankings.cache_get", mock_cache_get)

    resp = client.get("/rankings?factor=not_a_real_factor")

    assert resp.status_code == 400
    mock_cache_get.assert_not_called()
