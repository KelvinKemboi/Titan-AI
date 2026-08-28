from unittest.mock import MagicMock

import redis

from src.data import cache


def _mock_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(cache, "_get_client", lambda: client)
    return client


def test_cache_get_returns_the_stored_value_on_a_hit(monkeypatch):
    client = _mock_client(monkeypatch)
    client.get.return_value = '{"ticker": "AAPL"}'

    assert cache.cache_get("some:key") == '{"ticker": "AAPL"}'
    client.get.assert_called_once_with("some:key")


def test_cache_get_returns_none_on_a_miss(monkeypatch):
    client = _mock_client(monkeypatch)
    client.get.return_value = None

    assert cache.cache_get("some:key") is None


# a Redis outage must degrade to "cache miss", not raise - callers never
# need their own try/except around this
def test_cache_get_returns_none_on_redis_error(monkeypatch):
    client = _mock_client(monkeypatch)
    client.get.side_effect = redis.exceptions.ConnectionError("refused")

    assert cache.cache_get("some:key") is None


def test_cache_set_writes_with_a_ttl(monkeypatch):
    client = _mock_client(monkeypatch)

    cache.cache_set("some:key", "value", ttl=60)

    client.setex.assert_called_once_with("some:key", 60, "value")


def test_cache_set_swallows_redis_errors(monkeypatch):
    client = _mock_client(monkeypatch)
    client.setex.side_effect = redis.exceptions.TimeoutError("timed out")

    cache.cache_set("some:key", "value")  # must not raise


def test_invalidate_deletes_all_given_keys(monkeypatch):
    client = _mock_client(monkeypatch)

    cache.invalidate(["a", "b"])

    client.delete.assert_called_once_with("a", "b")


def test_invalidate_with_no_keys_does_not_call_redis_at_all(monkeypatch):
    client = _mock_client(monkeypatch)

    cache.invalidate([])

    client.delete.assert_not_called()


def test_invalidate_swallows_redis_errors(monkeypatch):
    client = _mock_client(monkeypatch)
    client.delete.side_effect = redis.exceptions.ConnectionError("refused")

    cache.invalidate(["a"]) # must not raise


def test_rankings_cache_key_defaults_to_composite_for_no_factor():
    assert cache.rankings_cache_key(None) == "rankings:v1:composite"


def test_rankings_cache_key_normalizes_case_and_whitespace():
    assert cache.rankings_cache_key(" Momentum ") == "rankings:v1:momentum"


def test_company_cache_key_normalizes_case_and_whitespace():
    assert cache.company_cache_key(" aapl ") == "company:v1:AAPL"


# a scan invalidates every rankings sort order and one company key per ticker the scan actually updated
def test_invalidate_scan_caches_covers_rankings_and_scanned_tickers(monkeypatch):
    mock_invalidate = MagicMock()
    monkeypatch.setattr(cache, "invalidate", mock_invalidate)

    cache.invalidate_scan_caches(["AAPL", "MSFT"])

    (called_keys,), _ = mock_invalidate.call_args
    called_keys = list(called_keys)
    for sort_key in cache.RANKINGS_SORT_KEYS:
        assert f"rankings:v1:{sort_key}" in called_keys
    assert "company:v1:AAPL" in called_keys
    assert "company:v1:MSFT" in called_keys


def test_invalidate_scan_caches_with_no_tickers_still_invalidates_rankings(monkeypatch):
    mock_invalidate = MagicMock()
    monkeypatch.setattr(cache, "invalidate", mock_invalidate)

    cache.invalidate_scan_caches([])

    (called_keys,), _ = mock_invalidate.call_args
    assert len(list(called_keys)) == len(cache.RANKINGS_SORT_KEYS)


# chat_cache_key
def test_chat_cache_key_is_stable_for_the_same_question_and_version():
    assert cache.chat_cache_key("Explain AAPL's score", 42) == cache.chat_cache_key("Explain AAPL's score", 42)


def test_chat_cache_key_normalizes_case_and_whitespace():
    assert cache.chat_cache_key("  Explain AAPL's Score  ", 42) == cache.chat_cache_key("explain aapl's score", 42)


def test_chat_cache_key_differs_for_different_questions():
    assert cache.chat_cache_key("Explain AAPL's score", 42) != cache.chat_cache_key("Explain MSFT's score", 42)


# the data version (latest scan_run_id) must be part of the key - otherwise a new
# scan would never actually invalidate a previously-cached chat answer
def test_chat_cache_key_differs_across_data_versions():
    assert cache.chat_cache_key("Explain AAPL's score", 42) != cache.chat_cache_key("Explain AAPL's score", 43)


def test_chat_cache_key_handles_a_missing_data_version_without_crashing():
    key = cache.chat_cache_key("Explain AAPL's score", None)
    assert key.startswith("chat:v1:none:")
