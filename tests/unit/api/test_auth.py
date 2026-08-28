import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.api import auth


def _creds(token):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


#_load_api_keys
def test_load_api_keys_parses_user_colon_key_pairs(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:key-a,bob:key-b")

    assert auth._load_api_keys() == {"key-a": "alice", "key-b": "bob"}


def test_load_api_keys_is_empty_when_unset(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)

    assert auth._load_api_keys() == {}


def test_load_api_keys_trims_whitespace_around_pairs_and_entries(monkeypatch):
    monkeypatch.setenv("API_KEYS", " alice : key-a , bob:key-b ")

    assert auth._load_api_keys() == {"key-a": "alice", "key-b": "bob"}


# a malformed entry (missing user_id or key) is skipped
def test_load_api_keys_skips_malformed_entries(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:key-a,no-colon-here,:key-c,bob:")

    assert auth._load_api_keys() == {"key-a": "alice"}


# get_current_user
def test_get_current_user_returns_the_mapped_user_id(monkeypatch):
    monkeypatch.setattr(auth, "_API_KEYS", {"key-a": "alice"})

    assert auth.get_current_user(_creds("key-a")) == "alice"


def test_get_current_user_rejects_an_unknown_key(monkeypatch):
    monkeypatch.setattr(auth, "_API_KEYS", {"key-a": "alice"})

    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_user(_creds("wrong-key"))

    assert exc_info.value.status_code == 401


# auto_error=False on the HTTPBearer means a missing header reaches this function as credentials=None
def test_get_current_user_rejects_a_missing_credential(monkeypatch):
    monkeypatch.setattr(auth, "_API_KEYS", {"key-a": "alice"})

    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_user(None)

    assert exc_info.value.status_code == 401


def test_get_current_user_fails_closed_when_no_keys_are_configured(monkeypatch):
    monkeypatch.setattr(auth, "_API_KEYS", {})

    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_user(_creds("any-key"))

    assert exc_info.value.status_code == 401
