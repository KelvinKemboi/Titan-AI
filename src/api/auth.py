"""
MVP API-key auth: every /chat request must present a valid key via
`Authorization: Bearer <key>`, resolved to a user_id that scopes
chat_sessions/chat_messages - replacing ChatSession.user_id's previous
unpopulated, effectively-anonymous state
"""
import os
from typing import Dict, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# auto_error=False - a *missing* Authorization header reaches get_current_user as credentials=None
_security = HTTPBearer(auto_error=False)


def _load_api_keys() -> Dict[str, str]:
    """
    Parses API_KEYS ("user_id:key,user_id:key,...") into {key: user_id}.
    Unset/empty/malformed entries are skipped 
    """
    raw = os.environ.get("API_KEYS", "")
    keys = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        user_id, _, key = pair.partition(":")
        user_id, key = user_id.strip(), key.strip()
        if user_id and key:
            keys[key] = user_id
    return keys


# Loaded once at import time, matching src/api/config.py's Settings
_API_KEYS = _load_api_keys()

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),) -> str:
    """
    FastAPI dependency: resolves the bearer token to a user_id, or raises
    401 for a missing header entirely, or one that doesn't match any
    configured key.
    """
    user_id = _API_KEYS.get(credentials.credentials) if credentials else None
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return user_id
