"""
Earnings transcript provider client: the one place
that knows API Ninjas is the transcript provider. Everything above this
module (a later ingestion service) works with `Transcript`/
`TranscriptSearchResult` and calls `get_transcript`/`search_transcripts`
"""
import logging
import os
from datetime import date
from typing import Dict, Iterator, Optional

import requests
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.api-ninjas.com/v1"
_API_KEY_ENV_VAR = "EARNINGS_PROVIDER_API_KEY"

# API Ninjas signals a rate-limited call with 429
_MAX_RETRIES = 5
_BACKOFF_FACTOR = 1.0
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_TIMEOUT_SECONDS = 30

# earningstranscriptsearch's own page size - a page shorter than this is the last one.
_SEARCH_PAGE_SIZE = 50

_session: Optional[requests.Session] = None


class EarningsProviderError(Exception):
    """Any provider call failure that isn't a plain "no transcript for this
    ticker/quarter" (see get_transcript's None return for that case)."""


class EarningsProviderAuthError(EarningsProviderError):
    """EARNINGS_PROVIDER_API_KEY is unset, or the provider rejected it. Not
    swallowed like a per-ticker coverage gap - a bad key fails every call in
    the run identically, so the caller should stop and fix configuration
    rather than burn through retries ticker by ticker."""


class Transcript(BaseModel):
    """
    One earnings call transcript, in Titan's own shape (mirrors
    `earnings_transcripts`).
    """

    ticker: str
    fiscal_year: int
    fiscal_quarter: str  # e.g. "Q2"
    call_date: Optional[date] = None
    raw_text: str
    source_url: Optional[str] = None


class TranscriptSearchResult(BaseModel):
    """One row from `search_transcripts`: enough to know a transcript exists
    for (ticker, fiscal_year, fiscal_quarter) and decide whether it's worth
    fetching, without paying for the full transcript text up front."""

    ticker: str
    fiscal_year: int
    fiscal_quarter: str
    call_date: Optional[date] = None


def _session_with_retries() -> requests.Session:
    """Lazily creates the shared, retry-configured session (mirrors
    src/data/cache.py's lazy singleton for the Redis client)."""
    global _session
    if _session is None:
        session = requests.Session()
        retry = Retry(
            total=_MAX_RETRIES,
            backoff_factor=_BACKOFF_FACTOR,
            status_forcelist=_RETRYABLE_STATUSES,
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        _session = session
    return _session


def _headers() -> Dict[str, str]:
    api_key = os.environ.get(_API_KEY_ENV_VAR)
    if not api_key:
        raise EarningsProviderAuthError(
            f"{_API_KEY_ENV_VAR} is not set - cannot call the earnings transcript provider"
        )
    return {"X-Api-Key": api_key}


def _get(path: str, params: dict, *, session: Optional[requests.Session] = None) -> requests.Response:
    session = session or _session_with_retries()
    try:
        response = session.get(
            f"{_BASE_URL}{path}", headers=_headers(), params=params, timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise EarningsProviderError(f"Request to {path} failed: {exc}") from exc

    # api-ninjas.com/error-codes: 400 covers both "missing API key" and "invalid API key"
    if response.status_code == 400 and "api key" in response.text.lower():
        raise EarningsProviderAuthError(f"Provider rejected {_API_KEY_ENV_VAR}: {response.text}")
    if not response.ok:
        raise EarningsProviderError(f"{path} returned {response.status_code}: {response.text}")
    return response


def _to_transcript(payload: dict) -> Transcript:
    return Transcript(
        ticker=payload["ticker"],
        fiscal_year=int(payload["year"]),
        fiscal_quarter=f"Q{payload['quarter']}",
        call_date=payload.get("date") or None,
        raw_text=payload["transcript"],
    )


def get_transcript(
    ticker: str,
    *,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> Optional[Transcript]:
    """
    Fetches one transcript for `ticker`. Omitting both `year` and `quarter`
    returns the provider's own "latest available" default.

    Returns None when the provider has nothing for this
    ticker/quarter
    """
    ticker = ticker.strip().upper()
    params: Dict[str, object] = {"ticker": ticker}
    if year is not None:
        params["year"] = year
    if quarter is not None:
        params["quarter"] = quarter

    response = _get("/earningstranscript", params, session=session)
    payload = response.json()
    if not payload or not payload.get("transcript"):
        return None
    return _to_transcript(payload)


def search_transcripts(
    ticker: str, *, session: Optional[requests.Session] = None,
) -> Iterator[TranscriptSearchResult]:
    """
    Yields every transcript the provider's `earningstranscriptsearch`
    endpoint knows about for `ticker`, most recent call first, transparently
    walking every `_SEARCH_PAGE_SIZE`-row page of results
    """
    ticker = ticker.strip().upper()
    offset = 0
    while True:
        response = _get(
            "/earningstranscriptsearch",
            {"ticker": ticker, "offset": offset, "limit": _SEARCH_PAGE_SIZE},
            session=session,
        )
        page = response.json()
        if not page:
            return
        for row in page:
            yield TranscriptSearchResult(
                ticker=row["ticker"],
                fiscal_year=int(row["year"]),
                fiscal_quarter=f"Q{row['quarter']}",
                call_date=row.get("date") or None,
            )
        if len(page) < _SEARCH_PAGE_SIZE:
            return
        offset += _SEARCH_PAGE_SIZE
