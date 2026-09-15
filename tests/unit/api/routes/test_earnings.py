from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_db
from src.api.main import app
from src.data.models import Company, EarningsInsight, EarningsTranscript


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _transcript(id=1, ticker="AAPL", fiscal_year=2024, fiscal_quarter="Q2", ingested_at=None):
    return EarningsTranscript(
        id=id, ticker=ticker, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        raw_text="irrelevant", source_url="https://api.api-ninjas.com/v1/earningstranscript?ticker=AAPL",
        ingested_at=ingested_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _insight(transcript_id=1, summary="a summary", guidance_direction="raised", guidance_quote="a quote",
             sentiment_score=0.5, risks=None):
    return EarningsInsight(
        transcript_id=transcript_id, summary=summary, guidance_direction=guidance_direction,
        guidance_quote=guidance_quote, sentiment_score=sentiment_score, risks=risks or [],
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_unknown_ticker_returns_404(db, client):
    db.get.return_value = None

    resp = client.get("/earnings/ZZZZNOTREAL")

    assert resp.status_code == 404


def test_known_ticker_with_no_transcripts_returns_an_empty_list(db, client):
    db.get.return_value = Company(ticker="AAPL")
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

    resp = client.get("/earnings/AAPL")

    assert resp.status_code == 200
    assert resp.json() == {"ticker": "AAPL", "transcripts": []}


def test_returns_a_transcript_paired_with_its_insight(db, client):
    db.get.return_value = Company(ticker="AAPL")
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [_transcript()]
    db.query.return_value.filter.return_value.one_or_none.return_value = _insight()

    resp = client.get("/earnings/AAPL")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert len(body["transcripts"]) == 1
    t = body["transcripts"][0]
    assert t["transcript_id"] == 1
    assert t["fiscal_year"] == 2024
    assert t["fiscal_quarter"] == "Q2"
    assert t["insight"]["summary"] == "a summary"
    assert t["insight"]["guidance_direction"] == "raised"
    assert t["insight"]["guidance_quote"] == "a quote"
    assert t["insight"]["sentiment_score"] == 0.5


def test_a_transcript_with_no_insight_generated_yet_has_a_null_insight(db, client):
    db.get.return_value = Company(ticker="AAPL")
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [_transcript()]
    db.query.return_value.filter.return_value.one_or_none.return_value = None

    resp = client.get("/earnings/AAPL")

    assert resp.json()["transcripts"][0]["insight"] is None


def test_ticker_is_normalized(db, client):
    db.get.return_value = None

    client.get("/earnings/aapl")

    assert db.get.call_args.args[1] == "AAPL"
