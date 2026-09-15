from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.analytics.earnings_search import EarningsChunkHit, search_earnings
from src.data.models import EarningsChunk, EarningsTranscript


def _chunk_and_transcript(
    ticker="AAPL", transcript_id=1, fiscal_year=2024, fiscal_quarter="Q2",
    chunk_type="qna", chunk_text="some text", ingested_at=None,
):
    transcript = EarningsTranscript(
        id=transcript_id, ticker=ticker, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        raw_text="irrelevant", ingested_at=ingested_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    chunk = EarningsChunk(
        transcript_id=transcript_id, chunk_type=chunk_type, chunk_text=chunk_text, embedding=[0.0] * 1536,
    )
    return chunk, transcript


def test_search_earnings_embeds_the_query_as_a_query_type(monkeypatch):
    mock_embed_text = MagicMock(return_value=[0.1] * 1536)
    monkeypatch.setattr("src.analytics.earnings_search.embed_text", mock_embed_text)
    db = MagicMock()
    db.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []

    search_earnings(db, "AI capex spending")

    mock_embed_text.assert_called_once_with("AI capex spending", input_type="query")


def test_search_earnings_maps_rows_into_hits(monkeypatch):
    monkeypatch.setattr("src.analytics.earnings_search.embed_text", MagicMock(return_value=[0.1] * 1536))
    db = MagicMock()
    chunk, transcript = _chunk_and_transcript()
    db.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = [
        (chunk, transcript)
    ]

    hits = search_earnings(db, "query")

    assert hits == [
        EarningsChunkHit(
            ticker="AAPL", transcript_id=1, fiscal_year=2024, fiscal_quarter="Q2",
            chunk_type="qna", chunk_text="some text", as_of=transcript.ingested_at,
        )
    ]


def test_search_earnings_respects_top_k(monkeypatch):
    monkeypatch.setattr("src.analytics.earnings_search.embed_text", MagicMock(return_value=[0.1] * 1536))
    db = MagicMock()
    db.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []

    search_earnings(db, "query", top_k=3)

    db.query.return_value.join.return_value.order_by.return_value.limit.assert_called_once_with(3)


def test_search_earnings_with_no_hits_returns_empty_list(monkeypatch):
    monkeypatch.setattr("src.analytics.earnings_search.embed_text", MagicMock(return_value=[0.1] * 1536))
    db = MagicMock()
    db.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []

    assert search_earnings(db, "query") == []
