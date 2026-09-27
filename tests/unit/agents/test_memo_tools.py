from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.agents.tools.memo_tools import DISPATCH, TOOLS, search_memos
from src.analytics.memo_search import MemoHit


def _hit(ticker, scan_run_id, memo_text="a memo", as_of=None):
    return MemoHit(
        ticker=ticker,
        scan_run_id=scan_run_id,
        memo_text=memo_text,
        as_of=as_of or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# every memo hit becomes a `type="memo"` Source citing its ticker + scan_run_id
def test_search_memos_returns_sourced_results(monkeypatch):
    hits = [_hit("AAPL", 142), _hit("MSFT", 142)]
    monkeypatch.setattr("src.agents.tools.memo_tools._search_memos", MagicMock(return_value=hits))

    result = search_memos(db=MagicMock(), query="deep competitive moats")

    assert result.data == {"results": [hit.model_dump(mode="json") for hit in hits]}
    assert [s.type for s in result.sources] == ["memo", "memo"]
    assert [s.ticker for s in result.sources] == ["AAPL", "MSFT"]
    assert [s.ref_id for s in result.sources] == [142, 142]


# each source's `detail` carries the actual memo text - the primary evidence for a
# qualitative claim, so a citation UI can show it without a second fetch
def test_search_memos_source_detail_carries_the_memo_text(monkeypatch):
    hits = [_hit("AAPL", 142, memo_text="AAPL has a deep competitive moat.")]
    monkeypatch.setattr("src.agents.tools.memo_tools._search_memos", MagicMock(return_value=hits))

    result = search_memos(db=MagicMock(), query="deep competitive moats")

    assert result.sources[0].detail == {"memo_text": "AAPL has a deep competitive moat."}


# no hits -> empty data/sources, not an error
def test_search_memos_with_no_hits_returns_empty_result(monkeypatch):
    monkeypatch.setattr("src.agents.tools.memo_tools._search_memos", MagicMock(return_value=[]))

    result = search_memos(db=MagicMock(), query="something nobody's memo mentions")

    assert result.data == {"results": []}
    assert result.sources == []


# the query string reaches the underlying analytics search unmodified
def test_search_memos_passes_query_through(monkeypatch):
    mock_search = MagicMock(return_value=[])
    monkeypatch.setattr("src.agents.tools.memo_tools._search_memos", mock_search)

    search_memos(db="the-db", query="high-quality compounders")

    mock_search.assert_called_once_with("the-db", "high-quality compounders")


def test_dispatch_maps_search_memos_to_query_input(monkeypatch):
    db = MagicMock()
    mock_search = MagicMock(return_value=[])
    monkeypatch.setattr("src.agents.tools.memo_tools._search_memos", mock_search)

    DISPATCH["search_memos"](db, {"query": "moats"})

    mock_search.assert_called_once_with(db, "moats")


def test_schema_is_registered_in_tools():
    assert [t["name"] for t in TOOLS] == ["search_memos"]
    assert "query" in TOOLS[0]["input_schema"]["properties"]
