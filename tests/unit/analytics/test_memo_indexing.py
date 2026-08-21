from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy.dialects import postgresql

from src.analytics.memo_indexing import index_memos


def _result(ticker, memo):
    return SimpleNamespace(ticker=ticker, memo=memo)


def _bound_params(stmt):
    """Compiles a pg_insert(...).on_conflict_do_update(...) statement to inspect the values it was built with."""
    return stmt.compile(dialect=postgresql.dialect()).params


# no analyzed tickers -> nothing to embed or persist
def test_no_results_is_a_noop(monkeypatch):
    mock_embed_texts = MagicMock()
    monkeypatch.setattr("src.analytics.memo_indexing.embed_texts", mock_embed_texts)
    session = MagicMock()

    index_memos(session, scan_run_id=1, results=[])

    mock_embed_texts.assert_not_called()
    session.execute.assert_not_called()


# a result with no memo text (e.g. generate_memo() was never called) is skipped, not embedded as an empty string
def test_results_without_memo_text_are_skipped(monkeypatch):
    mock_embed_texts = MagicMock(return_value=[[0.1] * 3])
    monkeypatch.setattr("src.analytics.memo_indexing.embed_texts", mock_embed_texts)
    session = MagicMock()

    index_memos(session, scan_run_id=1, results=[_result("AAPL", "a memo"), _result("ZZZZ", "")])

    mock_embed_texts.assert_called_once_with(["a memo"])
    assert session.execute.call_count == 1


# every memo is embedded in one batched call, then one upsert per ticker for this scan_run_id
def test_indexes_one_row_per_result(monkeypatch):
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    mock_embed_texts = MagicMock(return_value=vectors)
    monkeypatch.setattr("src.analytics.memo_indexing.embed_texts", mock_embed_texts)
    session = MagicMock()

    index_memos(session, scan_run_id=7, results=[_result("MSFT", "msft memo"), _result("GOOGL", "googl memo")])

    mock_embed_texts.assert_called_once_with(["msft memo", "googl memo"])
    assert session.execute.call_count == 2

    first_stmt = session.execute.call_args_list[0].args[0]
    second_stmt = session.execute.call_args_list[1].args[0]
    first_params = _bound_params(first_stmt)
    second_params = _bound_params(second_stmt)

    assert first_params["ticker"] == "MSFT"
    assert first_params["scan_run_id"] == 7
    assert first_params["memo_text"] == "msft memo"
    assert first_params["embedding"] == [1.0, 2.0]

    assert second_params["ticker"] == "GOOGL"
    assert second_params["memo_text"] == "googl memo"
    assert second_params["embedding"] == [3.0, 4.0]


# the upsert targets the (ticker, scan_run_id) unique index, so re-running
# indexing for the same scan (e.g. a backfill re-run) updates rather than duplicates
def test_upsert_conflict_target_is_ticker_and_scan_run_id(monkeypatch):
    monkeypatch.setattr("src.analytics.memo_indexing.embed_texts", MagicMock(return_value=[[0.1]]))
    session = MagicMock()

    index_memos(session, scan_run_id=1, results=[_result("AAPL", "memo")])

    stmt = session.execute.call_args.args[0]
    assert set(stmt._post_values_clause.inferred_target_elements) == {"ticker", "scan_run_id"}


# an embeddings-API failure must not raise - a scan's factor scores must
# still count as persisted even if memo indexing can't reach Voyage this run
def test_embedding_failure_is_logged_and_swallowed(monkeypatch):
    monkeypatch.setattr(
        "src.analytics.memo_indexing.embed_texts", MagicMock(side_effect=RuntimeError("Voyage API down"))
    )
    session = MagicMock()

    index_memos(session, scan_run_id=1, results=[_result("AAPL", "memo")])  # must not raise

    session.execute.assert_not_called()
