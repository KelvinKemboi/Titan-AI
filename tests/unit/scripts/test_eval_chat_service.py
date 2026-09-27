from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.chat_service import ChatAnswer
from src.agents.tools.base import Source
from scripts.eval_chat_service import (
    EVAL_SET,
    EvalCase,
    all_of,
    cites_ticker,
    cites_tickers_from_the_same_scan,
    detail_contains_text,
    detail_field_is_not_none,
    detail_field_is_truthy,
    evaluate_case,
    has_source_type,
    main,
    no_source_for_ticker,
    qoq_status_is,
    run_eval,
    source_as_of_older_than,
)


def _source(ticker="AAPL", ref_id=1, type="factor_score", as_of=None, detail=None):
    return Source(type=type, ticker=ticker, ref_id=ref_id, as_of=as_of, detail=detail)


def _answer(*sources, response="an answer"):
    return ChatAnswer(response=response, sources=list(sources))


# shape-checkers: cites_ticker 
def test_cites_ticker_passes_when_present():
    answer = _answer(_source(ticker="AAPL"))
    assert cites_ticker("AAPL")(answer) is None


def test_cites_ticker_fails_when_absent():
    answer = _answer(_source(ticker="MSFT"))
    reason = cites_ticker("AAPL")(answer)
    assert reason is not None
    assert "AAPL" in reason


def test_cites_ticker_honors_source_type_filter():
    answer = _answer(_source(ticker="AAPL", type="memo"))
    assert cites_ticker("AAPL", source_type="factor_score")(answer) is not None
    assert cites_ticker("AAPL", source_type="memo")(answer) is None


# cites_tickers_from_the_same_scan
def test_cites_tickers_from_the_same_scan_passes_when_ref_ids_match():
    answer = _answer(
        _source(ticker="MSFT", ref_id=42, type="factor_score"),
        _source(ticker="GOOGL", ref_id=42, type="factor_score"),
    )
    assert cites_tickers_from_the_same_scan("MSFT", "GOOGL")(answer) is None


def test_cites_tickers_from_the_same_scan_fails_on_mismatched_scan_run_id():
    answer = _answer(
        _source(ticker="MSFT", ref_id=42, type="factor_score"),
        _source(ticker="GOOGL", ref_id=43, type="factor_score"),
    )
    reason = cites_tickers_from_the_same_scan("MSFT", "GOOGL")(answer)
    assert reason is not None
    assert "share one scan_run_id" in reason


def test_cites_tickers_from_the_same_scan_fails_when_one_ticker_missing():
    answer = _answer(_source(ticker="MSFT", ref_id=42, type="factor_score"))
    reason = cites_tickers_from_the_same_scan("MSFT", "GOOGL")(answer)
    assert reason is not None
    assert "GOOGL" in reason


# has_source_type 
def test_has_source_type_passes_when_any_type_matches():
    answer = _answer(_source(type="memo"))
    assert has_source_type("memo", "earnings_chunk")(answer) is None


def test_has_source_type_fails_when_no_type_matches():
    answer = _answer(_source(type="factor_score"))
    assert has_source_type("memo")(answer) is not None


# detail_field_is_not_none / detail_field_is_truthy 
def test_detail_field_is_not_none_passes_when_present_and_not_none():
    answer = _answer(_source(ticker="NVDA", detail={"summary": "a summary"}))
    assert detail_field_is_not_none("NVDA", "summary")(answer) is None


def test_detail_field_is_not_none_fails_when_field_is_none():
    answer = _answer(_source(ticker="NVDA", detail={"summary": None}))
    assert detail_field_is_not_none("NVDA", "summary")(answer) is not None


def test_detail_field_is_not_none_fails_when_field_absent():
    answer = _answer(_source(ticker="NVDA", detail={"other": 1}))
    assert detail_field_is_not_none("NVDA", "summary")(answer) is not None


def test_detail_field_is_truthy_fails_on_empty_list():
    answer = _answer(_source(ticker="NVDA", detail={"risks": []}))
    assert detail_field_is_truthy("NVDA", "risks")(answer) is not None


def test_detail_field_is_truthy_passes_on_nonempty_list():
    answer = _answer(_source(ticker="NVDA", detail={"risks": [{"risk": "x", "quote": "y"}]}))
    assert detail_field_is_truthy("NVDA", "risks")(answer) is None


# qoq_status_is 
def test_qoq_status_is_passes_on_matching_status():
    answer = _answer(_source(ticker="NVDA", detail={"qoq_changes": {"status": "ok"}}))
    assert qoq_status_is("NVDA", "ok")(answer) is None


def test_qoq_status_is_fails_on_mismatched_status():
    answer = _answer(_source(ticker="AMD", detail={"qoq_changes": {"status": "insufficient_history"}}))
    reason = qoq_status_is("AMD", "ok")(answer)
    assert reason is not None
    assert "insufficient_history" in reason


def test_qoq_status_is_fails_when_no_qoq_detail_present():
    answer = _answer(_source(ticker="AMD", detail={"summary": "x"}))
    assert qoq_status_is("AMD", "ok")(answer) is not None


# detail_contains_text 
def test_detail_contains_text_passes_on_substring_match():
    answer = _answer(_source(ticker="NVDA", detail={"memo": "...**Recent Earnings:**..."}))
    assert detail_contains_text("NVDA", "memo", "Recent Earnings")(answer) is None


def test_detail_contains_text_fails_when_substring_absent():
    answer = _answer(_source(ticker="NVDA", detail={"memo": "no earnings section here"}))
    assert detail_contains_text("NVDA", "memo", "Recent Earnings")(answer) is not None


# no_source_for_ticker
def test_no_source_for_ticker_passes_when_absent():
    answer = _answer(_source(ticker="MSFT"))
    assert no_source_for_ticker("ZZZZNOTREAL")(answer) is None


def test_no_source_for_ticker_fails_when_present():
    answer = _answer(_source(ticker="ZZZZNOTREAL"))
    reason = no_source_for_ticker("ZZZZNOTREAL")(answer)
    assert reason is not None
    assert "ZZZZNOTREAL" in reason


# source_as_of_older_than 
def test_source_as_of_older_than_passes_when_old_enough():
    old = datetime.now(timezone.utc) - timedelta(days=120)
    answer = _answer(_source(ticker="ZEVALSTALE", as_of=old))
    assert source_as_of_older_than("ZEVALSTALE", days=30)(answer) is None


def test_source_as_of_older_than_fails_when_too_fresh():
    fresh = datetime.now(timezone.utc) - timedelta(days=1)
    answer = _answer(_source(ticker="ZEVALSTALE", as_of=fresh))
    reason = source_as_of_older_than("ZEVALSTALE", days=30)(answer)
    assert reason is not None
    assert "1d old" in reason


def test_source_as_of_older_than_fails_when_no_as_of_present():
    answer = _answer(_source(ticker="ZEVALSTALE", as_of=None))
    assert source_as_of_older_than("ZEVALSTALE", days=30)(answer) is not None


def test_source_as_of_older_than_handles_naive_datetimes():
    old_naive = datetime.now() - timedelta(days=120)
    answer = _answer(_source(ticker="ZEVALSTALE", as_of=old_naive))
    assert source_as_of_older_than("ZEVALSTALE", days=30)(answer) is None


# all_of
def test_all_of_passes_only_when_every_check_passes():
    passing = lambda answer: None
    failing = lambda answer: "nope"
    answer = _answer()
    assert all_of(passing, passing)(answer) is None
    assert all_of(passing, failing)(answer) == "nope"


def test_all_of_short_circuits_on_the_first_failure():
    calls = []

    def first(answer):
        calls.append("first")
        return "first failed"

    def second(answer):
        calls.append("second")
        return None

    all_of(first, second)(_answer())
    assert calls == ["first"]


# evaluate_case / run_eval: the actual "CI fails on regression" mechanics
def _case(id="c1", check=lambda answer: None):
    return EvalCase(id=id, category="phase1", question="a question", check=check)


def test_evaluate_case_passes_when_check_returns_none():
    result = evaluate_case(_case(check=lambda a: None), ask=lambda q: _answer())
    assert result.passed is True
    assert result.reason is None


def test_evaluate_case_fails_when_check_returns_a_reason():
    result = evaluate_case(_case(check=lambda a: "shape mismatch"), ask=lambda q: _answer())
    assert result.passed is False
    assert result.reason == "shape mismatch"


# a crash while asking must be recorded as a failure, not propagate and abort the whole run
def test_evaluate_case_does_not_raise_when_ask_itself_raises():
    def broken_ask(question):
        raise RuntimeError("boom")

    result = evaluate_case(_case(), ask=broken_ask)
    assert result.passed is False
    assert "boom" in result.reason


def test_run_eval_runs_every_case_even_after_an_earlier_one_crashes():
    call_count = {"n": 0}

    def ask(question):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return _answer()

    cases = [_case(id="a"), _case(id="b")]
    results = run_eval(cases, ask)

    assert [r.case.id for r in results] == ["a", "b"]
    assert results[0].passed is False  # crashed
    assert results[1].passed is True  # still ran, unaffected


def test_run_eval_reports_mixed_pass_and_fail():
    cases = [
        _case(id="passes", check=lambda a: None),
        _case(id="fails", check=lambda a: "bad shape"),
    ]
    results = run_eval(cases, ask=lambda q: _answer())

    by_id = {r.case.id: r for r in results}
    assert by_id["passes"].passed is True
    assert by_id["fails"].passed is False


# main(): the actual CI exit-code gate
def test_main_exits_zero_when_every_case_passes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("scripts.eval_chat_service.SessionLocal", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("scripts.eval_chat_service.answer_question", lambda db, q: _answer(_source(ticker="X")))
    monkeypatch.setattr(
        "scripts.eval_chat_service.EVAL_SET",
        [_case(id="always_passes", check=lambda a: None)],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code in (0, None)
    # no SystemExit at all is also success (falls through main() normally)


# a regressed answer shape must make the process exit non-zero, not just print FAIL
def test_main_exits_nonzero_when_a_case_regresses(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("scripts.eval_chat_service.SessionLocal", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("scripts.eval_chat_service.answer_question", lambda db, q: _answer())
    monkeypatch.setattr(
        "scripts.eval_chat_service.EVAL_SET",
        [_case(id="regressed", check=lambda a: "expected ticker citation, found none")],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_exits_nonzero_when_the_chat_service_itself_crashes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("scripts.eval_chat_service.SessionLocal", MagicMock(return_value=MagicMock()))

    def crash(db, q):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("scripts.eval_chat_service.answer_question", crash)
    monkeypatch.setattr(
        "scripts.eval_chat_service.EVAL_SET",
        [_case(id="crashes")],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_exits_nonzero_immediately_when_anthropic_api_key_is_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mock_session = MagicMock()
    monkeypatch.setattr("scripts.eval_chat_service.SessionLocal", mock_session)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    mock_session.assert_not_called()  # fails before ever touching the DB


# requires_env: a case whose dependency isn't configured skips, doesn't fail 
def test_evaluate_case_skips_when_a_required_env_var_is_unset(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_TEST_VAR", raising=False)
    case = _case(check=lambda a: "would have failed if it ran")
    case.requires_env = ["SOME_UNSET_TEST_VAR"]

    def ask_that_must_not_be_called(question):
        raise AssertionError("ask() should not be called for a skipped case")

    result = evaluate_case(case, ask=ask_that_must_not_be_called)

    assert result.skipped is True
    assert result.passed is True  # a skip never counts as a regression
    assert "SOME_UNSET_TEST_VAR" in result.reason


def test_evaluate_case_runs_normally_when_required_env_var_is_set(monkeypatch):
    monkeypatch.setenv("SOME_SET_TEST_VAR", "value")
    case = _case(check=lambda a: None)
    case.requires_env = ["SOME_SET_TEST_VAR"]

    result = evaluate_case(case, ask=lambda q: _answer())

    assert result.skipped is False
    assert result.passed is True


def test_a_skipped_case_does_not_fail_the_overall_gate(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("scripts.eval_chat_service.SessionLocal", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("scripts.eval_chat_service.answer_question", lambda db, q: _answer())
    skip_case = _case(id="needs_a_key", check=lambda a: "unreachable")
    skip_case.requires_env = ["SOME_UNSET_TEST_VAR_2"]
    monkeypatch.delenv("SOME_UNSET_TEST_VAR_2", raising=False)
    monkeypatch.setattr("scripts.eval_chat_service.EVAL_SET", [skip_case])

    try:
        main()
    except SystemExit as exc:
        assert exc.code in (0, None)


# the real eval set itself
def test_eval_set_covers_all_three_phase1_product_brief_questions():
    phase1_questions = {c.question for c in EVAL_SET if c.category == "phase1"}
    assert len(phase1_questions) == 3


def test_eval_set_covers_all_six_phase2_product_brief_questions():
    phase2_questions = {c.question for c in EVAL_SET if c.category == "phase2"}
    assert len(phase2_questions) == 6


def test_eval_set_covers_every_named_edge_case_category():
    edge_case_ids = {c.id for c in EVAL_SET if c.category == "edge_case"}
    assert any("unknown" in cid for cid in edge_case_ids)
    assert any("insufficient_history" in cid for cid in edge_case_ids)
    assert any("stale" in cid for cid in edge_case_ids)


def test_every_case_has_a_unique_id():
    ids = [c.id for c in EVAL_SET]
    assert len(ids) == len(set(ids))
