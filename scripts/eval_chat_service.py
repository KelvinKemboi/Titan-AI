"""
CI-gating eval harness for the Chat/Agent Service: formalizes the ad-hoc "manual test" acceptance criteria scattered
across prior issues (see scripts/manual_test_tool_calling.py,
scripts/manual_test_phase2_questions.py, scripts/eval_conversation_memory.py)
into one repeatable eval set, run against the real `answer_question()` -
the same function `/chat` calls in production, not a re-implementation of
its tool-calling loop.
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

from dotenv import load_dotenv

from src.agents.chat_service import ChatAnswer, answer_question
from src.data.db import SessionLocal

load_dotenv()

Check = Callable[[ChatAnswer], Optional[str]]


# composable shape-checkers 
def all_of(*checks: Check) -> Check:
    def check(answer: ChatAnswer) -> Optional[str]:
        for c in checks:
            reason = c(answer)
            if reason:
                return reason
        return None
    return check


def cites_ticker(ticker: str, *, source_type: Optional[str] = None) -> Check:
    def check(answer: ChatAnswer) -> Optional[str]:
        matches = [s for s in answer.sources if s.ticker == ticker and (source_type is None or s.type == source_type)]
        if not matches:
            got = sorted({(s.ticker, s.type) for s in answer.sources})
            want = f"ticker={ticker}" + (f" type={source_type}" if source_type else "")
            return f"expected a source citing {want}, got {got}"
        return None
    return check


def cites_tickers_from_the_same_scan(*tickers: str) -> Check:
    """The acceptance criterion's own example: N tickers' factor_score
    sources must all carry the same ref_id (scan_run_id) - a comparison
    pinned to one scan, not stale-vs-fresh."""
    def check(answer: ChatAnswer) -> Optional[str]:
        by_ticker = {
            t: {s.ref_id for s in answer.sources if s.ticker == t and s.type == "factor_score"}
            for t in tickers
        }
        missing = [t for t, refs in by_ticker.items() if not refs]
        if missing:
            return f"missing factor_score source(s) for {missing} (have: {by_ticker})"
        all_ref_ids = {r for refs in by_ticker.values() for r in refs}
        if len(all_ref_ids) != 1:
            return f"expected {tickers} to share one scan_run_id, got {by_ticker}"
        return None
    return check


def has_source_type(*types: str) -> Check:
    def check(answer: ChatAnswer) -> Optional[str]:
        actual = {s.type for s in answer.sources}
        if not actual & set(types):
            return f"expected a source of type in {types}, got {sorted(actual)}"
        return None
    return check


def detail_field_is_not_none(ticker: str, field: str) -> Check:
    def check(answer: ChatAnswer) -> Optional[str]:
        matches = [s for s in answer.sources if s.ticker == ticker and s.detail and field in s.detail]
        if not matches:
            return f"no source for {ticker} carries a '{field}' detail field"
        if matches[0].detail.get(field) is None:
            return f"{ticker}'s '{field}' detail field is None"
        return None
    return check


def detail_field_is_truthy(ticker: str, field: str) -> Check:
    def check(answer: ChatAnswer) -> Optional[str]:
        matches = [s for s in answer.sources if s.ticker == ticker and s.detail and field in s.detail]
        if not matches:
            return f"no source for {ticker} carries a '{field}' detail field"
        if not matches[0].detail.get(field):
            return f"{ticker}'s '{field}' detail field is empty/falsy: {matches[0].detail.get(field)!r}"
        return None
    return check


def qoq_status_is(ticker: str, expected_status: str) -> Check:
    def check(answer: ChatAnswer) -> Optional[str]:
        matches = [s for s in answer.sources if s.ticker == ticker and s.detail and "qoq_changes" in s.detail]
        if not matches:
            return f"no source for {ticker} carries a 'qoq_changes' detail field"
        qoq = matches[0].detail.get("qoq_changes") or {}
        actual_status = qoq.get("status")
        if actual_status != expected_status:
            return f"expected {ticker} qoq_changes.status={expected_status!r}, got {actual_status!r}"
        return None
    return check


def detail_contains_text(ticker: str, field: str, substring: str) -> Check:
    def check(answer: ChatAnswer) -> Optional[str]:
        matches = [s for s in answer.sources if s.ticker == ticker and s.detail and field in s.detail]
        if not matches:
            return f"no source for {ticker} carries a '{field}' detail field"
        text = matches[0].detail.get(field) or ""
        if substring not in text:
            return f"expected {ticker}'s '{field}' to contain {substring!r}, got: {text[:200]!r}"
        return None
    return check


def no_source_for_ticker(ticker: str) -> Check:
    """Edge case: an unknown ticker must never end up cited as if it were
    real data - the model fabricating a source would be worse than it
    just saying it doesn't know."""
    def check(answer: ChatAnswer) -> Optional[str]:
        bad = [s for s in answer.sources if s.ticker == ticker]
        if bad:
            return f"expected no source for unknown ticker {ticker}, got {bad}"
        return None
    return check


def source_as_of_older_than(ticker: str, days: int) -> Check:
    """Edge case: stale data must still be attributed with its real
    (old) as_of, not silently presented as current - the response
    generator can only warn about staleness if the source metadata
    actually carries the true date."""
    def check(answer: ChatAnswer) -> Optional[str]:
        matches = [s for s in answer.sources if s.ticker == ticker and s.as_of is not None]
        if not matches:
            return f"no source with an as_of timestamp for {ticker}"
        as_of = matches[0].as_of
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - as_of
        if age < timedelta(days=days):
            return f"expected {ticker}'s source as_of to be >{days}d old, got {age.days}d old ({as_of})"
        return None
    return check


# the eval set
@dataclass
class EvalCase:
    id: str
    category: str
    question: str
    check: Check
    # Env vars this case's underlying tool needs beyond ANTHROPIC_API_KEY
    # (which every case needs and main() checks once, up front). 
    requires_env: List[str] = field(default_factory=list)


EVAL_SET: List[EvalCase] = [
    # Phase 1: the three product-brief example questions
    EvalCase(
        id="phase1_comparison_msft_googl",
        category="phase1",
        question="Why is Microsoft ranked above Google?",
        check=cites_tickers_from_the_same_scan("MSFT", "GOOGL"),
    ),
    EvalCase(
        id="phase1_structured_nvda_risks",
        category="phase1",
        question="What are Nvidia's biggest risks?",
        check=cites_ticker("NVDA", source_type="factor_score"),
    ),
    EvalCase(
        id="phase1_qualitative_deep_moat",
        category="phase1",
        question="Which companies have a deep competitive moat?",
        check=has_source_type("memo"),
        requires_env=["VOYAGE_API_KEY"],
    ),

    # Phase 2: the six product-brief example questions (all NVDA, which scripts/seed_eval_fixtures.py gives 2 ingested quarters)
    EvalCase(
        id="phase2_summarize_latest_call",
        category="phase2",
        question="Summarize NVDA's latest earnings call.",
        check=all_of(
            cites_ticker("NVDA", source_type="earnings_insight"),
            detail_field_is_not_none("NVDA", "summary"),
        ),
    ),
    EvalCase(
        id="phase2_guidance_improved_or_worsened",
        category="phase2",
        question="Did NVDA's guidance improve or worsen last quarter?",
        check=detail_field_is_not_none("NVDA", "guidance_direction"),
    ),
    EvalCase(
        id="phase2_why_stock_moved",
        category="phase2",
        question="Why did NVDA's management sound the way they did on the earnings call?",
        check=detail_field_is_not_none("NVDA", "sentiment_score"),
    ),
    EvalCase(
        id="phase2_biggest_risks",
        category="phase2",
        question="What are NVDA's biggest risks, based on its latest earnings call?",
        check=detail_field_is_truthy("NVDA", "risks"),
    ),
    EvalCase(
        id="phase2_what_changed_last_quarter",
        category="phase2",
        question="What changed for NVDA from last quarter?",
        check=qoq_status_is("NVDA", "ok"),
    ),
    EvalCase(
        id="phase2_analyst_report",
        category="phase2",
        question="Give me a full analyst report on NVDA.",
        check=detail_contains_text("NVDA", "memo", "Recent Earnings"),
    ),

    # Edge cases surfaced during implementation
    EvalCase(
        id="edge_unknown_ticker",
        category="edge_case",
        question="What's ZZZZNOTREAL's factor score?",
        check=no_source_for_ticker("ZZZZNOTREAL"),
    ),
    EvalCase(
        id="edge_insufficient_history",
        category="edge_case",
        question="What changed for AMD from last quarter?",
        check=qoq_status_is("AMD", "insufficient_history"),
    ),
    EvalCase(
        id="edge_stale_data",
        category="edge_case",
        question="What's ZEVALSTALE's current rating?",
        check=all_of(
            cites_ticker("ZEVALSTALE", source_type="factor_score"),
            source_as_of_older_than("ZEVALSTALE", days=30),
        ),
    ),
]


# runner
@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    reason: Optional[str]
    answer: Optional[ChatAnswer]
    skipped: bool = False


def evaluate_case(case: EvalCase, ask: Callable[[str], ChatAnswer]) -> EvalResult:
    """Runs one case's question through `ask` and applies its check.
    A crash while asking (e.g. an unexpected exception, as opposed to
    answer_question's own graceful anthropic.APIError -> FALLBACK_RESPONSE
    handling) is recorded as a failure with the exception as the reason,
    not re-raised - one case's crash must not abort the whole run and
    hide every other case's result.

    A case with an unmet `requires_env` is skipped (not failed) - it
    doesn't count against the pass/fail gate, distinct from a real
    regression."""
    missing_env = [var for var in case.requires_env if not os.environ.get(var)]
    if missing_env:
        return EvalResult(
            case=case, passed=True, skipped=True,
            reason=f"skipped: {', '.join(missing_env)} not configured", answer=None,
        )

    try:
        answer = ask(case.question)
    except Exception as exc:
        return EvalResult(case=case, passed=False, reason=f"ask() raised: {exc!r}", answer=None)

    reason = case.check(answer)
    return EvalResult(case=case, passed=reason is None, reason=reason, answer=answer)


def run_eval(cases: List[EvalCase], ask: Callable[[str], ChatAnswer]) -> List[EvalResult]:
    return [evaluate_case(case, ask) for case in cases]


def _print_report(results: List[EvalResult]) -> bool:
    """Prints a pass/fail/skip report grouped by category; returns True
    iff no case actually failed (a skip is neither a pass nor a
    regression, and doesn't affect the gate)."""
    by_category: dict = {}
    for result in results:
        by_category.setdefault(result.case.category, []).append(result)

    for category, category_results in by_category.items():
        print(f"\n=== {category} ===")
        for result in category_results:
            status = "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL")
            print(f"[{status}] {result.case.id}: {result.case.question}")
            if result.skipped or not result.passed:
                print(f"       reason: {result.reason}")
            if result.answer is not None:
                print(f"       response: {result.answer.response[:200]}")

    skipped = sum(1 for r in results if r.skipped)
    failed = [r for r in results if not r.passed and not r.skipped]
    passed = len(results) - skipped - len(failed)
    print(f"\n{passed}/{len(results)} cases passed, {len(failed)} failed, {skipped} skipped.")
    return not failed


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("FAIL: ANTHROPIC_API_KEY is not set - every case needs it to call the Chat/Agent Service at all.")
        sys.exit(1)

    db = SessionLocal()
    try:
        results = run_eval(EVAL_SET, lambda question: answer_question(db, question))
    finally:
        db.close()

    all_passed = _print_report(results)
    if not all_passed:
        print("\nFAIL: one or more eval cases regressed.")
        sys.exit(1)
    print("\nPASS: every non-skipped eval case produced the expected answer shape.")


if __name__ == "__main__":
    main()
