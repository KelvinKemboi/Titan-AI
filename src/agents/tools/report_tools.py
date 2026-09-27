"""
get_analyst_report: Claude tool schema + implementation over
src/analytics/report.py's combined factor-score + earnings-insight memo
"""
from sqlalchemy.orm import Session

from src.agents.tools.base import Source, ToolResult
from src.analytics.report import generate_ticker_report
from src.api.routes.earnings import get_earnings_for_ticker as _get_earnings_for_ticker

GET_ANALYST_REPORT_SCHEMA = {
    "name": "get_analyst_report",
    "description": (
        "Get Titan's full analyst-style writeup for a single ticker - "
        "rating, investment thesis, and key drivers from the latest "
        "factor scores, plus (when available) a 'Recent Earnings' "
        "section (summary, guidance direction, top risk) from the "
        "ticker's latest earnings call. Use for 'give me a report on X' "
        "/ 'what's your take on X' style questions wanting the full "
        "writeup - NOT for a single numeric answer (use "
        "get_factor_scores for that)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'AAPL'",
            },
        },
        "required": ["ticker"],
    },
}


def get_analyst_report(db: Session, ticker: str) -> ToolResult:
    """
    Tool implementation backing GET_ANALYST_REPORT_SCHEMA. Raises
    ValueError for a ticker with no factor_scores row. Returns two
    sources when the memo's "Recent Earnings" section is populated (the
    factor_scores row and the earnings_insights row it also draws on),
    one otherwise, so every claim in the memo traces to its own row.
    """
    report = generate_ticker_report(db, ticker)

    sources = [
        Source(
            type="factor_score",
            ticker=report.ticker,
            ref_id=report.scan_run_id,
            as_of=report.computed_at,
            detail={
                "composite_score": report.composite_score,
                "rating": report.rating,
                "memo": report.memo,
            },
        )
    ]
    if report.has_earnings_data:
        transcripts = _get_earnings_for_ticker(db, report.ticker)  # most-recent first
        if transcripts:
            latest = transcripts[0]
            sources.append(
                Source(
                    type="earnings_insight",
                    ticker=report.ticker,
                    ref_id=latest.transcript_id,
                    as_of=latest.ingested_at,
                    detail={"fiscal_year": latest.fiscal_year, "fiscal_quarter": latest.fiscal_quarter},
                )
            )

    return ToolResult(data=report.model_dump(mode="json"), sources=sources)


TOOLS = [GET_ANALYST_REPORT_SCHEMA]
DISPATCH = {"get_analyst_report": lambda db, tool_input: get_analyst_report(db, tool_input["ticker"])}
