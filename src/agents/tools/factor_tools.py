from typing import List

from sqlalchemy.orm import Session

from src.agents.tools.base import Source, ToolResult
from src.analytics.explain import explain_ticker
from src.api.routes.compare import compare_tickers as _compare_tickers
from src.data.models import ScanRun

GET_FACTOR_SCORES_SCHEMA = {
    "name": "get_factor_scores",
    "description": (
        "Get a single ticker's latest factor scores (Value, Momentum, "
        "Quality, Solvency, Volatility) with each factor's weight, its "
        "contribution to the composite score, and the specific metric "
        "that drove it. Use for 'why is X ranked...' / 'explain X's "
        "score' questions about ONE ticker."
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

COMPARE_TICKERS_SCHEMA = {
    "name": "compare_tickers",
    "description": (
        "Compare 2 or more tickers' latest factor scores side by side, "
        "all pinned to the same scan so the comparison is never stale-"
        "vs-fresh. Returns the biggest score deltas between the first "
        "two tickers, sorted by magnitude. Use for any 'X vs Y' / 'why "
        "is X above Y' comparison question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "2+ ticker symbols to compare, e.g. ['MSFT', 'GOOGL']",
            },
        },
        "required": ["tickers"],
    },
}

TOOLS = [GET_FACTOR_SCORES_SCHEMA, COMPARE_TICKERS_SCHEMA]

def _scan_run_as_of(db: Session, scan_run_id: int):
    """The ScanRun's completed_at, or started_at if it hasn't completed."""
    scan_run = db.get(ScanRun, scan_run_id)
    if scan_run is None:
        return None
    return scan_run.completed_at or scan_run.started_at

def get_factor_scores(db: Session, ticker: str) -> ToolResult:
    """Tool implementation backing GET_FACTOR_SCORES_SCHEMA."""
    explanation = explain_ticker(db, ticker)
    as_of = _scan_run_as_of(db, explanation.scan_run_id)
    return ToolResult(
        data=explanation.model_dump(mode="json"),
        sources=[
            Source(
                type="factor_score",
                ticker=explanation.ticker,
                ref_id=explanation.scan_run_id,
                as_of=as_of,
                detail={
                    "composite_score": explanation.composite_score,
                    "rating": explanation.rating,
                    "factors": [f.model_dump(mode="json") for f in explanation.factors],
                },
            )
        ],
    )

def compare_tickers(db: Session, tickers: List[str]) -> ToolResult:
    """Tool implementation backing COMPARE_TICKERS_SCHEMA."""
    # Dedupe first, or ["MSFT", "msft"] normalizes to a degenerate "MSFT vs MSFT" comparison.
    seen = set()
    deduped = []
    for t in tickers:
        normalized = t.strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    result = _compare_tickers(db, deduped)
    as_of = _scan_run_as_of(db, result.scan_run_id)
    return ToolResult(
        data=result.model_dump(mode="json"),
        sources=[
            Source(
                type="factor_score",
                ticker=t.ticker,
                ref_id=result.scan_run_id,
                as_of=as_of,
                detail=t.model_dump(mode="json", exclude={"ticker"}),
            )
            for t in result.tickers
        ],
    )


DISPATCH = {
    "get_factor_scores": lambda db, tool_input: get_factor_scores(db, tool_input["ticker"]),
    "compare_tickers": lambda db, tool_input: compare_tickers(db, tool_input["tickers"]),
}
