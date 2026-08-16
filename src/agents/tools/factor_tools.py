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

# scan_run_as_of returns the completed_at timestamp of the ScanRun with the given scan_run_id, or the started_at timestamp if completed_at is None
def _scan_run_as_of(db: Session, scan_run_id: int):
    scan_run = db.get(ScanRun, scan_run_id)
    if scan_run is None:
        return None
    return scan_run.completed_at or scan_run.started_at

# factor_scores_tool_result returns a ToolResult object containing the explanation of the factor scores for a given ticker, along with the source information including the scan_run_id and the as_of timestamp
def get_factor_scores(db: Session, ticker: str) -> ToolResult:
    """Tool implementation backing GET_FACTOR_SCORES_SCHEMA (#9's Explanation Engine)."""
    explanation = explain_ticker(db, ticker)
    as_of = _scan_run_as_of(db, explanation.scan_run_id) # retrieves the as_of timestamp for the scan_run_id associated with the explanation
    return ToolResult(
        data=explanation.model_dump(mode="json"),
        sources=[
            Source(
                type="factor_score",
                ticker=explanation.ticker,
                ref_id=explanation.scan_run_id,
                as_of=as_of,
            )
        ],
    )

# compare_tickers_tool_result returns a ToolResult object containing the comparison of factor scores for a list of tickers, along with the source information for each ticker including the scan_run_id and the as_of timestamp
def compare_tickers(db: Session, tickers: List[str]) -> ToolResult:
    """Tool implementation backing COMPARE_TICKERS_SCHEMA."""
    result = _compare_tickers(db, [t.strip().upper() for t in tickers])
    as_of = _scan_run_as_of(db, result.scan_run_id)
    return ToolResult(
        data=result.model_dump(mode="json"),
        sources=[
            Source(
                type="factor_score",
                ticker=t.ticker,
                ref_id=result.scan_run_id,
                as_of=as_of,
            )
            for t in result.tickers
        ],
    )


DISPATCH = {
    "get_factor_scores": lambda db, tool_input: get_factor_scores(db, tool_input["ticker"]),
    "compare_tickers": lambda db, tool_input: compare_tickers(db, tool_input["tickers"]),
}

# call_tool dispatches a tool call to the appropriate implementation based on the tool name and input, raising a ValueError for unrecognized tool names or failed lookups
def call_tool(db: Session, name: str, tool_input: dict) -> ToolResult:
    """
    Dispatches a Claude tool_use block (`name` + `input`) to its
    implementation. Raises ValueError for an unrecognized tool name or a
    failed lookup (e.g. unknown ticker).
    """
    if name not in DISPATCH:
        raise ValueError(f"Unknown tool '{name}'")
    return DISPATCH[name](db, tool_input)
