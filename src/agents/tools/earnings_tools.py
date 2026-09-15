"""
get_earnings_insight / search_earnings: Claude tool schemas +
implementations over Titan's ingested earnings call data
"""
from typing import List, Optional

from sqlalchemy.orm import Session

from src.agents.tools.base import Source, ToolResult
from src.analytics.earnings_search import search_earnings as _search_earnings
from src.api.routes.earnings import get_earnings_for_ticker as _get_earnings_for_ticker

GET_EARNINGS_INSIGHT_SCHEMA = {
    "name": "get_earnings_insight",
    "description": (
        "Get a single ticker's earnings call insights: summary, forward "
        "guidance direction (with supporting quote), management tone "
        "sentiment score, and risks management discussed on the call. "
        "Defaults to the ticker's latest ingested quarter if `quarter` is "
        "omitted. Use for 'summarize X's latest earnings call' / 'did X's "
        "guidance improve' / 'what risks did X mention' questions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'NVDA'",
            },
            "quarter": {
                "type": "integer",
                "enum": [1, 2, 3, 4],
                "description": "Fiscal quarter (1-4). Omit for the latest ingested quarter.",
            },
        },
        "required": ["ticker"],
    },
}

SEARCH_EARNINGS_SCHEMA = {
    "name": "search_earnings",
    "description": (
        "Semantic search over Titan's ingested earnings call transcripts "
        "(prepared remarks and Q&A alike), across every ticker. Use for "
        "qualitative questions with no single ticker named, e.g. 'which "
        "companies discussed AI capex on their earnings calls' - NOT for "
        "one named ticker's own earnings call (use get_earnings_insight)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language question, e.g. 'AI capex spending plans'",
            },
        },
        "required": ["query"],
    },
}


def get_earnings_insight(db: Session, ticker: str, quarter: Optional[int] = None) -> ToolResult:
    """Tool implementation backing GET_EARNINGS_INSIGHT_SCHEMA."""
    ticker = ticker.strip().upper()
    transcripts = _get_earnings_for_ticker(db, ticker)  # most-recent fiscal_year/fiscal_quarter first
    if not transcripts:
        raise ValueError(f"No earnings transcripts ingested for ticker '{ticker}'")

    if quarter is not None:
        fiscal_quarter = f"Q{quarter}"
        match = next((t for t in transcripts if t.fiscal_quarter == fiscal_quarter), None)
        if match is None:
            raise ValueError(f"No {fiscal_quarter} earnings transcript ingested for ticker '{ticker}'")
    else:
        match = transcripts[0]

    insight = match.insight
    insight_fields = {
        "summary": insight.summary if insight else None,
        "guidance_direction": insight.guidance_direction if insight else None,
        "guidance_quote": insight.guidance_quote if insight else None,
        # Numeric (DB) -> float: data/detail are loosely-typed Dict[str, Any]
        "sentiment_score": float(insight.sentiment_score) if insight and insight.sentiment_score is not None else None,
        "risks": insight.risks if insight else [],
    }
    return ToolResult(
        data={
            "ticker": ticker,
            "fiscal_year": match.fiscal_year,
            "fiscal_quarter": match.fiscal_quarter,
            **insight_fields,
        },
        sources=[
            Source(
                type="earnings_insight",
                ticker=ticker,
                ref_id=match.transcript_id,
                as_of=match.ingested_at,
                detail={
                    "fiscal_year": match.fiscal_year,
                    "fiscal_quarter": match.fiscal_quarter,
                    **insight_fields,
                },
            )
        ],
    )


def search_earnings(db: Session, query: str) -> ToolResult:
    """Tool implementation backing SEARCH_EARNINGS_SCHEMA."""
    hits = _search_earnings(db, query)
    return ToolResult(
        data={"results": [hit.model_dump(mode="json") for hit in hits]},
        sources=[
            Source(
                type="earnings_chunk",
                ticker=hit.ticker,
                ref_id=hit.transcript_id,
                as_of=hit.as_of,
                detail={"chunk_type": hit.chunk_type, "chunk_text": hit.chunk_text},
            )
            for hit in hits
        ],
    )


TOOLS: List[dict] = [GET_EARNINGS_INSIGHT_SCHEMA, SEARCH_EARNINGS_SCHEMA]
DISPATCH = {
    "get_earnings_insight": lambda db, tool_input: get_earnings_insight(
        db, tool_input["ticker"], tool_input.get("quarter"),
    ),
    "search_earnings": lambda db, tool_input: search_earnings(db, tool_input["query"]),
}
