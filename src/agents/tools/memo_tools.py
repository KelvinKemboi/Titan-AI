"""
search_memos: Claude tool schema + implementation for semantic search over
Titan's own analyst memos (memo_embeddings) - the qualitative counterpart
to factor_tools.py's structured numeric lookups.
"""
from typing import List

from sqlalchemy.orm import Session

from src.agents.tools.base import Source, ToolResult
from src.analytics.memo_search import search_memos as _search_memos

SEARCH_MEMOS_SCHEMA = {
    "name": "search_memos",
    "description": (
        "Semantic search over Titan's analyst memos - the qualitative "
        "writeup (investment thesis, competitive moat / quality "
        "commentary, momentum and risk narrative) generated per ticker "
        "per scan. Use for qualitative questions with no single ticker "
        "named, e.g. 'which companies have a deep competitive moat' or "
        "'find high-quality compounders' - NOT for a named ticker's "
        "numeric score (use get_factor_scores) or an X-vs-Y comparison "
        "(use compare_tickers)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language qualitative question, e.g. 'deep competitive moats'",
            },
        },
        "required": ["query"],
    },
}


def search_memos(db: Session, query: str) -> ToolResult:
    """Tool implementation backing SEARCH_MEMOS_SCHEMA."""
    hits = _search_memos(db, query)
    return ToolResult(
        data={"results": [hit.model_dump(mode="json") for hit in hits]},
        sources=[
            Source(
                type="memo",
                ticker=hit.ticker,
                ref_id=hit.scan_run_id,
                as_of=hit.as_of,
                detail={"memo_text": hit.memo_text},
            )
            for hit in hits
        ],
    )


TOOLS = [SEARCH_MEMOS_SCHEMA]
DISPATCH = {"search_memos": lambda db, tool_input: search_memos(db, tool_input["query"])}
