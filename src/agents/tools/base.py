from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Source(BaseModel):
    """
    Source-attribution contract: every retrieval
    tool must return this alongside its data, so a
    response generator downstream can cite exactly which scan/ticker
    backs each claim instead of stating facts from memory. Enforced at
    the tool-schema level.

    `detail` is an optional, type-specific bag of the underlying data a
    tool already computed (e.g. a factor_score source's composite score
    and per-factor breakdown, or a memo source's actual memo text) - it
    exists so a citation UI can let a user verify a claim without a
    second round-trip to re-fetch what the tool already had in hand.
    Left untyped (rather than a per-type subclass) since each tool's
    `detail` shape is unrelated to any other's; the UI keys off `type`
    to decide how to render it.
    """

    type: str
    ticker: str
    ref_id: int
    as_of: Optional[datetime] = None
    detail: Optional[Dict[str, Any]] = None

class ToolResult(BaseModel):
    """Envelope every tool in src/agents/tools/ returns: `data` (typed,
    matching the underlying service function's own response shape
    verbatim) plus the `sources` the response generator must cite."""

    data: Dict[str, Any]
    sources: List[Source]
