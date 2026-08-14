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
    """

    type: str
    ticker: str
    ref_id: int
    as_of: Optional[datetime] = None

class ToolResult(BaseModel):
    """Envelope every tool in src/agents/tools/ returns: `data` (typed,
    matching the underlying service function's own response shape
    verbatim) plus the `sources` the response generator must cite."""

    data: Dict[str, Any]
    sources: List[Source]
