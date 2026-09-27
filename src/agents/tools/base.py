from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Source(BaseModel):
    """
    Source-attribution contract: every retrieval tool returns this
    alongside its data, so a response can cite exactly which scan or
    ticker backs each claim instead of stating facts from memory.
    `detail` carries whatever data the tool already computed (a
    factor_score's per-factor breakdown, a memo's actual text) so a
    citation UI can show it without a second round-trip. Left untyped
    since each tool's `detail` shape differs; the UI keys off `type`.
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
