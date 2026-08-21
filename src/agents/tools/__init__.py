"""
Aggregates every tool submodule's TOOLS/DISPATCH into the single registry
the Chat/Agent Service (src/agents/chat_service.py) passes to Claude - so
adding a new tool category (memo_tools, and later Phase 2's earnings
tools) means adding a submodule here, not touching chat_service.py.
"""
from typing import List

from sqlalchemy.orm import Session

from src.agents.tools import factor_tools, memo_tools
from src.agents.tools.base import ToolResult

TOOLS: List[dict] = factor_tools.TOOLS + memo_tools.TOOLS
DISPATCH = {**factor_tools.DISPATCH, **memo_tools.DISPATCH}


def call_tool(db: Session, name: str, tool_input: dict) -> ToolResult:
    """
    Dispatches a Claude tool_use block (`name` + `input`) to its
    implementation. Raises ValueError for an unrecognized tool name or a
    failed lookup (e.g. unknown ticker) - the tool implementation's own
    ValueError.
    """
    if name not in DISPATCH:
        raise ValueError(f"Unknown tool '{name}'")
    return DISPATCH[name](db, tool_input)
