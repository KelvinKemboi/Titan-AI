"""
Queryable LLM spend and request tracing over the llm_calls/tool_calls
rows src/observability/tracing.py persists. Session-level spend is
derived via chat_messages.request_id -> llm_calls.request_id rather
than a session_id duplicated onto every trace row, since llm_calls and
tool_calls have no session_id of their own.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.data.models import ChatMessage, LLMCall, ToolCall


class CallTypeBreakdown(BaseModel):
    call_type: str
    call_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float] = None


class SpendSummary(BaseModel):
    call_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float] = None  # None only when call_count == 0
    by_call_type: List[CallTypeBreakdown] = []


class LLMCallDetail(BaseModel):
    call_type: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    created_at: Optional[datetime] = None


class ToolCallDetail(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any] = {}
    success: bool
    error: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    created_at: Optional[datetime] = None


class RequestTrace(BaseModel):
    request_id: str
    llm_calls: List[LLMCallDetail] = []
    tool_calls: List[ToolCallDetail] = []


def _summarize(rows: List[LLMCall]) -> SpendSummary:
    """Pure aggregation over already-fetched LLMCall rows - no DB access,
    so this is trivially unit-testable without a session."""
    by_call_type: Dict[str, Dict[str, Any]] = {}
    total_input = total_output = 0
    total_cost: Optional[float] = None

    for row in rows:
        total_input += row.input_tokens or 0
        total_output += row.output_tokens or 0
        if row.cost_usd is not None:
            total_cost = (total_cost or 0.0) + float(row.cost_usd)

        bucket = by_call_type.setdefault(
            row.call_type, {"call_count": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": None},
        )
        bucket["call_count"] += 1
        bucket["input_tokens"] += row.input_tokens or 0
        bucket["output_tokens"] += row.output_tokens or 0
        if row.cost_usd is not None:
            bucket["cost_usd"] = (bucket["cost_usd"] or 0.0) + float(row.cost_usd)

    return SpendSummary(
        call_count=len(rows),
        input_tokens=total_input,
        output_tokens=total_output,
        cost_usd=total_cost,
        by_call_type=[
            CallTypeBreakdown(call_type=call_type, **fields) for call_type, fields in sorted(by_call_type.items())
        ],
    )


def get_aggregate_spend(db: Session, *, since: Optional[datetime] = None) -> SpendSummary:
    """Spend across every traced LLM call, optionally restricted to
    calls made at or after `since`."""
    query = db.query(LLMCall)
    if since is not None:
        query = query.filter(LLMCall.created_at >= since)
    return _summarize(query.all())


def get_session_spend(db: Session, session_id: UUID) -> SpendSummary:
    """Spend across every traced LLM call made while answering any
    question in `session_id`. A session with no traced calls returns an
    all-zero SpendSummary, not an error."""
    request_ids = [
        row[0] for row in
        db.query(ChatMessage.request_id)
        .filter(ChatMessage.session_id == session_id, ChatMessage.request_id.isnot(None))
        .distinct()
        .all()
    ]
    if not request_ids:
        return _summarize([])
    rows = db.query(LLMCall).filter(LLMCall.request_id.in_(request_ids)).all()
    return _summarize(rows)


def get_top_sessions_by_spend(db: Session, limit: int = 10) -> List[Dict[str, Any]]:
    """The `limit` sessions with the highest total traced LLM cost,
    highest first. Joins llm_calls against a *distinct*
    (request_id, session_id) subquery rather than the raw chat_messages
    table, since one turn writes 2 chat_messages rows (user + assistant)
    sharing a request_id, and joining directly would double-count."""
    request_sessions = (
        db.query(ChatMessage.request_id, ChatMessage.session_id)
        .filter(ChatMessage.request_id.isnot(None))
        .distinct()
        .subquery()
    )
    rows = (
        db.query(
            request_sessions.c.session_id,
            func.sum(LLMCall.cost_usd).label("cost_usd"),
            func.sum(LLMCall.input_tokens).label("input_tokens"),
            func.sum(LLMCall.output_tokens).label("output_tokens"),
            func.count(LLMCall.id).label("call_count"),
        )
        .join(LLMCall, LLMCall.request_id == request_sessions.c.request_id)
        .group_by(request_sessions.c.session_id)
        .order_by(func.sum(LLMCall.cost_usd).desc().nullslast())
        .limit(limit)
        .all()
    )
    return [
        {
            "session_id": str(row.session_id),
            "cost_usd": float(row.cost_usd) if row.cost_usd is not None else None,
            "input_tokens": int(row.input_tokens or 0),
            "output_tokens": int(row.output_tokens or 0),
            "call_count": row.call_count,
        }
        for row in rows
    ]


def get_request_trace(db: Session, request_id: str) -> RequestTrace:
    """Every LLM call and tool call made while answering one /chat
    request, oldest first. An unknown request_id returns empty lists,
    not an error - the caller decides whether that's a 404."""
    llm_calls = (
        db.query(LLMCall).filter(LLMCall.request_id == request_id).order_by(LLMCall.created_at).all()
    )
    tool_calls = (
        db.query(ToolCall).filter(ToolCall.request_id == request_id).order_by(ToolCall.created_at).all()
    )
    return RequestTrace(
        request_id=request_id,
        llm_calls=[
            LLMCallDetail(
                call_type=c.call_type, model=c.model, input_tokens=c.input_tokens, output_tokens=c.output_tokens,
                cost_usd=float(c.cost_usd) if c.cost_usd is not None else None,
                latency_ms=c.latency_ms, created_at=c.created_at,
            )
            for c in llm_calls
        ],
        tool_calls=[
            ToolCallDetail(
                tool_name=c.tool_name, tool_input=c.tool_input or {}, success=c.success,
                error=c.error, sources=c.sources or [], created_at=c.created_at,
            )
            for c in tool_calls
        ],
    )
