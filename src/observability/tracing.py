"""
LLM/tool call tracing: the Chat/Agent Service
calls multiple models (intent classifier, main generation) across
multiple tools per turn - this module is the single place that persists
what each of those calls actually did, so a bad answer or a cost spike
is debuggable after the fact instead of only visible in transient logs.

Every call recorded here shares a `request_id` - a fresh UUID minted
once per `answer_question()` invocation (src/agents/chat_service.py) -
which is also stored on that turn's `chat_messages` row. That's the
join: given a bad answer, `chat_messages.request_id` -> every LLMCall
and ToolCall row sharing it is the complete "what did we ask, what did
each tool return, what did each model call cost" trace for that turn.

Recording never raises: a tracing failure (a session that hasn't
committed the new tables' migration yet, a locked table, etc.) must not
break the actual chat response it's trying to observe. Logged and
swallowed, matching this codebase's established "an ancillary write
must not fail the primary operation" convention (e.g.
src/analytics/memo_indexing.py's embedding-failure handling).
"""
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.data.models import LLMCall, ToolCall

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output) - Anthropic's current first-party
# pricing (per the claude-api skill, cached 2026-06-24). Best-effort
# figures for cost-TREND visibility (a spike is a spike regardless of a
# few percent of pricing drift), not a billing-accurate ledger - confirm
# against Anthropic's pricing page before treating this as authoritative.
# claude-haiku-4-5-20251001 is this codebase's dated snapshot of the
# undated claude-haiku-4-5 the current pricing table lists - priced the
# same; re-check if that snapshot is ever repriced independently of it.
MODEL_PRICING_PER_MILLION_TOKENS: Dict[str, Dict[str, float]] = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}


def estimate_cost_usd(model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """
    Estimated USD cost for one call, or None if `model` isn't in
    MODEL_PRICING_PER_MILLION_TOKENS or either token count is unknown -
    None (not 0.0) so "unpriced" is never silently indistinguishable
    from "free" in an aggregate sum.
    """
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


def record_llm_call(
    db: Optional[Session],
    *,
    request_id: str,
    call_type: str,
    model: str,
    response: Any,
    latency_ms: Optional[int] = None,
) -> None:
    """
    Persists one LLMCall row from a completed `client.messages.create`
    `response`. `db` is Optional so every call site (including ones that
    don't have a DB session in scope, e.g. classify_intent's direct
    callers/tests/eval scripts) can opt out of tracing by simply passing
    None, rather than every caller needing its own if-tracing-enabled
    branch. Token counts default to None (not persisted as 0) if
    `response` has no `.usage` - a mocked test response, or a future SDK
    response shape this hasn't been updated for.
    """
    if db is None:
        return

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None

    try:
        db.add(LLMCall(
            request_id=request_id,
            call_type=call_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost_usd(model, input_tokens, output_tokens),
            latency_ms=latency_ms,
        ))
    except Exception:
        logger.exception("Failed to record LLM call trace for request_id=%s call_type=%s", request_id, call_type)


def record_tool_call(
    db: Optional[Session],
    *,
    request_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    success: bool,
    error: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Persists one ToolCall row - the per-call input/output granularity
    behind "trace a bad answer back to which tool call(s) produced the
    incorrect grounding data" (independent of whatever the final
    aggregated ChatAnswer.sources ends up containing)."""
    if db is None:
        return

    try:
        db.add(ToolCall(
            request_id=request_id,
            tool_name=tool_name,
            tool_input=tool_input,
            success=success,
            error=error,
            sources=sources or [],
        ))
    except Exception:
        logger.exception("Failed to record tool call trace for request_id=%s tool_name=%s", request_id, tool_name)
