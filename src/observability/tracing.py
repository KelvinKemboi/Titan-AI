"""
LLM/tool call tracing: the Chat/Agent Service calls multiple models
across multiple tools per turn, and this module is the single place
that persists what each call actually did. Every call recorded here
shares a `request_id` - the same id stored on that turn's chat_messages
row - so a bad answer or a cost spike can be traced back to exactly
what was asked, what each tool returned, and what each model call cost.

Recording never raises: a tracing failure must not break the actual
chat response it's observing, so every write here is logged and
swallowed rather than propagated.
"""
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.data.models import LLMCall, ToolCall

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output) - Anthropic's current first-party
# pricing. Best-effort for cost-trend visibility, not a billing-accurate
# ledger; confirm against Anthropic's pricing page before relying on it.
MODEL_PRICING_PER_MILLION_TOKENS: Dict[str, Dict[str, float]] = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}


def estimate_cost_usd(model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Estimated USD cost for one call. Returns None (not 0.0) if `model`
    is unpriced or either token count is unknown, so "unpriced" is never
    silently indistinguishable from "free" in an aggregate sum."""
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
    response. `db` is optional so a caller with no DB session in scope
    can opt out of tracing by passing None. Token counts default to
    None if `response` has no `.usage`, rather than being recorded as 0.
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
    """Persists one ToolCall row: this call's own input and sources,
    independent of whatever the final aggregated answer ends up citing."""
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
