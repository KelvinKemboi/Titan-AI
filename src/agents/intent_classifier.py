"""
Intent classification: a cheap model call that classifies each incoming
chat message before the Chat/Agent Service decides which tools to
foreground.
"""
import time
from typing import Literal, Optional

import anthropic
from sqlalchemy.orm import Session

from src.observability.tracing import record_llm_call

# Haiku-class model
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
CLASSIFIER_MAX_TOKENS = 20

Intent = Literal["structured", "qualitative", "comparison"]
INTENTS = ("structured", "qualitative", "comparison")

_SYSTEM_PROMPT = """Classify an investment-research question into exactly one category, \
based on what kind of data would answer it - not on whether you personally know the answer:

- structured: about ONE named ticker's own factor scores, rating, or ranking (e.g. \
"why is AAPL rated a BUY", "explain MSFT's score", "what's NVDA's biggest risk factor").
- comparison: TWO OR MORE named tickers being compared or ranked against each other \
(e.g. "is AMD or INTC stronger", "compare MSFT and GOOGL", "why is X ranked above Y").
- qualitative: no single ticker named, or a subjective/thematic question only answerable \
from prose commentary rather than a number (e.g. "which companies have a deep competitive \
moat", "find high-quality compounders", "what looks undervalued right now").

Call classify_intent with your answer."""

_CLASSIFY_INTENT_TOOL = {
    "name": "classify_intent",
    "description": "Records the classified intent for one investment-research question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(INTENTS),
            },
        },
        "required": ["intent"],
    },
}

# Injected into chat_service's system prompt as a routing hint.
INTENT_HINTS = {
    "structured": (
        "This question looks like it's about one named ticker's own factor scores - "
        "prefer get_factor_scores if exactly one ticker is named."
    ),
    "comparison": (
        "This question looks like a comparison between tickers - prefer compare_tickers "
        "if 2 or more tickers are named."
    ),
    "qualitative": (
        "This question looks qualitative/thematic with no single ticker to look up - "
        "prefer search_memos."
    ),
}


def classify_intent(
    question: str,
    *,
    client: Optional[anthropic.Anthropic] = None,
    db: Optional[Session] = None,
    request_id: Optional[str] = None,
) -> Optional[Intent]:
    """
    Classifies `question` into one of Intent's three values via a forced
    tool call, so the model returns a value constrained by the tool
    schema's enum rather than free text. `db`/`request_id` are optional
    tracing hooks - omit them and the call simply isn't traced, so
    existing callers keep working unchanged.
    """
    client = client or anthropic.Anthropic()
    start = time.monotonic()
    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            tools=[_CLASSIFY_INTENT_TOOL],
            tool_choice={"type": "tool", "name": "classify_intent"},
            messages=[{"role": "user", "content": question}],
        )
    except Exception:
        return None
    latency_ms = int((time.monotonic() - start) * 1000)

    if db is not None and request_id is not None:
        record_llm_call(
            db, request_id=request_id, call_type="intent_classification",
            model=CLASSIFIER_MODEL, response=response, latency_ms=latency_ms,
        )

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        return None

    intent = tool_use_block.input.get("intent")
    return intent if intent in INTENTS else None
