"""
Explicit entity tracking (technical-design.md #4, Conversation Memory):
tracks which tickers and factors have come up across a session's
chat_messages, beyond what the raw last-K-turns window gives for free, so
ellipsis/pronoun follow-ups ("what about its momentum?", "how does it
compare to AMD?") resolve reliably instead of depending on the model
re-deriving context from a stack of prior turns and tool-call JSON blobs.
"""
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from src.data.models import ChatMessage
from titan.config import WEIGHTS

logger = logging.getLogger(__name__)

FACTOR_NAMES = list(WEIGHTS.keys())  # ['Value', 'Momentum', 'Quality', 'Solvency', 'Volatility']


class EntityState(BaseModel):
    """
    Session-scoped entity memory: tickers/factors mentioned so far, oldest
    first, with the most recent one (for pronoun resolution) pulled out as
    its own field. A flat, plain-data model - not a class with computed
    properties - so a whole state dump is one `model_dump_json()` call,
    satisfying "inspectable/loggable for debugging incorrect resolutions"
    directly rather than needing extra tooling to read it.
    """

    tickers_mentioned: List[str] = Field(default_factory=list)
    factors_mentioned: List[str] = Field(default_factory=list)
    last_ticker: Optional[str] = None
    last_factor: Optional[str] = None


def _bump(ordered: List[str], value: str) -> None:
    """Moves `value` to the end of `ordered` (most-recently-mentioned last), inserting if new."""
    if value in ordered:
        ordered.remove(value)
    ordered.append(value)


def extract_entities(history: List[ChatMessage]) -> EntityState:
    """
    Scans `history` (oldest-first) for tickers and factors mentioned, in
    the order they came up.

    Tickers come from each assistant turn's `sources` - already-verified
    ticker data every tool call returns (the source-attribution contract,
    technical-design.md #5) - not regex/NLP over free text, so this can't
    mis-extract a ticker Titan never actually looked up. Factors are
    matched by name (Value, Momentum, Quality, Solvency, Volatility)
    against each turn's text, case-insensitively, since that's a small
    closed vocabulary (titan/config.py's WEIGHTS) - no NLP needed.
    """
    tickers: List[str] = []
    factors: List[str] = []

    for message in history:
        if message.role == "assistant" and message.sources:
            for source in message.sources:
                ticker = source.get("ticker")
                if ticker:
                    _bump(tickers, ticker)

        content_lower = message.content.lower()
        for factor in FACTOR_NAMES:
            if factor.lower() in content_lower:
                _bump(factors, factor)

    return EntityState(
        tickers_mentioned=tickers,
        factors_mentioned=factors,
        last_ticker=tickers[-1] if tickers else None,
        last_factor=factors[-1] if factors else None,
    )


def entity_hint(state: EntityState) -> Optional[str]:
    """
    A short natural-language hint for the system prompt, grounding
    pronoun/ellipsis resolution ("it", "its", "that") in the
    most-recently-discussed ticker/factor. Returns None if no ticker has
    been discussed yet this session - nothing to resolve against, and an
    empty hint would just be noise in the prompt.
    """
    if state.last_ticker is None:
        return None

    hint = f"Most recently discussed ticker this session: {state.last_ticker}."
    if state.last_factor is not None:
        hint += f" Most recently discussed factor: {state.last_factor}."
    hint += (
        " If the user's question uses a pronoun ('it', 'its', 'that') or is otherwise "
        "ambiguous about which ticker/factor it refers to, resolve it against this context."
    )
    return hint
