"""
Explicit entity tracking: tracks which tickers and factors have come up
across a session's chat_messages, so pronoun/ellipsis follow-ups ("what
about its momentum?", "how does it compare to AMD?") resolve reliably
instead of depending on the model re-deriving context from raw history.
"""
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from src.data.models import ChatMessage
from titan.config import WEIGHTS

logger = logging.getLogger(__name__)

FACTOR_NAMES = list(WEIGHTS.keys())  # ['Value', 'Momentum', 'Quality', 'Solvency', 'Volatility']


class EntityState(BaseModel):
    """Session-scoped entity memory: tickers/factors mentioned so far,
    oldest first, with the most recent of each pulled out for pronoun
    resolution. Flat and plain-data so the whole state can be logged
    with a single `model_dump_json()` call."""

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
    Scans `history` (oldest first) for tickers and factors mentioned, in
    the order they came up. Tickers come from each assistant turn's
    already-verified `sources`, not regex/NLP over free text, so a
    ticker Titan never actually looked up can't be mis-extracted.
    Factors are matched by name against the small closed vocabulary in
    titan/config.py's WEIGHTS.
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
    been discussed yet this session.
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
