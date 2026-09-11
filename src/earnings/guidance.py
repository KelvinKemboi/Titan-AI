"""
 a targeted, structured extraction pass classifying how a company's forward guidance changed this
call, with a supporting quote
"""
import logging
from typing import Optional, Tuple

import anthropic
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.models import EarningsInsight
from src.earnings.chunking import chunk_transcript

logger = logging.getLogger(__name__)

# Sonnet-class model.
GUIDANCE_MODEL = "claude-sonnet-5"
GUIDANCE_MAX_TOKENS = 300

GUIDANCE_DIRECTIONS = ("raised", "maintained", "lowered", "none_given", "unclear")

# keywords that indicate a chunk of transcript text is likely to contain forward guidance discussion
_GUIDANCE_KEYWORDS = (
    "guidance", "outlook", "forecast", "raise", "raising", "raised",
    "lower", "lowering", "lowered", "cut", "reaffirm", "maintain",
    "full-year", "full year", "next quarter", "fiscal year", "expect",
)


def _mentions_guidance(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _GUIDANCE_KEYWORDS)


_SYSTEM_PROMPT = """Classify how this earnings call's forward guidance changed relative to the \
company's own previously communicated guidance, based only on what management explicitly says - \
not analyst speculation, and not your own judgment about whether the results were good or bad.

- raised: management explicitly increased a forward guidance range or target from what they'd \
previously communicated.
- maintained: management explicitly reaffirmed or left a forward guidance range or target \
unchanged from what they'd previously communicated.
- lowered: management explicitly decreased, cut, or narrowed down a forward guidance range or \
target from what they'd previously communicated.
- none_given: forward guidance is not discussed anywhere in this excerpt.
- unclear: forward guidance is discussed, but the direction is ambiguous, mixed across different \
segments/metrics, or not stated clearly enough to classify with confidence.

Call extract_guidance with your classification and, unless it's none_given, a short quote copied \
verbatim from the excerpt that supports it."""

_EXTRACT_GUIDANCE_TOOL = {
    "name": "extract_guidance",
    "description": "Records the classified forward-guidance direction for this earnings call, with a supporting quote.",
    "input_schema": {
        "type": "object",
        "properties": {
            "guidance_direction": {"type": "string", "enum": list(GUIDANCE_DIRECTIONS)},
            "quote": {
                "type": "string",
                "description": "A short quote copied verbatim from the excerpt supporting this "
                "classification. Empty string only when guidance_direction is none_given.",
            },
        },
        "required": ["guidance_direction", "quote"],
    },
}


def extract_guidance(
    raw_text: str, *, client: Optional[anthropic.Anthropic] = None,
) -> Tuple[str, Optional[str]]:
    """
    Classifies `raw_text`'s forward-guidance direction, returning
    (guidance_direction, quote). `quote` is None for none_given, or
    whenever the model didn't return one.
    """
    client = client or anthropic.Anthropic()
    relevant_texts = [c.chunk_text for c in chunk_transcript(raw_text) if _mentions_guidance(c.chunk_text)]
    if not relevant_texts:
        return "none_given", None

    response = client.messages.create(
        model=GUIDANCE_MODEL,
        max_tokens=GUIDANCE_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        tools=[_EXTRACT_GUIDANCE_TOOL],
        tool_choice={"type": "tool", "name": "extract_guidance"},
        messages=[{"role": "user", "content": "\n\n".join(relevant_texts)}],
    )

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        return "unclear", None

    direction = tool_use_block.input.get("guidance_direction")
    if direction not in GUIDANCE_DIRECTIONS:
        return "unclear", None

    quote = tool_use_block.input.get("quote") or None
    return direction, quote


def generate_guidance(
    db: Session, transcript_id: int, raw_text: str, *, client: Optional[anthropic.Anthropic] = None,
) -> Optional[EarningsInsight]:
    """
    Extracts and persists `transcript_id`'s earnings_insights.guidance_direction
    + guidance_quote. Upserts on the table's own unique index (transcript_id),
    same convention as src/earnings/summary.py:generate_summary for earnings_insights.summary_text
    """
    try:
        direction, quote = extract_guidance(raw_text, client=client)
    except Exception:
        logger.exception(
            "Guidance extraction failed for transcript_id=%s; transcript is still persisted", transcript_id,
        )
        return None

    stmt = (
        pg_insert(EarningsInsight)
        .values(transcript_id=transcript_id, guidance_direction=direction, guidance_quote=quote)
        .on_conflict_do_update(
            index_elements=["transcript_id"],
            set_={"guidance_direction": direction, "guidance_quote": quote},
        )
        .returning(EarningsInsight.id)
    )
    insight_id = db.execute(stmt).scalar_one()
    return db.get(EarningsInsight, insight_id)
