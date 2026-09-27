"""
Risk extraction: pulls out the risks management discussed during this
call, each with a supporting quote.
"""
import logging
from typing import Dict, List, Optional

import anthropic
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.models import EarningsInsight
from src.earnings.chunking import chunk_transcript

logger = logging.getLogger(__name__)

# Sonnet-class model
RISK_MODEL = "claude-sonnet-5"
RISK_MAX_TOKENS = 1024

# set of key words to look out for
_RISK_KEYWORDS = (
    "risk", "risks", "headwind", "headwinds", "challenge", "challenges", "challenging",
    "uncertain", "uncertainty", "pressure", "competitive", "competition", "regulatory",
    "supply chain", "litigation", "exposure", "concern", "concerns", "volatility",
    "slowdown", "weakness", "tariff", "tariffs", "disruption",
)


def _mentions_risk(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _RISK_KEYWORDS)


_SYSTEM_PROMPT = """Extract the risks management actually discussed during this earnings call excerpt \
- prepared remarks and/or Q&A. This is scoped strictly to what was SAID IN THIS CALL, not a general \
risk-factor summary of the kind you'd find in a 10-K filing:

- Only include a risk if management (not just the analyst asking a question) explicitly raised or \
acknowledged it somewhere in this excerpt.
- Do NOT include generic risks that would apply to any company in this industry just because they \
seem obvious or likely - if it wasn't actually said in this excerpt, leave it out.
- Do NOT include a risk an analyst merely asked about if management's answer didn't itself \
acknowledge or engage with it as a real risk.

For each risk found, state it in your own words in 5-15 words, and include a short quote copied \
verbatim from the excerpt that supports it. If management didn't discuss any specific risks in this \
excerpt, call extract_risks with an empty list - do not invent one to avoid an empty result."""

_EXTRACT_RISKS_TOOL = {
    "name": "extract_risks",
    "description": "Records the risks management explicitly discussed during this earnings call, each with a supporting quote.",
    "input_schema": {
        "type": "object",
        "properties": {
            "risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "risk": {
                            "type": "string",
                            "description": "A short (5-15 word) statement of the risk, in your own words.",
                        },
                        "quote": {
                            "type": "string",
                            "description": "A short quote copied verbatim from the excerpt supporting this risk.",
                        },
                    },
                    "required": ["risk", "quote"],
                },
            },
        },
        "required": ["risks"],
    },
}


def extract_risks(raw_text: str, *, client: Optional[anthropic.Anthropic] = None) -> List[Dict[str, str]]:
    """Extracts the risks management discussed in `raw_text`, returning a
    list of {"risk": ..., "quote": ...} dicts - [] if none were
    discussed, or nothing mentions risk-related language at all."""
    relevant_texts = [c.chunk_text for c in chunk_transcript(raw_text) if _mentions_risk(c.chunk_text)]
    if not relevant_texts:
        return []

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=RISK_MODEL,
        max_tokens=RISK_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        tools=[_EXTRACT_RISKS_TOOL],
        tool_choice={"type": "tool", "name": "extract_risks"},
        messages=[{"role": "user", "content": "\n\n".join(relevant_texts)}],
    )

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        return []

    raw_risks = tool_use_block.input.get("risks")
    if not isinstance(raw_risks, list):
        return []

    return [
        {"risk": item["risk"], "quote": item["quote"]}
        for item in raw_risks
        if isinstance(item, dict) and isinstance(item.get("risk"), str) and isinstance(item.get("quote"), str)
    ]


def generate_risks(
    db: Session, transcript_id: int, raw_text: str, *, client: Optional[anthropic.Anthropic] = None,
) -> Optional[EarningsInsight]:
    """Extracts and persists `transcript_id`'s risks, upserting on the
    table's transcript_id index."""
    try:
        risks = extract_risks(raw_text, client=client)
    except Exception:
        logger.exception(
            "Risk extraction failed for transcript_id=%s; transcript is still persisted", transcript_id,
        )
        return None

    stmt = (
        pg_insert(EarningsInsight)
        .values(transcript_id=transcript_id, risks=risks)
        .on_conflict_do_update(index_elements=["transcript_id"], set_={"risks": risks})
        .returning(EarningsInsight.id)
    )
    insight_id = db.execute(stmt).scalar_one()
    return db.get(EarningsInsight, insight_id)
