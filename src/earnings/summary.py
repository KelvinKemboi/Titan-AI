"""
Earnings Summary Generation: a single LLM summarization pass over one transcript's prepared-remarks chunks
"""
import logging
from typing import List, Optional

import anthropic
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.models import EarningsInsight
from src.earnings.chunking import chunk_transcript

logger = logging.getLogger(__name__)

# Sonnet-class model
SUMMARY_MODEL = "claude-sonnet-5"
SUMMARY_MAX_TOKENS = 512

# input-size bound for one summarization call 
MAX_SUMMARIZATION_INPUT_CHARS = 12000

_SECTION_SYSTEM_PROMPT = """You summarize excerpts of a company's earnings call prepared remarks \
(management's own narration, not analyst Q&A). In 2-4 sentences, capture the concrete financial \
results, business highlights, and forward-looking commentary in this excerpt - state numbers and \
facts as given, don't editorialize or speculate beyond what's said."""

_REDUCE_SYSTEM_PROMPT = """You are given several summaries of consecutive sections of one \
earnings call's prepared remarks, in order. Combine them into a single cohesive 3-5 sentence \
summary of the entire call's prepared remarks - merge repeated points, preserve the specific \
numbers and facts each section summary states, and don't introduce anything not present in them."""


def _complete(system_prompt: str, user_content: str, *, client: anthropic.Anthropic) -> str:
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=SUMMARY_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _group_into_sections(texts: List[str], max_chars: int) -> List[str]:
    """Packs consecutive chunk texts into sections up to `max_chars` each,
    so the map step summarizes a handful of sections rather than one tiny
    LLM call per chunk (chunk_transcript's chunks are already far smaller
    than a summarization-sized section)."""
    sections: List[str] = []
    current: List[str] = []
    current_len = 0
    for text in texts:
        if current and current_len + len(text) + 2 > max_chars:
            sections.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(text)
        current_len += len(text) + 2
    if current:
        sections.append("\n\n".join(current))
    return sections


def summarize_transcript(raw_text: str, *, client: Optional[anthropic.Anthropic] = None) -> Optional[str]:
    """
    Summarizes `raw_text`'s prepared-remarks chunks (chunk_transcript)
    Returns None for a transcript with no prepared-remarks chunks at
    all (e.g. blank raw_text), without calling the model.

    Chunk-then-reduce: if the concatenated prepared-remarks text fits
    within MAX_SUMMARIZATION_INPUT_CHARS, one call summarizes it directly.
    Otherwise it's grouped into sections, each section is summarized on
    its own (the "map" step), and a final call combines those section
    summaries into one summary (the "reduce" step)
    """
    client = client or anthropic.Anthropic()
    prepared_texts = [c.chunk_text for c in chunk_transcript(raw_text) if c.chunk_type == "prepared_remarks"]
    if not prepared_texts:
        return None

    full_text = "\n\n".join(prepared_texts)
    if len(full_text) <= MAX_SUMMARIZATION_INPUT_CHARS:
        return _complete(_SECTION_SYSTEM_PROMPT, full_text, client=client)

    sections = _group_into_sections(prepared_texts, MAX_SUMMARIZATION_INPUT_CHARS)
    section_summaries = [_complete(_SECTION_SYSTEM_PROMPT, section, client=client) for section in sections]
    return _complete(_REDUCE_SYSTEM_PROMPT, "\n\n".join(section_summaries), client=client)


def generate_summary(
    db: Session, transcript_id: int, raw_text: str, *, client: Optional[anthropic.Anthropic] = None,
) -> Optional[EarningsInsight]:
    """
    Generates and persists `transcript_id`'s earnings_insights.summary.
    Upserts on the table's own unique index (transcript_id) so a later
    extraction pass (guidance/sentiment/risk) can update the summary without overwriting it
    """
    try:
        summary = summarize_transcript(raw_text, client=client)
    except Exception:
        logger.exception("Summary generation failed for transcript_id=%s; transcript is still persisted", transcript_id)
        return None

    if summary is None:
        logger.info("Nothing to summarize for transcript_id=%s (no prepared-remarks chunks)", transcript_id)
        return None

    stmt = (
        pg_insert(EarningsInsight)
        .values(transcript_id=transcript_id, summary=summary)
        .on_conflict_do_update(index_elements=["transcript_id"], set_={"summary": summary})
        .returning(EarningsInsight.id)
    )
    insight_id = db.execute(stmt).scalar_one()
    return db.get(EarningsInsight, insight_id)
