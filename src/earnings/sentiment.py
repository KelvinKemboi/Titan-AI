"""
Sentiment Analysis (technical-design.md §10): rubric-based LLM scoring of
management's tone during the Q&A portion of an earnings call - not a
generic sentiment classifier, since financial tone != general sentiment
("we're seeing headwinds" reads negative with no negative-sounding words).
Runs at ingestion time, wired into src/earnings/ingestion.py:ingest_transcript
right after guidance extraction.

The rubric below is the whole prompt - nothing about how -1/0/+1 are
defined lives anywhere else, so it's auditable by reading this file, not
a black box tuned by trial and error against hidden examples.
"""
import logging
from typing import Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

import anthropic

from src.data.models import EarningsInsight
from src.earnings.chunking import chunk_transcript

logger = logging.getLogger(__name__)

# Sonnet-class model
SENTIMENT_MODEL = "claude-sonnet-5"
SENTIMENT_MAX_TOKENS = 300

# Scoring rubric 
#  -1.0  Clearly negative: management explicitly describes results, a
#        program, or the outlook as disappointing, weak, or missing
#        expectations; language is defensive, evasive, or apologetic when
#        pressed. Example anchor: "the quarter was disappointing here."
#  -0.5  Mildly negative: acknowledges real headwinds/softness/pressure
#        without alarm - measured concern, not crisis language.
#   0.0  Neutral: factual, matter-of-fact tone; answers the question
#        directly with data, no clear positive or negative lean, or a
#        genuine even balance of good and bad points.
#  +0.5  Mildly positive: confident but measured - describes results as
#        solid/in line/on track, modest optimism without hard superlatives.
#  +1.0  Clearly positive: enthusiastic, describes results/outlook as
#        strong, record-setting, or ahead of expectations, with no
#        material hedging. Example anchor: unequivocally affirming
#        sustained growth across multiple demand drivers.
_SYSTEM_PROMPT = """Score management's tone during the Q&A portion of this earnings call, from -1 \
(clearly negative) to +1 (clearly positive). Score FINANCIAL tone, not general politeness or \
pleasantness - assess confidence, hedging, and framing about the business itself. A courteous \
"thanks for the question" is not positive sentiment; "we're seeing headwinds in that segment" is \
negative even without any negative-sounding words.

Rubric (anchor points - use the full range between them as appropriate):
  -1.0  Clearly negative: management explicitly describes results or outlook as disappointing, \
weak, or missing expectations; defensive, evasive, or apologetic language when pressed.
  -0.5  Mildly negative: acknowledges real headwinds or softness without alarm - measured \
concern, not crisis language.
   0.0  Neutral: factual, matter-of-fact tone with no clear positive or negative lean, or a \
genuine balance of good and bad points.
  +0.5  Mildly positive: confident but measured - results described as solid or on track, modest \
optimism without hard superlatives.
  +1.0  Clearly positive: enthusiastic, results or outlook described as strong or ahead of \
expectations, with no material hedging.

Score based only on management's own words and tone in their answers - not the analyst's framing, \
and not your own judgment of whether the underlying numbers are objectively good or bad. Call \
score_sentiment with a single number in [-1, 1]."""

_SCORE_SENTIMENT_TOOL = {
    "name": "score_sentiment",
    "description": "Records a financial-tone sentiment score for management's Q&A answers on this earnings call.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment_score": {
                "type": "number",
                "minimum": -1,
                "maximum": 1,
                "description": "Rubric-based score from -1 (clearly negative) to 1 (clearly positive).",
            },
        },
        "required": ["sentiment_score"],
    },
}


def score_sentiment(raw_text: str, *, client: Optional[anthropic.Anthropic] = None) -> Optional[float]:
    """
    Scores `raw_text`'s Q&A-chunk tone per the rubric above, returning a
    float in [-1, 1]. Returns None for a transcript with no qna chunks at
    all 
    """
    qna_texts = [c.chunk_text for c in chunk_transcript(raw_text) if c.chunk_type == "qna"]
    if not qna_texts:
        return None

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=SENTIMENT_MODEL,
        max_tokens=SENTIMENT_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        tools=[_SCORE_SENTIMENT_TOOL],
        tool_choice={"type": "tool", "name": "score_sentiment"},
        messages=[{"role": "user", "content": "\n\n".join(qna_texts)}],
    )

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        return None

    score = tool_use_block.input.get("sentiment_score")
    if not isinstance(score, (int, float)):
        return None
    return max(-1.0, min(1.0, float(score)))


def generate_sentiment(
    db: Session, transcript_id: int, raw_text: str, *, client: Optional[anthropic.Anthropic] = None,
) -> Optional[EarningsInsight]:
    """
    Scores and persists `transcript_id`'s earnings_insights.sentiment_score.
    """
    try:
        score = score_sentiment(raw_text, client=client)
    except Exception:
        logger.exception(
            "Sentiment scoring failed for transcript_id=%s; transcript is still persisted", transcript_id,
        )
        return None

    if score is None:
        logger.info("Nothing to score for transcript_id=%s (no qna chunks)", transcript_id)
        return None

    stmt = (
        pg_insert(EarningsInsight) 
        .values(transcript_id=transcript_id, sentiment_score=score)
        .on_conflict_do_update(index_elements=["transcript_id"], set_={"sentiment_score": score})
        .returning(EarningsInsight.id)
    )
    insight_id = db.execute(stmt).scalar_one()
    return db.get(EarningsInsight, insight_id)
