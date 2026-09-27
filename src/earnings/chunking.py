"""
Splits one earnings_transcripts.raw_text into earnings_chunks-shaped
rows: chunked by speaker turn and tagged prepared_remarks/qna, never by
a fixed token window, so "who said this" and which section it came from
both survive into the embedded chunk.

Format assumption: each speaker turn is its own paragraph, starting
with a short capitalized name followed by ": " (e.g. "Tim Cook: Thank
you, Suhasini..."). A transcript that doesn't follow this convention,
or never explicitly announces the Q&A, degrades to a single
prepared_remarks section rather than misclassifying content as qna.
"""
import re
from typing import List, Tuple

from pydantic import BaseModel

# A rough proxy for token count, sized for retrieval granularity rather
# than a context-window limit - small enough that a similarity search
# returns one focused exchange, not a whole multi-topic monologue.
MAX_CHUNK_CHARS = 2000

# Speaker-turn label: 1-4 capitalized words (allows "O'Brien", "St. Clair",
# "Jean-Michel"), immediately followed by ": " - matches "Tim Cook: ",
# "Operator: ", "Suhasini Chandramouli: ". Anchored to the start of a line
# since every real transcript observed puts one turn per paragraph.
_SPEAKER_TURN = re.compile(
    r"^([A-Z][a-zA-Z'.-]*(?:\s+[A-Z][a-zA-Z'.-]*){0,3}):\s+", re.MULTILINE,
)

# Phrasings that mark the operator/host actually opening the floor to
# questions right now - whichever turn contains one is the last
# prepared_remarks turn, everything after is qna. Deliberately excludes
# a bare "question-and-answer session" mention, since operators often
# announce that in future tense long before it actually happens.
_QNA_TRANSITION_PATTERN = re.compile(
    r"first question comes from"
    r"|may we have the first question"
    r"|move (?:over |on )?to (?:the )?q\s*&\s*a"
    r"|open(?:ing)? (?:the|up) call (?:up )?to questions"
    r"|open (?:the|up) call for questions"
    r"|begin the question-and-answer session"
    r"|we will now begin the question",
    re.IGNORECASE,
)


class TranscriptChunk(BaseModel):
    """One earnings_chunks-shaped row: chunk_type ('prepared_remarks' |
    'qna') + chunk_text. Everything else earnings_chunks needs
    (transcript_id, embedding) is the caller's job - this is pure text
    processing, no DB/embedding access."""

    chunk_type: str
    chunk_text: str


def _split_into_turns(raw_text: str) -> List[Tuple[str, str]]:
    """Splits `raw_text` into (speaker, turn_text) pairs, `turn_text`
    including the "Speaker: " prefix so speaker attribution survives into
    the chunk. A transcript with no recognizable speaker-turn markers at
    all comes back as a single unlabeled turn, not an error."""
    matches = list(_SPEAKER_TURN.finditer(raw_text))
    if not matches:
        return [("", raw_text.strip())] if raw_text.strip() else []

    turns = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        turns.append((match.group(1), raw_text[start:end].strip()))
    return turns


def _tag_sections(turns: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """Returns (speaker, turn_text, chunk_type) triples: every turn up to
    and including the first one that announces the Q&A is prepared_remarks,
    everything after is qna. No transition found -> everything is
    prepared_remarks (never guess at a qna boundary that isn't there)."""
    transition_index = next(
        (i for i, (_, text) in enumerate(turns) if _QNA_TRANSITION_PATTERN.search(text)), None,
    )
    return [
        (speaker, text, "qna" if transition_index is not None and i > transition_index else "prepared_remarks")
        for i, (speaker, text) in enumerate(turns)
    ]


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """Splits one over-long turn on sentence boundaries, greedily filling
    each piece up to `max_chars` - used only for the rare turn that alone
    exceeds MAX_CHUNK_CHARS, so even a very long monologue stays embeddable."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
        # A single "sentence" longer than max_chars on its own: hard-cut it
        # rather than emit an oversized chunk.
        while len(current) > max_chars:
            pieces.append(current[:max_chars])
            current = current[max_chars:]
    if current:
        pieces.append(current)
    return pieces


def chunk_transcript(raw_text: str) -> List[TranscriptChunk]:
    """
    Splits one transcript's raw text into TranscriptChunk rows: grouped by
    consecutive same-section speaker turns, capped at MAX_CHUNK_CHARS, never
    mixing prepared_remarks and qna in the same chunk. A section change
    always starts a new chunk even if the current one has room left -
    section boundaries take priority over packing chunks as full as possible.
    """
    tagged_turns = _tag_sections(_split_into_turns(raw_text))

    chunks: List[TranscriptChunk] = []
    current_type = None
    current_parts: List[str] = []
    current_len = 0

    def _flush():
        if current_parts:
            chunks.append(TranscriptChunk(chunk_type=current_type, chunk_text="\n\n".join(current_parts)))

    for _, turn_text, chunk_type in tagged_turns:
        if chunk_type != current_type or current_len + len(turn_text) + 2 > MAX_CHUNK_CHARS:
            _flush()
            current_parts, current_len, current_type = [], 0, chunk_type

        if len(turn_text) > MAX_CHUNK_CHARS:
            _flush()
            current_parts, current_len = [], 0
            for piece in _split_long_text(turn_text, MAX_CHUNK_CHARS):
                chunks.append(TranscriptChunk(chunk_type=chunk_type, chunk_text=piece))
            current_type = None  # force a fresh chunk for whatever comes next
            continue

        current_parts.append(turn_text)
        current_len += len(turn_text) + 2

    _flush()
    return chunks
