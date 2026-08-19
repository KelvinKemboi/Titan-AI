"""
Reusable text-embedding helper backing this milestone's pgvector setup and
Phase 2's earnings-chunk ingestion (docs/architecture.md #2, #4, #6).

Model/dimension choice: Anthropic has no embeddings endpoint of its own and
names Voyage AI as its recommended embeddings partner, so it's the natural
pairing for a chat stack already built on `anthropic` (src/agents/chat_service.py).
`voyage-large-2` is pinned specifically (over newer voyage-3.x models) because
it natively outputs EMBEDDING_DIMENSION=1536 floats with no `output_dimension`
override - matching `vector(1536)` on `earnings_chunks.embedding`
(architecture.md #4) exactly. If that column width ever changes, this
constant and EMBEDDING_DIMENSION must change together.
"""
from typing import List, Optional

import voyageai

EMBEDDING_MODEL = "voyage-large-2"
EMBEDDING_DIMENSION = 1536

# Hard cap the Voyage API enforces per embed request; larger inputs must be
# split into multiple calls.
_BATCH_SIZE = voyageai.VOYAGE_EMBED_BATCH_SIZE

# Retries (exponential backoff with jitter) on rate limits/timeouts/transient
# 5xxs are handled by voyageai.Client itself - this just sets how many.
_MAX_RETRIES = 5


def embed_texts(
    texts: List[str],
    *,
    input_type: Optional[str] = "document",
    client: Optional[voyageai.Client] = None,
) -> List[List[float]]:
    """
    Embeds `texts` with EMBEDDING_MODEL, transparently batching in chunks of
    up to `_BATCH_SIZE` (the Voyage API's per-request cap) so callers can
    pass an arbitrarily long list - e.g. every chunk of one earnings
    transcript - without hand-rolling batching themselves. Returns one
    EMBEDDING_DIMENSION-length vector per input text, in the same order as
    `texts`.

    `input_type` should be "document" for text going into the vector store
    (the default - matches Phase 2 ingestion's use case) and "query" when
    embedding a user's search query at retrieval time: Voyage tunes the
    embedding differently for each, and retrieval quality suffers if the two
    are mismatched.
    """
    if not texts:
        return []

    client = client or voyageai.Client(max_retries=_MAX_RETRIES)
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        result = client.embed(batch, model=EMBEDDING_MODEL, input_type=input_type)
        embeddings.extend(result.embeddings)
    return embeddings


def embed_text(
    text: str,
    *,
    input_type: Optional[str] = "document",
    client: Optional[voyageai.Client] = None,
) -> List[float]:
    """Embeds a single string. Thin wrapper over `embed_texts` for the common one-off case."""
    return embed_texts([text], input_type=input_type, client=client)[0]
