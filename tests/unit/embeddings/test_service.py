from types import SimpleNamespace
from unittest.mock import MagicMock

from src.embeddings.service import (
    _BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    embed_text,
    embed_texts,
)

def _vector(seed):
    return [float(seed)] * EMBEDDING_DIMENSION


def _client_returning(*vector_batches):
    """A mock client whose .embed() returns the next batch of vectors on each call."""
    client = MagicMock()
    client.embed.side_effect = [SimpleNamespace(embeddings=list(batch)) for batch in vector_batches]
    return client


# one input -> one call, one vector back, in the pinned model/dimension
def test_embed_texts_returns_one_vector_per_input():
    client = _client_returning([_vector(1), _vector(2)])

    result = embed_texts(["hello", "world"], client=client)

    assert result == [_vector(1), _vector(2)]
    client.embed.assert_called_once_with(["hello", "world"], model=EMBEDDING_MODEL, input_type="document")


# inputs beyond the Voyage API's per-request cap (_BATCH_SIZE) must be split
# across multiple calls, not sent in one oversized request, with results
# concatenated back in the original order
def test_embed_texts_batches_large_input_across_multiple_calls():
    texts = [f"chunk-{i}" for i in range(_BATCH_SIZE + 1)]
    first_batch_vectors = [_vector(i) for i in range(_BATCH_SIZE)]
    second_batch_vectors = [_vector(_BATCH_SIZE)]
    client = _client_returning(first_batch_vectors, second_batch_vectors)

    result = embed_texts(texts, client=client)

    assert client.embed.call_count == 2
    first_call_texts = client.embed.call_args_list[0].args[0]
    second_call_texts = client.embed.call_args_list[1].args[0]
    assert len(first_call_texts) == _BATCH_SIZE
    assert second_call_texts == [f"chunk-{_BATCH_SIZE}"]
    assert result == first_batch_vectors + second_batch_vectors


# an empty input list is a no-op - notably, must not require a client/API key
# to be configured, since there's nothing to call the API for
def test_embed_texts_empty_list_returns_empty_without_a_client():
    assert embed_texts([]) == []


# embed_text is a thin single-string wrapper over embed_texts
def test_embed_text_returns_single_vector():
    client = _client_returning([_vector(7)])

    result = embed_text("solo", client=client)

    assert result == _vector(7)
    client.embed.assert_called_once_with(["solo"], model=EMBEDDING_MODEL, input_type="document")


# input_type controls document- vs query-time embedding tuning and must reach the API call
def test_input_type_is_passed_through():
    client = _client_returning([_vector(1)])

    embed_texts(["what's AAPL's momentum score?"], input_type="query", client=client)

    client.embed.assert_called_once_with(
        ["what's AAPL's momentum score?"], model=EMBEDDING_MODEL, input_type="query"
    )


# pins the model/dimension pair so a future edit to one without the other
# (see module docstring) fails loudly instead of silently drifting from the
# vector(1536) column width in earnings_chunks
def test_model_and_dimension_are_pinned_together():
    assert EMBEDDING_MODEL == "voyage-large-2"
    assert EMBEDDING_DIMENSION == 1536
