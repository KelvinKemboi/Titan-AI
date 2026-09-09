from unittest.mock import MagicMock

from src.data.models import EarningsChunk
from src.earnings.chunk_indexing import index_transcript_chunks
from src.earnings.chunking import TranscriptChunk


def _chunk(chunk_type, chunk_text):
    return TranscriptChunk(chunk_type=chunk_type, chunk_text=chunk_text)


# no chunks (e.g. blank raw_text) -> nothing to embed or persist
def test_no_chunks_is_a_noop(monkeypatch):
    monkeypatch.setattr("src.earnings.chunk_indexing.chunk_transcript", MagicMock(return_value=[]))
    mock_embed_texts = MagicMock()
    monkeypatch.setattr("src.earnings.chunk_indexing.embed_texts", mock_embed_texts)
    db = MagicMock()

    result = index_transcript_chunks(db, transcript_id=1, raw_text="")

    assert result == []
    mock_embed_texts.assert_not_called()
    db.add_all.assert_not_called()


# every chunk is embedded in one batched call, not one API call per chunk
def test_indexes_one_row_per_chunk_with_a_single_batched_embed_call(monkeypatch):
    chunks = [_chunk("prepared_remarks", "Tim Cook: revenue was strong."), _chunk("qna", "Analyst: any color on guidance?")]
    monkeypatch.setattr("src.earnings.chunk_indexing.chunk_transcript", MagicMock(return_value=chunks))
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    mock_embed_texts = MagicMock(return_value=vectors)
    monkeypatch.setattr("src.earnings.chunk_indexing.embed_texts", mock_embed_texts)
    db = MagicMock()

    result = index_transcript_chunks(db, transcript_id=42, raw_text="irrelevant - chunk_transcript is mocked")

    mock_embed_texts.assert_called_once_with(
        ["Tim Cook: revenue was strong.", "Analyst: any color on guidance?"]
    )
    assert len(result) == 2
    assert all(isinstance(row, EarningsChunk) for row in result)

    assert result[0].transcript_id == 42
    assert result[0].chunk_type == "prepared_remarks"
    assert result[0].chunk_text == "Tim Cook: revenue was strong."
    assert result[0].embedding == [1.0, 2.0]

    assert result[1].transcript_id == 42
    assert result[1].chunk_type == "qna"
    assert result[1].chunk_text == "Analyst: any color on guidance?"
    assert result[1].embedding == [3.0, 4.0]

    db.add_all.assert_called_once_with(result)
    db.flush.assert_called_once()
    db.commit.assert_not_called()


# an embeddings-API failure must not raise - the transcript row must still
# count as ingested even if chunk indexing can't reach Voyage this run
def test_embedding_failure_is_logged_and_swallowed(monkeypatch):
    monkeypatch.setattr(
        "src.earnings.chunk_indexing.chunk_transcript",
        MagicMock(return_value=[_chunk("prepared_remarks", "some text")]),
    )
    monkeypatch.setattr(
        "src.earnings.chunk_indexing.embed_texts", MagicMock(side_effect=RuntimeError("Voyage API down"))
    )
    db = MagicMock()

    result = index_transcript_chunks(db, transcript_id=1, raw_text="irrelevant")  # must not raise

    assert result == []
    db.add_all.assert_not_called()
