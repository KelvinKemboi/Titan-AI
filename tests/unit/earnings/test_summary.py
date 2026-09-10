from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy.dialects import postgresql

from src.data.models import EarningsInsight
from src.earnings import chunking as chunking_module
from src.earnings import summary as summary_module
from src.earnings.summary import generate_summary, summarize_transcript

# 4 short prepared_remarks turns + 1 qna turn.
LONG_TRANSCRIPT = (
    "Suhasini Chandramouli: Welcome to the call.\n\n"
    "Tim Cook: Revenue grew nicely this quarter.\n\n"
    "Luca Maestri: Margins improved too, nicely.\n\n"
    "Suhasini Chandramouli: Operator, may we have the first question, please?\n\n"
    "Operator: Next we hear from Jane Analyst.\n"
)

AAPL_TRANSCRIPT = """\
Suhasini Chandramouli: Good Afternoon, and welcome to the Apple Q2 Fiscal Year 2024 Earnings Conference Call.

Tim Cook: Thank you, Suhasini. Today, Apple is reporting revenue of $90.8 billion and an EPS record of $1.53 for the March quarter.

Suhasini Chandramouli: We ask that you limit yourself to two questions. Operator, may we have the first question, please?

Operator: We will go ahead and take our first question from Mike Ng with Goldman Sachs. Please go ahead.
"""


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _client_returning(*texts):
    client = MagicMock()
    client.messages.create.side_effect = [SimpleNamespace(content=[_text_block(t)]) for t in texts]
    return client


def _bound_params(stmt):
    return stmt.compile(dialect=postgresql.dialect()).params


# if nothing to summarize
def test_no_prepared_remarks_chunks_returns_none_without_calling_the_model():
    client = MagicMock()

    result = summarize_transcript("", client=client)

    assert result is None
    client.messages.create.assert_not_called()


# single-pass summarization for a transcript that fits in one call 
def test_short_transcript_does_a_single_summarization_call():
    client = _client_returning("AAPL beat on revenue and EPS.")

    result = summarize_transcript(AAPL_TRANSCRIPT, client=client)

    assert result == "AAPL beat on revenue and EPS."
    assert client.messages.create.call_count == 1
    # only prepared_remarks content reaches the model - not the qna turn
    sent_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Mike Ng" not in sent_content
    assert "revenue of $90.8 billion" in sent_content


def test_uses_the_sonnet_class_model():
    client = _client_returning("summary")

    summarize_transcript(AAPL_TRANSCRIPT, client=client)

    assert client.messages.create.call_args.kwargs["model"] == summary_module.SUMMARY_MODEL


# chunk-then-reduce for a transcript exceeding the practical input size
def test_long_transcript_is_chunked_then_reduced_not_truncated(monkeypatch):
    monkeypatch.setattr(chunking_module, "MAX_CHUNK_CHARS", 80)
    monkeypatch.setattr(summary_module, "MAX_SUMMARIZATION_INPUT_CHARS", 200)
    client = _client_returning("section summary A", "section summary B", "final combined summary")

    result = summarize_transcript(LONG_TRANSCRIPT, client=client)

    assert result == "final combined summary"
    assert client.messages.create.call_count == 3  # 2 section (map) calls + 1 reduce call

    map_calls = client.messages.create.call_args_list[:-1]
    reduce_call = client.messages.create.call_args_list[-1]
    assert all(c.kwargs["system"] == summary_module._SECTION_SYSTEM_PROMPT for c in map_calls)
    assert reduce_call.kwargs["system"] == summary_module._REDUCE_SYSTEM_PROMPT
    # the reduce call combines the section summaries, not the raw transcript text
    reduce_content = reduce_call.kwargs["messages"][0]["content"]
    assert "section summary A" in reduce_content
    assert "section summary B" in reduce_content


def test_no_single_section_sent_to_the_model_exceeds_the_configured_max(monkeypatch):
    monkeypatch.setattr(chunking_module, "MAX_CHUNK_CHARS", 80)
    monkeypatch.setattr(summary_module, "MAX_SUMMARIZATION_INPUT_CHARS", 200)
    client = _client_returning("A", "B", "final")

    summarize_transcript(LONG_TRANSCRIPT, client=client)

    map_calls = client.messages.create.call_args_list[:-1]
    for call in map_calls:
        assert len(call.kwargs["messages"][0]["content"]) <= 200


# generate_summary upserts on transcript_id, returns the persisted row
def test_generate_summary_upserts_on_transcript_id(monkeypatch):
    monkeypatch.setattr(summary_module, "summarize_transcript", MagicMock(return_value="a summary"))
    db = MagicMock()
    db.execute.return_value.scalar_one.return_value = 7
    persisted_row = object()
    db.get.return_value = persisted_row

    result = generate_summary(db, transcript_id=7, raw_text="irrelevant - summarize_transcript is mocked")

    assert result is persisted_row
    db.get.assert_called_once_with(EarningsInsight, 7)

    stmt = db.execute.call_args.args[0]
    params = _bound_params(stmt)
    assert params["transcript_id"] == 7
    assert params["summary"] == "a summary"
    assert set(stmt._post_values_clause.inferred_target_elements) == {"transcript_id"}


def test_generate_summary_returns_none_and_writes_nothing_when_theres_nothing_to_summarize(monkeypatch):
    monkeypatch.setattr(summary_module, "summarize_transcript", MagicMock(return_value=None))
    db = MagicMock()

    result = generate_summary(db, transcript_id=1, raw_text="")

    assert result is None
    db.execute.assert_not_called()


# swallows a model failure, logs it, and returns None without writing anything
def test_generate_summary_swallows_a_model_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        summary_module, "summarize_transcript", MagicMock(side_effect=RuntimeError("model unavailable"))
    )
    db = MagicMock()

    with caplog.at_level("ERROR", logger="src.earnings.summary"):
        result = generate_summary(db, transcript_id=5, raw_text="irrelevant")  # must not raise

    assert result is None
    db.execute.assert_not_called()
    assert any("5" in record.message for record in caplog.records)
