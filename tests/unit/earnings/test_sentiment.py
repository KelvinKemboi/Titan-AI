from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy.dialects import postgresql

from src.data.models import EarningsInsight
from src.earnings import sentiment as sentiment_module
from src.earnings.sentiment import generate_sentiment, score_sentiment

NO_QNA_TRANSCRIPT = (
    "Suhasini Chandramouli: Good afternoon, welcome to the call.\n\n"
    "Tim Cook: Thanks everyone for joining us today.\n"
)

QNA_TRANSCRIPT = (
    "Suhasini Chandramouli: We'll now take questions. Operator, may we have the first question, please?\n\n"
    "Operator: Our first question comes from an analyst.\n\n"
    "Analyst: How do you feel about the quarter?\n\n"
    "CFO: We're very pleased with these results - a record quarter across every metric.\n"
)


def _tool_use_block(score):
    return SimpleNamespace(type="tool_use", id="t1", name="score_sentiment", input={"sentiment_score": score})


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _client_returning(content):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=content)
    return client


def _bound_params(stmt):
    return stmt.compile(dialect=postgresql.dialect()).params


# nothing to score
def test_no_qna_chunks_returns_none_without_calling_the_model():
    client = MagicMock()

    result = score_sentiment(NO_QNA_TRANSCRIPT, client=client)

    assert result is None
    client.messages.create.assert_not_called()


def test_only_qna_chunks_reach_the_model():
    client = _client_returning([_tool_use_block(0.8)])

    score_sentiment(QNA_TRANSCRIPT, client=client)

    sent_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "record quarter" in sent_content
    assert "Suhasini Chandramouli" not in sent_content  # her turn is prepared_remarks, not qna


# forced structured extraction 
def test_returns_the_models_score():
    client = _client_returning([_tool_use_block(0.8)])

    score = score_sentiment(QNA_TRANSCRIPT, client=client)

    assert score == 0.8


def test_uses_the_sonnet_class_model_and_forces_the_score_sentiment_tool():
    client = _client_returning([_tool_use_block(0.0)])

    score_sentiment(QNA_TRANSCRIPT, client=client)

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == sentiment_module.SENTIMENT_MODEL
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "score_sentiment"}


# clamping / graceful degradation
def test_a_score_above_one_is_clamped():
    client = _client_returning([_tool_use_block(2.5)])

    assert score_sentiment(QNA_TRANSCRIPT, client=client) == 1.0


def test_a_score_below_negative_one_is_clamped():
    client = _client_returning([_tool_use_block(-3)])

    assert score_sentiment(QNA_TRANSCRIPT, client=client) == -1.0


def test_no_tool_use_block_returns_none():
    client = _client_returning([_text_block("I can't determine a score.")])

    assert score_sentiment(QNA_TRANSCRIPT, client=client) is None


def test_a_non_numeric_score_returns_none():
    client = _client_returning([_tool_use_block("very positive")])

    assert score_sentiment(QNA_TRANSCRIPT, client=client) is None


# generate_sentiment: persistence 
def test_generate_sentiment_upserts_on_transcript_id(monkeypatch):
    monkeypatch.setattr(sentiment_module, "score_sentiment", MagicMock(return_value=0.6))
    db = MagicMock()
    db.execute.return_value.scalar_one.return_value = 11
    persisted_row = object()
    db.get.return_value = persisted_row

    result = generate_sentiment(db, transcript_id=11, raw_text="irrelevant - score_sentiment is mocked")

    assert result is persisted_row
    db.get.assert_called_once_with(EarningsInsight, 11)

    stmt = db.execute.call_args.args[0]
    params = _bound_params(stmt)
    assert params["transcript_id"] == 11
    assert params["sentiment_score"] == 0.6
    assert set(stmt._post_values_clause.inferred_target_elements) == {"transcript_id"}


def test_generate_sentiment_returns_none_and_writes_nothing_when_theres_nothing_to_score(monkeypatch):
    monkeypatch.setattr(sentiment_module, "score_sentiment", MagicMock(return_value=None))
    db = MagicMock()

    result = generate_sentiment(db, transcript_id=1, raw_text="")

    assert result is None
    db.execute.assert_not_called()


def test_generate_sentiment_swallows_a_model_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        sentiment_module, "score_sentiment", MagicMock(side_effect=RuntimeError("model unavailable"))
    )
    db = MagicMock()

    with caplog.at_level("ERROR", logger="src.earnings.sentiment"):
        result = generate_sentiment(db, transcript_id=4, raw_text="irrelevant")  # must not raise

    assert result is None
    db.execute.assert_not_called()
    assert any("4" in record.message for record in caplog.records)
