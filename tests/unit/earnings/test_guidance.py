from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy.dialects import postgresql

from src.data.models import EarningsInsight
from src.earnings import chunking as chunking_module
from src.earnings import guidance as guidance_module
from src.earnings.guidance import extract_guidance, generate_guidance

NO_GUIDANCE_TRANSCRIPT = (
    "Suhasini Chandramouli: Good afternoon, welcome to the call.\n\n"
    "Tim Cook: Thanks everyone for joining us today.\n"
)

GUIDANCE_TRANSCRIPT = (
    "Suhasini Chandramouli: Good afternoon, welcome to the call.\n\n"
    "Phil Sawarynski: Relative to our initial fiscal 2025 guidance we provided in December, "
    "we have raised our revenue and EPS guidance.\n"
)


def _tool_use_block(direction, quote):
    return SimpleNamespace(
        type="tool_use", id="t1", name="extract_guidance", input={"guidance_direction": direction, "quote": quote},
    )


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _client_returning(content):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=content)
    return client


def _bound_params(stmt):
    return stmt.compile(dialect=postgresql.dialect()).params


# pre-filter: nothing guidance-related -> none_given, no model call
def test_no_guidance_mentions_returns_none_given_without_calling_the_model():
    client = MagicMock()

    direction, quote = extract_guidance(NO_GUIDANCE_TRANSCRIPT, client=client)

    assert (direction, quote) == ("none_given", None)
    client.messages.create.assert_not_called()


def test_only_chunks_mentioning_guidance_keywords_reach_the_model(monkeypatch):
    # only chunks that mention guidance-related keywords reach the model at all
    monkeypatch.setattr(chunking_module, "MAX_CHUNK_CHARS", 150)
    client = _client_returning([_tool_use_block("raised", "we have raised our revenue and EPS guidance")])

    extract_guidance(GUIDANCE_TRANSCRIPT, client=client)

    sent_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Phil Sawarynski" in sent_content
    assert "Suhasini Chandramouli" not in sent_content  # her turn doesn't mention guidance


# forced structured extraction 
def test_extracts_direction_and_quote_from_the_forced_tool_call():
    client = _client_returning([_tool_use_block("raised", "we have raised our revenue and EPS guidance")])

    direction, quote = extract_guidance(GUIDANCE_TRANSCRIPT, client=client)

    assert direction == "raised"
    assert quote == "we have raised our revenue and EPS guidance"


def test_uses_the_sonnet_class_model_and_forces_the_extract_guidance_tool():
    client = _client_returning([_tool_use_block("maintained", "we are reaffirming guidance")])

    extract_guidance(GUIDANCE_TRANSCRIPT, client=client)

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == guidance_module.GUIDANCE_MODEL
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "extract_guidance"}


def test_empty_quote_is_normalized_to_none():
    client = _client_returning([_tool_use_block("none_given", "")])

    _, quote = extract_guidance(GUIDANCE_TRANSCRIPT, client=client)

    assert quote is None


# graceful degradation for a non-compliant model response 
def test_no_tool_use_block_falls_back_to_unclear():
    client = _client_returning([_text_block("I'm not sure how to classify this.")])

    direction, quote = extract_guidance(GUIDANCE_TRANSCRIPT, client=client)

    assert (direction, quote) == ("unclear", None)


def test_an_invalid_direction_value_falls_back_to_unclear():
    client = _client_returning([_tool_use_block("skyrocketing", "some quote")])

    direction, quote = extract_guidance(GUIDANCE_TRANSCRIPT, client=client)

    assert (direction, quote) == ("unclear", None)


# generate_guidance: persistence 
def test_generate_guidance_upserts_on_transcript_id(monkeypatch):
    monkeypatch.setattr(guidance_module, "extract_guidance", MagicMock(return_value=("raised", "a quote")))
    db = MagicMock()
    db.execute.return_value.scalar_one.return_value = 9
    persisted_row = object()
    db.get.return_value = persisted_row

    result = generate_guidance(db, transcript_id=9, raw_text="irrelevant - extract_guidance is mocked")

    assert result is persisted_row
    db.get.assert_called_once_with(EarningsInsight, 9)

    stmt = db.execute.call_args.args[0]
    params = _bound_params(stmt)
    assert params["transcript_id"] == 9
    assert params["guidance_direction"] == "raised"
    assert params["guidance_quote"] == "a quote"
    assert set(stmt._post_values_clause.inferred_target_elements) == {"transcript_id"}


def test_generate_guidance_swallows_a_model_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        guidance_module, "extract_guidance", MagicMock(side_effect=RuntimeError("model unavailable"))
    )
    db = MagicMock()

    with caplog.at_level("ERROR", logger="src.earnings.guidance"):
        result = generate_guidance(db, transcript_id=3, raw_text="irrelevant")  # must not raise

    assert result is None
    db.execute.assert_not_called()
    assert any("3" in record.message for record in caplog.records)
