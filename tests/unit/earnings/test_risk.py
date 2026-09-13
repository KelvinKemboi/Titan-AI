from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy.dialects import postgresql

from src.data.models import EarningsInsight
from src.earnings import chunking as chunking_module
from src.earnings import risk as risk_module
from src.earnings.risk import extract_risks, generate_risks

NO_RISK_TRANSCRIPT = (
    "Suhasini Chandramouli: Good afternoon, welcome to the call.\n\n"
    "Tim Cook: Thanks everyone for joining us today.\n"
)

RISK_TRANSCRIPT = (
    "Suhasini Chandramouli: Good afternoon, welcome to the call.\n\n"
    "Tim Cook: We continue to see meaningful foreign currency headwinds and ongoing supply chain "
    "constraints in certain component categories.\n"
)


def _tool_use_block(risks):
    return SimpleNamespace(type="tool_use", id="t1", name="extract_risks", input={"risks": risks})


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _client_returning(content):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=content)
    return client


def _bound_params(stmt):
    return stmt.compile(dialect=postgresql.dialect()).params


# nothing risk-related -> [], no model call
def test_no_risk_mentions_returns_empty_list_without_calling_the_model():
    client = MagicMock()

    result = extract_risks(NO_RISK_TRANSCRIPT, client=client)

    assert result == []
    client.messages.create.assert_not_called()


def test_only_chunks_mentioning_risk_keywords_reach_the_model(monkeypatch):
    # Force each speaker turn into its own chunk (they'd otherwise be small
    # enough to get packed together by chunk_transcript's own default cap).
    monkeypatch.setattr(chunking_module, "MAX_CHUNK_CHARS", 150)
    client = _client_returning([_tool_use_block([{"risk": "FX headwinds", "quote": "foreign currency headwinds"}])])

    extract_risks(RISK_TRANSCRIPT, client=client)

    sent_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Tim Cook" in sent_content
    assert "Suhasini Chandramouli" not in sent_content  # her turn doesn't mention any risk keyword


# forced structured extraction
def test_extracts_a_list_of_risk_quote_pairs():
    client = _client_returning([_tool_use_block([
        {"risk": "Foreign currency headwinds", "quote": "foreign currency headwinds"},
        {"risk": "Supply chain constraints on components", "quote": "supply chain constraints in certain component categories"},
    ])])

    risks = extract_risks(RISK_TRANSCRIPT, client=client)

    assert risks == [
        {"risk": "Foreign currency headwinds", "quote": "foreign currency headwinds"},
        {"risk": "Supply chain constraints on components", "quote": "supply chain constraints in certain component categories"},
    ]


def test_uses_the_sonnet_class_model_and_forces_the_extract_risks_tool():
    client = _client_returning([_tool_use_block([])])

    extract_risks(RISK_TRANSCRIPT, client=client)

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == risk_module.RISK_MODEL
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "extract_risks"}


def test_an_empty_risks_list_from_the_model_is_a_valid_result():
    client = _client_returning([_tool_use_block([])])

    assert extract_risks(RISK_TRANSCRIPT, client=client) == []


# graceful degradation for a non-compliant model response
def test_no_tool_use_block_returns_empty_list():
    client = _client_returning([_text_block("I found no risks.")])

    assert extract_risks(RISK_TRANSCRIPT, client=client) == []


def test_risks_field_not_a_list_returns_empty_list():
    client = _client_returning([SimpleNamespace(type="tool_use", id="t1", name="extract_risks", input={"risks": "not a list"})])

    assert extract_risks(RISK_TRANSCRIPT, client=client) == []


def test_malformed_items_are_dropped_but_valid_ones_survive():
    client = _client_returning([_tool_use_block([
        {"risk": "Valid risk", "quote": "a real quote"},
        {"risk": "Missing quote key"},
        "not even a dict",
        {"quote": "missing risk key"},
    ])])

    assert extract_risks(RISK_TRANSCRIPT, client=client) == [{"risk": "Valid risk", "quote": "a real quote"}]


# generate_risks: persistence 
def test_generate_risks_upserts_on_transcript_id(monkeypatch):
    risks = [{"risk": "FX headwinds", "quote": "foreign currency headwinds"}]
    monkeypatch.setattr(risk_module, "extract_risks", MagicMock(return_value=risks))
    db = MagicMock()
    db.execute.return_value.scalar_one.return_value = 13
    persisted_row = object()
    db.get.return_value = persisted_row

    result = generate_risks(db, transcript_id=13, raw_text="irrelevant - extract_risks is mocked")

    assert result is persisted_row
    db.get.assert_called_once_with(EarningsInsight, 13)

    stmt = db.execute.call_args.args[0]
    params = _bound_params(stmt)
    assert params["transcript_id"] == 13
    assert params["risks"] == risks
    assert set(stmt._post_values_clause.inferred_target_elements) == {"transcript_id"}


def test_generate_risks_persists_an_empty_list_when_none_were_found(monkeypatch):
    monkeypatch.setattr(risk_module, "extract_risks", MagicMock(return_value=[]))
    db = MagicMock()
    db.execute.return_value.scalar_one.return_value = 2

    generate_risks(db, transcript_id=2, raw_text="irrelevant")

    stmt = db.execute.call_args.args[0]
    assert _bound_params(stmt)["risks"] == []


def test_generate_risks_swallows_a_model_failure(monkeypatch, caplog):
    monkeypatch.setattr(risk_module, "extract_risks", MagicMock(side_effect=RuntimeError("model unavailable")))
    db = MagicMock()

    with caplog.at_level("ERROR", logger="src.earnings.risk"):
        result = generate_risks(db, transcript_id=6, raw_text="irrelevant")  # must not raise

    assert result is None
    db.execute.assert_not_called()
    assert any("6" in record.message for record in caplog.records)
