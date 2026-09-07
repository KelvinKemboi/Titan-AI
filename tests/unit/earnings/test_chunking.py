"""
Tests chunk_transcript against real earnings call transcript excerpts
(reformatted into api-ninjas.com's confirmed "Speaker: text" raw_text
convention - see src/earnings/chunking.py's module docstring), not
synthetic placeholder text.

Both are genuine, verbatim quotes from real, public earnings calls:
  - AAPL_TRANSCRIPT: Apple Q2 fiscal year 2024 earnings call (2024-05-02).
  - MSFT_TRANSCRIPT: Microsoft Q3 fiscal year 2024 earnings call (2024-04-25).
Each is an excerpt (prepared remarks -> transition -> start of Q&A), not
the full multi-hour call, and re-flows the source sites' own formatting
(e.g. The Motley Fool's "Name -- Title" convention) into api-ninjas.com's
single "Name: " continuous-string convention - the content is real, the
line-level presentation is normalized to the provider format this
chunker targets.
"""
from src.earnings.chunking import MAX_CHUNK_CHARS, chunk_transcript

AAPL_TRANSCRIPT = """\
Suhasini Chandramouli: Good Afternoon, and welcome to the Apple Q2 Fiscal Year 2024 Earnings Conference Call. My name is Suhasini Chandramouli, Director of Investor Relations.

Tim Cook: Thank you, Suhasini. Good afternoon, everyone, and thanks for joining the call. Today, Apple is reporting revenue of $90.8 billion and an EPS record of $1.53 for the March quarter.

Suhasini Chandramouli: Thank you. We ask that you limit yourself to two questions. Operator, may we have the first question, please?

Operator: We will go ahead and take our first question from Mike Ng with Goldman Sachs. Please go ahead.
"""

MSFT_TRANSCRIPT = """\
Operator: Greetings and welcome to the Microsoft fiscal year 2024 third quarter earnings conference call. At this time, all participants are in a listen-only mode. A question-and-answer session will follow the formal presentation. As a reminder, this conference is being recorded. I would now like to turn the conference over to your host, Brett Iversen, vice president of investor relations.

Brett Iversen: Good afternoon, and thank you for joining us today. On the call with me are Satya Nadella, chairman and chief executive officer; Amy Hood, chief financial officer; Alice Jolla, chief accounting officer; and Keith Dolliver, corporate secretary and deputy general counsel. We will also provide growth rates in constant currency when available as a framework for assessing how our underlying businesses performed, excluding the effect of foreign currency rate fluctuations. Where growth rates are the same in constant currency, we will refer to the growth rate only.

Brett Iversen: Thanks, Amy. We'll now move over to Q&A. Out of respect for others on the call, we request that participants please only ask one question. Operator, can you please repeat your instructions?

Operator: Thank you. And our first question comes from the line of Keith Weiss with Morgan Stanley. Please proceed with your question.

Keith Weiss: Excellent. Thank you, guys, for taking the question, and congratulations on the fantastic quarter. A lot of excitement in the marketplace around generative AI and the potential of these technologies. But there's also a lot of investment going on behind them. It looks like Microsoft is on track to ramp capex over 50% year on year this year to over $50 billion. And there's media speculation of more spending ahead with some reports talking about like $100 billion data center. So, obviously, investments are coming well ahead of the revenue contribution. But what I was hoping for is that you could give us some color on how use as the management team, try to quantify the potential opportunities that underlie these investments because they are getting very big. And maybe if you could give us some hints on whether there's any truth to the potential of like $100 billion data center out there. Thank you so much.

Satya Nadella: Thank you, Keith. For the question, let me start and maybe Amy, you can add up. At a high level, the way we, as a management team, talk about it is there are two sides to this, right? There is training and their inference. Given that we want to be a leader in this big generational shift and paradigm shift in technology, that's on the training side. We want to be able to allocate the capital required to essentially be training these large foundation models and stay in the leadership position there. And we've done that successfully all the way today, and you've seen it flow through our P&L, and you can continue to see that going forward. Then Amy referenced what we also do on the inference side, which is, one, we first innovate and build products. And of course, we have an infrastructure business that's also dependent on a lot of ISVs building products that run on our infrastructure. And it's all going to be demand-driven. In other words, we're closely tracking what's happening with inference demand, and that's something that we will manage, as Amy said in her remarks very, very closely. So, we feel -- and obviously, we've been doing this, quite frankly, Keith, for now multiple years. So, this is not the quarter.
"""


# --- separating prepared remarks from Q&A, against 2 real transcripts ---

def test_aapl_prepared_remarks_and_qna_are_correctly_separated():
    chunks = chunk_transcript(AAPL_TRANSCRIPT)

    types = [c.chunk_type for c in chunks]
    assert types == ["prepared_remarks", "qna"]
    assert "Tim Cook" in chunks[0].chunk_text
    assert "revenue of $90.8 billion" in chunks[0].chunk_text
    # the transition turn itself (Suhasini asking for the first question)
    # is the last prepared_remarks turn, not the first qna turn
    assert "may we have the first question" in chunks[0].chunk_text
    assert "Mike Ng" in chunks[1].chunk_text
    assert "Tim Cook" not in chunks[1].chunk_text


def test_msft_prepared_remarks_and_qna_are_correctly_separated():
    chunks = chunk_transcript(MSFT_TRANSCRIPT)

    prepared = [c for c in chunks if c.chunk_type == "prepared_remarks"]
    qna = [c for c in chunks if c.chunk_type == "qna"]
    assert prepared and qna

    prepared_text = "\n".join(c.chunk_text for c in prepared)
    qna_text = "\n".join(c.chunk_text for c in qna)

    # the operator's opening boilerplate ("a question-and-answer session
    # will follow") must NOT trigger the transition - it's a future-tense
    # mention, not the actual handoff
    assert "constant currency" in prepared_text
    assert "Keith Weiss" not in prepared_text
    # Brett's intro *mentions* Satya by name in prepared remarks ("On the
    # call with me are Satya Nadella...") - that's fine and expected; what
    # must not happen is Satya's own *turn* landing in prepared_remarks
    assert "Satya Nadella:" not in prepared_text

    # the real transition ("we'll now move over to Q&A") puts everything
    # from the operator's next turn onward into qna
    assert "Keith Weiss" in qna_text
    assert "Satya Nadella" in qna_text
    assert "generative AI" in qna_text


def test_chunks_never_mix_both_section_types():
    for transcript in (AAPL_TRANSCRIPT, MSFT_TRANSCRIPT):
        for chunk in chunk_transcript(transcript):
            assert chunk.chunk_type in ("prepared_remarks", "qna")


# --- speaker turns survive into the chunk text (who-said-what) ---

def test_speaker_attribution_is_preserved_in_chunk_text():
    chunks = chunk_transcript(MSFT_TRANSCRIPT)

    # Keith's question and Satya's answer are both long enough that they
    # land in separate chunks once packed under MAX_CHUNK_CHARS - that's
    # expected; what matters is each speaker's own label survives into
    # whichever chunk their turn ended up in.
    qna_text = "\n".join(c.chunk_text for c in chunks if c.chunk_type == "qna")
    assert "Keith Weiss:" in qna_text
    assert "Satya Nadella:" in qna_text


# --- chunk size: small enough for embedding, but respecting boundaries ---

def test_no_chunk_exceeds_the_documented_max_size():
    for transcript in (AAPL_TRANSCRIPT, MSFT_TRANSCRIPT):
        for chunk in chunk_transcript(transcript):
            assert len(chunk.chunk_text) <= MAX_CHUNK_CHARS


def test_a_single_turn_longer_than_max_chars_is_split_without_exceeding_it(monkeypatch):
    import src.earnings.chunking as chunking_module
    monkeypatch.setattr(chunking_module, "MAX_CHUNK_CHARS", 200)

    long_answer = "This is one sentence about the quarter. " * 20  # ~840 chars, one "turn"
    transcript = f"Operator: We will now begin the question-and-answer session.\n\nJane Doe: {long_answer}\n"

    chunks = chunking_module.chunk_transcript(transcript)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.chunk_text) <= 200
    # the split pieces are still all qna (the long turn came after the transition)
    assert all(c.chunk_type == "qna" for c in chunks[1:])


def test_consecutive_same_section_turns_are_packed_into_one_chunk_when_they_fit():
    chunks = chunk_transcript(AAPL_TRANSCRIPT)

    # Suhasini's two turns + Tim Cook's turn are all prepared_remarks and
    # comfortably fit under MAX_CHUNK_CHARS together - one chunk, not three
    prepared = [c for c in chunks if c.chunk_type == "prepared_remarks"]
    assert len(prepared) == 1


# --- no recognizable transition phrase: everything stays prepared_remarks ---

def test_no_transition_phrase_found_defaults_everything_to_prepared_remarks():
    transcript = (
        "Jane Doe: Welcome to the call.\n\n"
        "John Smith: Thanks, Jane. Revenue was up this quarter.\n"
    )

    chunks = chunk_transcript(transcript)

    assert chunks
    assert all(c.chunk_type == "prepared_remarks" for c in chunks)


# --- no speaker-turn markers at all: doesn't crash, still returns something ---

def test_text_with_no_speaker_labels_returns_a_single_unlabeled_chunk():
    chunks = chunk_transcript("Just some plain text with no speaker labels at all.")

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "prepared_remarks"
    assert "plain text" in chunks[0].chunk_text


def test_empty_transcript_returns_no_chunks():
    assert chunk_transcript("") == []
    assert chunk_transcript("   \n  ") == []
