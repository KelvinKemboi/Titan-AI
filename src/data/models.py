from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    TIMESTAMP,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

from src.embeddings.service import EMBEDDING_DIMENSION

Base = declarative_base()


class Company(Base):
    """Reference data per ticker, refreshed on each scan run."""

    __tablename__ = "companies"

    ticker = Column(String, primary_key=True)
    name = Column(String)
    sector = Column(String)
    industry = Column(String)
    description = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


class ScanRun(Base):
    """One row per Scanner Service execution."""

    __tablename__ = "scan_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    started_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    completed_at = Column(TIMESTAMP(timezone=True))
    universe_size = Column(Integer)
    status = Column(String, nullable=False, default="running")


class FactorScore(Base):
    """One row per ticker per scan run - the factor-score time series."""

    __tablename__ = "factor_scores"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String, ForeignKey("companies.ticker"), nullable=False)
    scan_run_id = Column(BigInteger, ForeignKey("scan_runs.id"), nullable=False)
    value_score = Column(Numeric)
    momentum_score = Column(Numeric)
    quality_score = Column(Numeric)
    solvency_score = Column(Numeric)
    volatility_score = Column(Numeric)
    composite_score = Column(Numeric)
    rating = Column(String)  # STRONG BUY | BUY | HOLD | SELL
    raw_metrics = Column(JSONB)  # price, rsi, peg, beta, etc.
    computed_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_factor_scores_ticker_computed_at", "ticker", computed_at.desc()),
    )


class ChatSession(Base):
    """A chat conversation, owned by the authenticated caller. A session
    owned by one user can't be read or extended by another."""

    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class ChatMessage(Base):
    """One turn (user question or assistant response) in a chat session.
    `sources` mirrors ToolResult.sources for assistant messages, empty for
    user ones. `request_id` groups a turn with the llm_calls/tool_calls
    rows traced for it."""

    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    sources = Column(JSONB)
    request_id = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_chat_messages_session_id_created_at", "session_id", "created_at"),
        Index("ix_chat_messages_request_id", "request_id"),
    )


class MemoEmbedding(Base):
    """Embedded RoboAnalyst.generate_memo() text for one ticker in one scan
    run - the qualitative retrieval source behind the search_memos tool."""

    __tablename__ = "memo_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String, ForeignKey("companies.ticker"), nullable=False)
    scan_run_id = Column(BigInteger, ForeignKey("scan_runs.id"), nullable=False)
    memo_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "ix_memo_embeddings_ticker_scan_run_id",
            "ticker", "scan_run_id",
            unique=True,
        ),
    )


class EarningsTranscript(Base):
    """One earnings call transcript for one (ticker, fiscal_year,
    fiscal_quarter). Ingestion upserts on that same unique index, so
    re-ingesting an already-stored quarter is a no-op."""

    __tablename__ = "earnings_transcripts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String, ForeignKey("companies.ticker"), nullable=False)
    fiscal_quarter = Column(String, nullable=False)  # e.g. "Q2"
    fiscal_year = Column(Integer, nullable=False)
    raw_text = Column(Text, nullable=False)
    source_url = Column(String)
    ingested_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "ix_earnings_transcripts_ticker_fiscal_year_fiscal_quarter",
            "ticker", "fiscal_year", "fiscal_quarter",
            unique=True,
        ),
    )


class EarningsChunk(Base):
    """One embedded chunk of an earnings call transcript, tagged by
    section (prepared remarks or Q&A)."""

    __tablename__ = "earnings_chunks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    transcript_id = Column(BigInteger, ForeignKey("earnings_transcripts.id"), nullable=False)
    chunk_type = Column(String, nullable=False)  # prepared_remarks | qna
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=False)

    __table_args__ = (
        Index("ix_earnings_chunks_transcript_id", "transcript_id"),
    )


class EarningsInsight(Base):
    """Derived insights for one earnings call transcript - one row per
    transcript, populated incrementally as each extraction pass
    (summary, guidance, sentiment, risks, QoQ diff) completes."""

    __tablename__ = "earnings_insights"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    transcript_id = Column(BigInteger, ForeignKey("earnings_transcripts.id"), nullable=False)
    summary = Column(Text)
    guidance_direction = Column(String)  # raised | maintained | lowered | none_given | unclear
    guidance_quote = Column(Text)  # supporting quote for guidance_direction
    sentiment_score = Column(Numeric)  # -1..1
    risks = Column(JSONB)
    qoq_changes = Column(JSONB)
    generated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_earnings_insights_transcript_id", "transcript_id", unique=True),
    )


class EarningsIngestionJob(Base):
    """One calendar-discovered earnings date for one ticker, and the
    state of this system's attempts to ingest its transcript. Retries
    back off on a miss rather than failing permanently, and give up
    (`status="exhausted"`) only after repeated attempts or too much time
    elapsed."""

    __tablename__ = "earnings_ingestion_jobs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String, ForeignKey("companies.ticker"), nullable=False)
    earnings_date = Column(Date, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending | succeeded | exhausted
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(String)
    attempt_count = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(TIMESTAMP(timezone=True), nullable=False)
    last_attempted_at = Column(TIMESTAMP(timezone=True))
    last_error = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "ix_earnings_ingestion_jobs_ticker_earnings_date",
            "ticker", "earnings_date",
            unique=True,
        ),
        Index(
            "ix_earnings_ingestion_jobs_status_next_attempt_at",
            "status", "next_attempt_at",
        ),
    )


class EarningsBackfillProgress(Base):
    """Tracks the one-time historical backfill per ticker, so a killed
    or resumed run skips tickers already marked "done" instead of
    restarting from scratch. `status="failed"` (the provider call itself
    failed) is retried on the next run."""

    __tablename__ = "earnings_backfill_progress"

    ticker = Column(String, ForeignKey("companies.ticker"), primary_key=True)
    status = Column(String, nullable=False)  # done | failed
    quarters_ingested = Column(Integer, nullable=False, default=0)
    attempted_at = Column(TIMESTAMP(timezone=True), nullable=False)
    error = Column(Text)


class LLMCall(Base):
    """One `client.messages.create` call and its token usage, cost, and
    latency. `request_id` is a fresh UUID minted once per chat turn -
    every LLM and tool call made while answering that turn shares it, so
    a bad answer can be traced back to everything that produced it."""

    __tablename__ = "llm_calls"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String, nullable=False)
    call_type = Column(String, nullable=False)  # intent_classification | chat_generation | ...
    model = Column(String, nullable=False)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cost_usd = Column(Numeric)  # null when the model has no entry in the pricing table
    latency_ms = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_llm_calls_request_id", "request_id"),
        Index("ix_llm_calls_created_at", "created_at"),
    )


class ToolCall(Base):
    """One tool call's execution and outcome, keyed by the same
    request_id as the LLMCall rows from the same chat turn. `tool_input`
    and `sources` capture this call's own arguments and results,
    independent of what the final answer ended up citing."""

    __tablename__ = "tool_calls"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String, nullable=False)
    tool_name = Column(String, nullable=False)
    tool_input = Column(JSONB)
    success = Column(Boolean, nullable=False)
    error = Column(Text)
    sources = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_tool_calls_request_id", "request_id"),
    )
