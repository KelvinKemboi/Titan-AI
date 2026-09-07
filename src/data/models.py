from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    TIMESTAMP,
) # datatypes for the columns in the database tables
from sqlalchemy.dialects.postgresql import JSONB, UUID # PostgreSQL-specific JSONB/UUID datatypes
from sqlalchemy.orm import declarative_base # object-relational mapping (ORM) base class for defining models
from sqlalchemy.sql import func

from src.embeddings.service import EMBEDDING_DIMENSION

Base = declarative_base()

# reference data per ticker. Refreshed on each scan run.
class Company(Base):
    """Reference dataxa per ticker. Refreshed on each scan run."""

    __tablename__ = "companies"

    ticker = Column(String, primary_key=True)
    name = Column(String)
    sector = Column(String)
    industry = Column(String)
    description = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

# log of each ticker's scan run
class ScanRun(Base):
    """One row per Scanner Service execution."""

    __tablename__ = "scan_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    started_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    completed_at = Column(TIMESTAMP(timezone=True))
    universe_size = Column(Integer)
    status = Column(String, nullable=False, default="running")

# store the factor scores for each ticker in each scan run
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
    raw_metrics = Column(JSONB)  # price, rsi, peg, beta, etc. (RoboAnalyst.metrics)
    computed_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_factor_scores_ticker_computed_at", "ticker", computed_at.desc()),
    )

# one row per chat conversation
class ChatSession(Base):
    """A chat conversation, owned by the authenticated caller (src/api/auth.py's
    MVP API-key scheme) - chat_repository.get_or_create_session sets user_id on
    every new row and rejects a request for an existing session_id owned by a
    different user_id. Nullable at the DB level only for rows created before
    auth existed; every new row always has one."""

    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

# one row per user/assistant turn in a chat_sessions conversation
class ChatMessage(Base):
    """One turn (user question or assistant response) in a chat_sessions
    conversation. `sources`mirrors ToolResult.sources for assistant messages; empty for user ones."""

    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    sources = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_chat_messages_session_id_created_at", "session_id", "created_at"),
    )

# one row per ticker per scan run: the embedded analyst memo (RoboAnalyst.generate_memo()
# output) - Titan's first qualitative retrieval source (docs/architecture.md #2, #6)
class MemoEmbedding(Base):
    """Embedded RoboAnalyst.generate_memo() text for one ticker in one scan
    run, searched by src/analytics/memo_search.py (cosine similarity) and
    exposed to chat via the search_memos tool
    (src/agents/tools/memo_tools.py)."""

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

# one row per ticker per fiscal quarter
class EarningsTranscript(Base):
    """One earnings call transcript for one (ticker, fiscal_year,
    fiscal_quarter) - fetched via src/earnings/provider_client.py and
    persisted by src/earnings/ingestion.py:ingest_transcript, which is
    idempotent on the same unique index this table enforces."""

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
