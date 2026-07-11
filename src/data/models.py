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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Company(Base):
    """Reference data per ticker. Refreshed on each scan run."""

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
    """One row per ticker per scan run — the factor-score time series."""

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
