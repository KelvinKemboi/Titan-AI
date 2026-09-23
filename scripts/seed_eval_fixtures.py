"""
Deterministic fixture data for scripts/eval_chat_service.py - a small, known dataset the eval's shape
checks can rely on, instead of "whatever happens to already be in the
dev DB" (fragile/non-reproducible - the problem with
scripts/eval_conversation_memory.py's own "requires a Postgres with a
completed scan that includes every ticker below" precondition).

Always inserts a *fresh* ScanRun (and fresh FactorScore/EarningsInsight
rows off it) rather than upserting one of the existing dev DB's ScanRuns, so that the eval's own
"latest scan_run_id" notion is consistent with the fixture's own
FactorScore rows. Also inserts a deliberately-stale ScanRun (and a
single FactorScore row off it) to test that the eval's own "latest scan_run_id" notion is global, not per-ticker.
"""
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.data.db import SessionLocal
from src.data.models import Company, EarningsInsight, EarningsTranscript, FactorScore, MemoEmbedding, ScanRun
from src.embeddings.service import EMBEDDING_DIMENSION

load_dotenv()

# Placeholder vector for memo_embeddings 
_PLACEHOLDER_VECTOR = [0.0] * EMBEDDING_DIMENSION


def _raw_metrics(price, rsi, trend, val_metric, val_type, margin, debt, beta, scores):
    return {
        "Price": price, "RSI": rsi, "Trend": trend, "Val_Metric": val_metric, "Val_Type": val_type,
        "Margin": margin, "Debt": debt, "Beta": beta, "Scores": scores,
    }


def _upsert_company(db, ticker, name, sector, industry, description=None):
    stmt = (
        pg_insert(Company)
        .values(ticker=ticker, name=name, sector=sector, industry=industry, description=description)
        .on_conflict_do_update(
            index_elements=["ticker"],
            set_={"name": name, "sector": sector, "industry": industry, "description": description},
        )
    )
    db.execute(stmt)


def _upsert_transcript(db, ticker, fiscal_year, fiscal_quarter, raw_text, ingested_at) -> int:
    stmt = (
        pg_insert(EarningsTranscript)
        .values(
            ticker=ticker, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
            raw_text=raw_text, source_url=None, ingested_at=ingested_at,
        )
        .on_conflict_do_update(
            index_elements=["ticker", "fiscal_year", "fiscal_quarter"],
            set_={"raw_text": raw_text, "ingested_at": ingested_at},
        )
        .returning(EarningsTranscript.id)
    )
    return db.execute(stmt).scalar_one()


def _replace_insight(db, transcript_id, **fields) -> None:
    db.query(EarningsInsight).filter(EarningsInsight.transcript_id == transcript_id).delete()
    db.add(EarningsInsight(transcript_id=transcript_id, generated_at=datetime.now(timezone.utc), **fields))


def seed_eval_fixtures(db) -> dict:
    """Seeds MSFT/GOOGL/NVDA/AMD/IBM companies + factor scores (fresh
    scan), NVDA's 2 quarters of earnings history (Q1 insufficient_history,
    Q2 a real QoQ diff against Q1), AMD's 1 quarter (insufficient_history),
    and placeholder memo_embeddings for the qualitative search case.
    Returns {"scan_run_id": ..., "stale_scan_run_id": ...} for callers
    that want to reference the exact ids created.
    """
    now = datetime.now(timezone.utc)

    # The STALE scan must be created (and get its scan_run_id assigned)
    # BEFORE the fresh one below - compare_tickers/answer_question's FAQ
    # cache both treat the *global* MAX(scan_run_id) as "the latest scan,"
    # not a per-ticker notion, so this row must sit on a strictly older
    # (lower) scan_run_id or it would wrongly become "the latest scan"
    # itself and starve MSFT/GOOGL/NVDA/AMD of a comparable scan.
    #
    # Uses a dedicated synthetic ticker (not a real S&P 500 one, e.g.
    # IBM) precisely because get_factor_scores/explain_ticker resolve
    # "latest" per-ticker by MAX(computed_at) - a real ticker already
    # scanned by an actual run (this dev DB's IBM has real rows from
    # earlier full-universe scans, all newer than any deliberately-old
    # timestamp this fixture could set) would silently shadow this
    # fixture's intentionally-stale row with a real, fresher one.
    stale_computed_at = now - timedelta(days=120)
    stale_scan_run = ScanRun(
        status="complete", universe_size=1,
        started_at=stale_computed_at, completed_at=stale_computed_at,
    )
    db.add(stale_scan_run)
    db.flush()
    _upsert_company(db, "ZEVALSTALE", "Eval Fixture Stale Co.", "Technology", "IT Services")
    db.add(FactorScore(
        ticker="ZEVALSTALE", scan_run_id=stale_scan_run.id, value_score=60.0, momentum_score=50.0, quality_score=65.0,
        solvency_score=60.0, volatility_score=65.0, composite_score=59.75, rating="HOLD",
        raw_metrics=_raw_metrics(190.0, 50.0, "Bearish", 2.9, "PEG", 0.15, 90.0, 0.75, [60.0, 50.0, 65.0, 60.0, 65.0]),
        computed_at=stale_computed_at,
    ))

    scan_run = ScanRun(status="complete", universe_size=5, started_at=now, completed_at=now)
    db.add(scan_run)
    db.flush()  # assigns scan_run.id without committing yet

    for ticker, name, sector, industry in [
        ("MSFT", "Microsoft Corporation", "Technology", "Software"),
        ("GOOGL", "Alphabet Inc.", "Technology", "Internet Content & Information"),
        ("NVDA", "NVIDIA Corporation", "Technology", "Semiconductors"),
        ("AMD", "Advanced Micro Devices", "Technology", "Semiconductors"),
    ]:
        _upsert_company(db, ticker, name, sector, industry)

    # MSFT ranked above GOOGL - both from THIS scan_run (phase1_comparison case)
    db.add(FactorScore(
        ticker="MSFT", scan_run_id=scan_run.id, value_score=80.0, momentum_score=90.0, quality_score=95.0,
        solvency_score=85.0, volatility_score=80.0, composite_score=87.75, rating="STRONG BUY",
        raw_metrics=_raw_metrics(420.0, 62.0, "Bullish", 1.8, "PEG", 0.35, 40.0, 0.9, [80.0, 90.0, 95.0, 85.0, 80.0]),
        computed_at=now,
    ))
    db.add(FactorScore(
        ticker="GOOGL", scan_run_id=scan_run.id, value_score=70.0, momentum_score=65.0, quality_score=80.0,
        solvency_score=90.0, volatility_score=70.0, composite_score=73.5, rating="BUY",
        raw_metrics=_raw_metrics(170.0, 55.0, "Bullish", 2.2, "PEG", 0.28, 20.0, 1.05, [70.0, 65.0, 80.0, 90.0, 70.0]),
        computed_at=now,
    ))
    # NVDA - used for both the phase1 structured-risk case and every phase2 earnings case
    db.add(FactorScore(
        ticker="NVDA", scan_run_id=scan_run.id, value_score=40.0, momentum_score=95.0, quality_score=98.0,
        solvency_score=90.0, volatility_score=45.0, composite_score=76.75, rating="BUY",
        raw_metrics=_raw_metrics(140.0, 68.0, "Bullish", 3.4, "PEG", 0.55, 30.0, 1.7, [40.0, 95.0, 98.0, 90.0, 45.0]),
        computed_at=now,
    ))
    # AMD - only 1 ingested quarter (edge_insufficient_history)
    db.add(FactorScore(
        ticker="AMD", scan_run_id=scan_run.id, value_score=55.0, momentum_score=70.0, quality_score=75.0,
        solvency_score=80.0, volatility_score=55.0, composite_score=67.75, rating="BUY",
        raw_metrics=_raw_metrics(160.0, 58.0, "Bullish", 2.6, "PEG", 0.22, 45.0, 1.6, [55.0, 70.0, 75.0, 80.0, 55.0]),
        computed_at=now,
    ))

    db.flush()  # so FactorScore rows are queryable within this same transaction

    # NVDA: Q1 (first ever - insufficient_history) then Q2 (a real diff against Q1)
    nvda_q1_id = _upsert_transcript(
        db, "NVDA", 2026, "Q1", "synthetic NVDA Q1 transcript", now - timedelta(days=95),
    )
    db.flush()
    _replace_insight(
        db, nvda_q1_id, summary="Solid quarter driven by data center demand.",
        guidance_direction="maintained", guidance_quote="we expect continued strong demand",
        sentiment_score=0.3, risks=[{"risk": "Export restrictions on AI chips", "quote": "export controls remain a factor"}],
        qoq_changes={"status": "insufficient_history"},
    )

    nvda_q2_id = _upsert_transcript(
        db, "NVDA", 2026, "Q2", "synthetic NVDA Q2 transcript", now - timedelta(days=5),
    )
    db.flush()
    _replace_insight(
        db, nvda_q2_id, summary="Record data center revenue, driven by AI accelerator demand.",
        guidance_direction="raised", guidance_quote="we now expect accelerating growth next quarter",
        sentiment_score=0.8, risks=[{"risk": "Customer concentration among a few hyperscalers", "quote": "a small number of customers"}],
        qoq_changes={
            "status": "ok", "prior_transcript_id": nvda_q1_id, "prior_fiscal_year": 2026, "prior_fiscal_quarter": "Q1",
            "guidance_direction": {"prior": "maintained", "current": "raised", "changed": True},
            "sentiment_delta": 0.5,
            "new_risks": [{"risk": "Customer concentration among a few hyperscalers", "quote": "a small number of customers"}],
            "resolved_risks": [{"risk": "Export restrictions on AI chips", "quote": "export controls remain a factor"}],
        },
    )

    # AMD: exactly 1 ingested quarter -> insufficient_history
    amd_q1_id = _upsert_transcript(
        db, "AMD", 2026, "Q2", "synthetic AMD Q2 transcript", now - timedelta(days=5),
    )
    db.flush()
    _replace_insight(
        db, amd_q1_id, summary="Steady growth in the data center segment.",
        guidance_direction="maintained", guidance_quote="guidance is unchanged",
        sentiment_score=0.2, risks=[], qoq_changes={"status": "insufficient_history"},
    )

    # Placeholder memo_embeddings for the qualitative search case (real
    # ranking needs a working VOYAGE_API_KEY at eval time - see module docstring)
    for ticker, memo_text in [
        ("MSFT", "MSFT has a deep competitive moat in enterprise software and cloud infrastructure."),
        ("GOOGL", "GOOGL benefits from a durable moat in search and advertising distribution."),
        ("NVDA", "NVDA's moat rests on its CUDA software ecosystem and accelerator hardware lead."),
    ]:
        db.add(MemoEmbedding(
            ticker=ticker, scan_run_id=scan_run.id, memo_text=memo_text,
            embedding=_PLACEHOLDER_VECTOR, created_at=now,
        ))

    db.commit()
    return {"scan_run_id": scan_run.id, "stale_scan_run_id": stale_scan_run.id}


def main():
    db = SessionLocal()
    try:
        info = seed_eval_fixtures(db)
        print(f"Seeded eval fixtures: {info}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
