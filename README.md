# Titan AI: Market Scanner

Quantitative Stock Intelligence Platform

A Streamlit app that scans the S&P 500, scores each company on Value,
Momentum, Quality, Solvency, and Volatility factors, and surfaces the
top-ranked picks with a Wall Street-style writeup and radar chart.

## Roadmap

Titan is evolving into an AI-powered investment research platform on top
of the scanner above:

- **Phase 1 — AI Investment Chat**: ask natural-language questions
  ("Why is Microsoft ranked above Google?", "What are Nvidia's biggest
  risks?") and get answers grounded in Titan's own factor scores and
  rankings, with source citations — not generic LLM knowledge.
- **Phase 2 — Earnings Intelligence**: AI-generated earnings call
  summaries, guidance-direction tracking, sentiment, risk extraction, and
  quarter-over-quarter comparisons, feeding into the same chat.

The full design — system architecture, per-feature technical specs,
GitHub milestones/issues, target repo structure, and longer-term
expansion (Portfolio Builder, Backtesting, Multi-Agent Analyst, Titan
Copilot, Real-time Alerts) — lives in [`docs/`](docs/README.md). The
scanner in this repo keeps working as-is; Phase 1 is additive, not a
rewrite.

## Project structure

```
.
├── app.py                 # Streamlit entrypoint (UI + scan orchestration)
├── titan/
│   ├── config.py          # Scoring weights
│   ├── data.py             # S&P 500 ticker universe (Wikipedia scrape + fallback)
│   └── analyst.py          # RoboAnalyst: per-ticker fundamental/technical scoring
├── src/
│   ├── analytics/
│   │   ├── scanner_service.py  # Runs + persists a scan (scan_runs, factor_scores, companies)
│   │   └── scheduler.py        # Runs the scan on a recurring (hourly) schedule
│   ├── api/
│   │   ├── main.py             # FastAPI app + router registration
│   │   ├── config.py           # Settings sourced from env vars (API_HOST, API_PORT, ...)
│   │   ├── deps.py             # Shared FastAPI dependencies (e.g. get_db)
│   │   └── routes/
│   │       └── health.py       # GET /health — DB connectivity check
│   └── data/
│       ├── models.py       # SQLAlchemy models: companies, scan_runs, factor_scores
│       ├── db.py           # Engine/session, reads DATABASE_URL
│       └── migrations/     # Alembic migrations
├── docker-compose.yml     # Local Postgres for dev
├── alembic.ini
├── .streamlit/
│   └── config.toml        # Theme + Streamlit runtime settings
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Click **Initialize Market Scan** in the UI to pull the current S&P 500
constituent list, fetch price history + fundamentals for each ticker via
`yfinance`, and rank the results.

## Database (Postgres + Alembic)

Every scan persists to Postgres (`companies`, `scan_runs`, `factor_scores`
— see [docs/architecture.md](docs/architecture.md#4-database-schema-initial))
via `src/analytics/scanner_service.py:run_scan_and_persist()`, which both
the "Initialize Market Scan" button and the scheduler below call. A
Postgres instance is therefore required to run a scan, not optional.

```bash
# Start local Postgres (maps to host port 5433 to avoid clashing with
# any Postgres you already have on 5432)
docker compose up -d db

# Apply migrations
alembic upgrade head
```

By default the app connects to
`postgresql+psycopg2://titan:titan@localhost:5433/titan`. Override with
the `DATABASE_URL` environment variable to point at a different
database (e.g. in CI or production).

## Scheduled scanning

Instead of only scanning on a manual button click, `src/analytics/scheduler.py`
runs the same scan on a recurring interval (hourly by default, matching
`app.py`'s `st.cache_data(ttl=3600)` cadence) as a standalone process:

```bash
python -m src.analytics.scheduler
```

- Overlapping runs are prevented: `run_scan_and_persist()` refuses to
  start a new scan (raising `ScanAlreadyRunningError`) while a
  `scan_runs` row is still `status="running"` — whether that run was
  triggered by the scheduler or the Streamlit button. A `running` row
  older than 2 hours is treated as an abandoned/crashed run and no
  longer blocks new scans.
- A failed scheduled run is logged (`logger.exception`, full traceback)
  rather than crashing the process, so future scheduled runs still fire.

## API Gateway

`src/api/` scaffolds the FastAPI service that Phase 1's `/chat`,
`/rankings`, `/company/{ticker}`, `/compare`, and `/earnings/{ticker}`
endpoints (see [docs/architecture.md](docs/architecture.md)) get added
to. Routes are registered in `src/api/main.py`.

```bash
uvicorn src.api.main:app --reload
# or: python -m src.api.main
```

```bash
curl http://localhost:8000/health
# {"status": "ok", "db": "ok"}          -> 200, Postgres reachable
# {"detail": {"status": "error", ...}}  -> 503, Postgres unreachable

curl http://localhost:8000/rankings
# [{"ticker": "ACGL", "scan_run_id": 2, "computed_at": "...",
#   "composite_score": 98.8, "value_score": ..., "rating": "STRONG BUY", ...}, ...]
# sorted by composite_score desc; [] if no scan has produced any factor_scores yet

curl "http://localhost:8000/rankings?factor=momentum"
# same shape, sorted by momentum_score desc instead

curl http://localhost:8000/company/AAPL
# {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", ...,
#  "rating": "BUY", "composite_score": 71.68, "raw_metrics": {"RSI": ..., "Beta": ...}}
# companies row + its latest factor_scores row; 404 for an unknown ticker

curl "http://localhost:8000/compare?tickers=MSFT,GOOGL"
# {"scan_run_id": 7, "tickers": [{"ticker": "MSFT", ...}, {"ticker": "GOOGL", ...}],
#  "missing_tickers": [], "delta_between": ["MSFT", "GOOGL"],
#  "deltas": [{"factor": "momentum", "a": 80.0, "b": 100.0, "delta": -20.0}, ...]}
# sorted by abs(delta) desc; both tickers always pinned to the same (latest)
# scan_run_id — a ticker missing from that run is listed in missing_tickers
# rather than silently compared using stale data; 400 if fewer than 2 of the
# requested tickers have data in the latest scan
```

Config (`API_HOST`, `API_PORT`, `API_ENV`, `API_LOG_LEVEL`) is sourced
from environment variables via `src/api/config.py`, with local-dev
defaults — nothing is hardcoded. The DB connection itself reuses the
same pooled SQLAlchemy engine as the scanner (`src/data/db.py`,
`DATABASE_URL`), via the `get_db` dependency in `src/api/deps.py` that
future routes will depend on for their own DB access.

## Notes

- `app.py` disables SSL certificate verification globally
  (`ssl._create_default_https_context`) as a workaround for the
  "Certificate Verify Failed" error some Mac/Python installs hit when
  calling Yahoo Finance / Wikipedia. This is a known tradeoff for local
  convenience — do not rely on it for anything security-sensitive.
- Ticker scans run concurrently via `ThreadPoolExecutor`; the "Thread
  Power" slider in the sidebar controls how many tickers are analyzed at
  once (higher = faster scans, higher risk of being rate-limited by Yahoo
  Finance).
- This tool is for informational/educational purposes only and is not
  financial advice.
