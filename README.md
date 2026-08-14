# Titan AI: Market Scanner

Quantitative Stock Intelligence Platform

A Streamlit app that scans the S&P 500, scores each company on Value,
Momentum, Quality, Solvency, and Volatility factors, and surfaces the
top-ranked picks with a Wall Street-style writeup and radar chart.

## Roadmap

Titan is evolving into an AI-powered investment research platform on top
of the scanner above:

- **Phase 1 - AI Investment Chat**: ask natural-language questions
  ("Why is Microsoft ranked above Google?", "What are Nvidia's biggest
  risks?") and get answers grounded in Titan's own factor scores and
  rankings, with source citations - not generic LLM knowledge.
- **Phase 2 - Earnings Intelligence**: AI-generated earnings call
  summaries, guidance-direction tracking, sentiment, risk extraction, and
  quarter-over-quarter comparisons, feeding into the same chat.

The full design - system architecture, per-feature technical specs,
GitHub milestones/issues, target repo structure, and longer-term
expansion (Portfolio Builder, Backtesting, Multi-Agent Analyst, Titan
Copilot, Real-time Alerts) - lives in [`docs/`](docs/README.md). The
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
│   │   ├── scheduler.py        # Runs the scan on a recurring (hourly) schedule
│   │   └── explain.py          # Factor Score Explanation Engine (per-factor score/weight/driver)
│   ├── api/
│   │   ├── main.py             # FastAPI app + router registration
│   │   ├── config.py           # Settings sourced from env vars (API_HOST, API_PORT, ...)
│   │   ├── deps.py             # Shared FastAPI dependencies (e.g. get_db)
│   │   └── routes/
│   │       ├── health.py       # GET /health - DB connectivity check
│   │       ├── rankings.py     # GET /rankings
│   │       ├── company.py      # GET /company/{ticker}
│   │       └── compare.py      # GET /compare
│   ├── agents/
│   │   └── tools/
│   │       ├── base.py         # Shared tool contract: every tool returns source metadata
│   │       └── factor_tools.py # get_factor_scores, compare_tickers Claude tool schemas
│   └── data/
│       ├── models.py       # SQLAlchemy models: companies, scan_runs, factor_scores
│       ├── db.py           # Engine/session, reads DATABASE_URL
│       └── migrations/     # Alembic migrations
├── scripts/
│   └── manual_test_tool_calling.py  # Live Claude tool-use verification
├── tests/
│   └── unit/               # Mirrors src/ structure
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
- see [docs/architecture.md](docs/architecture.md#4-database-schema-initial))
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
  `scan_runs` row is still `status="running"` - whether that run was
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
# scan_run_id - a ticker missing from that run is listed in missing_tickers
# rather than silently compared using stale data; 400 if fewer than 2 of the
# requested tickers have data in the latest scan
```

### Factor Score Explanation Engine

`src/analytics/explain.py` turns a `factor_scores` row into a structured
(typed, not prose) explanation - per-factor score, its `WEIGHTS` value,
its contribution to composite (`score * weight`), and a one-line driver
templated from the same thresholds `titan/analyst.py` scores against
(e.g. "PEG < 1.0 is elite"). Wired as an LLM tool below; it's also a
plain callable on its own, not exposed via a route:

```python
from src.analytics.explain import explain_ticker
from src.data.db import SessionLocal

with SessionLocal() as db:
    result = explain_ticker(db, "AAPL")
    # FactorScoreExplanation(ticker='AAPL', composite_score=71.68, factors=[
    #   FactorExplanation(factor='value', score=14.5, weight=0.25,
    #     contribution=3.625, driver='PEG of 2.71 (PEG < 1.0 is elite, PEG > 3.0 is poor)'),
    #   ...
    # ])
```

`explain_factor_scores(factor_score)` is the pure/deterministic core (no
DB access) - see `tests/unit/analytics/test_explain.py`. `explain_ticker`
is a thin DB-fetching wrapper that raises `ValueError` for a ticker with
no factor_scores row.

## LLM Tools

`src/agents/tools/factor_tools.py` defines the Claude tool-use schemas
for `get_factor_scores(ticker)` (backed by the Explanation Engine above)
and `compare_tickers(tickers)` (backed by `src/api/routes/compare.py`),
for the Chat/Agent Service a later issue builds. Every tool result is
wrapped in a `ToolResult` (`src/agents/tools/base.py`): `data` matching
the underlying function's response shape verbatim, plus `sources`
(`type`, `ticker`, `ref_id` = `scan_run_id`, `as_of` = when that scan
completed) per the source-attribution contract in
[docs/technical-design.md §5](docs/technical-design.md).

```python
from src.agents.tools.factor_tools import TOOLS, call_tool
from src.data.db import SessionLocal

with SessionLocal() as db:
    result = call_tool(db, "compare_tickers", {"tickers": ["MSFT", "GOOGL"]})
    # result.data matches compare_tickers()'s response shape exactly
    # result.sources -> [Source(type='factor_score', ticker='MSFT', ref_id=9, as_of=...), ...]
```

`TOOLS` is the list of tool schemas to pass to the Anthropic Messages API
(`tools=TOOLS`); `call_tool(db, name, input)` dispatches a `tool_use`
block's `name`/`input` to the matching implementation and raises
`ValueError` for an unknown tool or ticker (turning that into a graceful
response is the Chat/Agent Service's job, not this dispatcher's).

Manual live-model verification (requires `ANTHROPIC_API_KEY` and a
completed scan in Postgres):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python -m scripts.manual_test_tool_calling
```

This sends a comparison question to Claude with both tool schemas
attached and checks it calls `compare_tickers` rather than answering
from memory, then executes the real tool call and feeds the result back
so the final answer can be checked against the actual numbers.

## Tests

```bash
pytest
```

Config (`API_HOST`, `API_PORT`, `API_ENV`, `API_LOG_LEVEL`) is sourced
from environment variables via `src/api/config.py`, with local-dev
defaults - nothing is hardcoded. The DB connection itself reuses the
same pooled SQLAlchemy engine as the scanner (`src/data/db.py`,
`DATABASE_URL`), via the `get_db` dependency in `src/api/deps.py` that
future routes will depend on for their own DB access.

## Notes

- `app.py` disables SSL certificate verification globally
  (`ssl._create_default_https_context`) as a workaround for the
  "Certificate Verify Failed" error some Mac/Python installs hit when
  calling Yahoo Finance / Wikipedia. This is a known tradeoff for local
  convenience - do not rely on it for anything security-sensitive.
- Ticker scans run concurrently via `ThreadPoolExecutor`; the "Thread
  Power" slider in the sidebar controls how many tickers are analyzed at
  once (higher = faster scans, higher risk of being rate-limited by Yahoo
  Finance).
- This tool is for informational/educational purposes only and is not
  financial advice.
