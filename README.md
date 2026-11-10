# Titan AI: Market Scanner

A quantitative research platform that scans the S&P 500, scores every
company on five factors, and layers an AI chat assistant with earnings
call intelligence on top.

## Features

- **Market Scanner** - scores each S&P 500 company on Value, Momentum,
  Quality, Solvency, and Volatility, then ranks and writes up the top
  picks with a radar chart (Streamlit UI).
- **AI Investment Chat** - ask natural-language questions ("Why is
  Microsoft ranked above Google?", "What are Nvidia's biggest risks?")
  and get answers grounded in Titan's own data, with source citations,
  conversation memory, and response caching.
- **Earnings Intelligence** - ingests earnings call transcripts and
  generates AI summaries, guidance-direction tracking, sentiment
  scoring, risk extraction, and quarter-over-quarter comparisons.
- **Observability** - every LLM/tool call is traced with cost and
  latency, queryable per session or per request.

## Project Structure

```
.
├── app.py                    # Streamlit entrypoint (UI + scan orchestration)
├── titan/
│   ├── config.py              # Scoring weights
│   ├── data.py                # S&P 500 ticker universe
│   └── analyst.py             # Per-ticker fundamental/technical scoring
├── src/
│   ├── analytics/              # Scan orchestration, scheduling, explanations, search
│   ├── api/                    # FastAPI app, routes, auth, config
│   ├── agents/                  # Chat service, intent classification, entity tracking, tools
│   ├── earnings/                 # Transcript ingestion, chunking, and AI extraction passes
│   ├── embeddings/               # Voyage AI embedding client
│   ├── observability/             # LLM spend + request tracing
│   └── data/                      # SQLAlchemy models, DB/cache setup, migrations
├── scripts/                    # Manual tests, evals, and one-off backfills
├── tests/unit/                 # Mirrors src/ structure
├── docker-compose.yml           # Local Postgres + Redis for dev
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

| Variable | Required for | Default |
|---|---|---|
| `DATABASE_URL` | Everything (Postgres) | `postgresql+psycopg2://titan:titan@localhost:5433/titan` |
| `REDIS_URL` | `/rankings` and `/company` caching | `redis://localhost:6380/0` (optional - degrades to no cache) |
| `API_KEYS` | `/chat` auth | none - unset means every request 401s |
| `ANTHROPIC_API_KEY` | Chat, intent classification, all earnings AI passes | - |
| `VOYAGE_API_KEY` | Embeddings (memo + earnings chunk search) | - |
| `EARNINGS_PROVIDER_API_KEY` | Earnings transcript ingestion (api-ninjas.com) | - |
| `API_HOST`, `API_PORT`, `API_ENV`, `API_LOG_LEVEL` | API server config | local-dev defaults |
| `CHAT_API_KEY` | Streamlit sidebar chat -> `/chat` (must be the key half of an `API_KEYS` pair) | none - chat 401s |
| `CHAT_TIMEOUT_SECONDS` | Streamlit -> `/chat` request timeout | `120` |

All entrypoints (`app.py`, the API, both schedulers) load `.env` from the repo root, e.g.:

```bash
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=...
API_KEYS=me:dev-key-123
CHAT_API_KEY=dev-key-123
```

## Running the App

```bash
streamlit run app.py
```

Click **Initialize Market Scan** to pull the current S&P 500 list, fetch
price/fundamentals via `yfinance`, and rank the results. A running
Postgres instance is required (see below).

## Database (Postgres + pgvector)

```bash
docker compose up -d db      # Postgres on host port 5433
alembic upgrade head          # applies all migrations, including pgvector setup
```

Every scan persists to `companies`, `scan_runs`, and `factor_scores` via
`src/analytics/scanner_service.py`. The `db` service uses
[pgvector](https://github.com/pgvector/pgvector) for the vector columns
backing memo and earnings-chunk similarity search - if you point
`DATABASE_URL` at a different Postgres instance, it needs pgvector
installed too.

## Redis (Caching)

```bash
docker compose up -d redis   # Redis on host port 6380
```

`GET /rankings` and `GET /company/{ticker}` cache their reads through
`src/data/cache.py`, invalidated on scan completion rather than a fixed
TTL. Redis is optional for correctness - any cache failure (unreachable,
timeout) falls back to a plain DB read, never a 500.

## Scheduled Scanning

```bash
python -m src.analytics.scheduler
```

Runs the same scan on a recurring hourly interval as a standalone
process. Overlapping runs are prevented (a run stuck for over 2 hours is
treated as abandoned and no longer blocks new ones); a failed run is
logged and skipped rather than crashing the process.

## API

```bash
uvicorn src.api.main:app --reload
```

### Auth

Every `/chat` and `/observability/*` request needs `Authorization: Bearer
<key>`. Keys are configured as `user_id:key` pairs in `API_KEYS`:

```bash
API_KEYS=alice:alice-key-123,bob:bob-key-456
```

An unset or empty `API_KEYS` fails closed (every request 401s) rather
than open. A `session_id` belonging to a different user is a 403, not a
silent cross-user read.

### Endpoints

| Method & Path | Description |
|---|---|
| `GET /health` | DB connectivity check |
| `GET /rankings` | Latest scan, sorted by composite score (or `?factor=` to sort by one factor) |
| `GET /company/{ticker}` | Company profile + latest factor scores |
| `GET /company/{ticker}/report` | Full analyst writeup, including recent earnings if available |
| `GET /compare?tickers=A,B` | Side-by-side factor deltas for two or more tickers |
| `POST /chat` | Ask a question; returns a sourced answer + `session_id` to continue the conversation |
| `GET /earnings/{ticker}` | Ingested transcripts + AI-extracted insights for a ticker |
| `GET /observability/spend` | Aggregate LLM spend across all traced calls |
| `GET /observability/spend/top-sessions` | Highest-cost chat sessions |
| `GET /observability/spend/{session_id}` | LLM spend for one session |
| `GET /observability/trace/{request_id}` | Every LLM/tool call made answering one request |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer alice-key-123" -H "Content-Type: application/json" \
  -d '{"message": "Why is AAPL rated a BUY?"}'
```

## AI Chat

`src/agents/chat_service.py` runs a Claude tool-calling loop over six
tools (`src/agents/tools/`): `get_factor_scores`, `compare_tickers`,
`search_memos`, `get_earnings_insight`, `search_earnings`, and
`get_analyst_report`. Every tool result carries source metadata (ticker,
scan/transcript reference, and verification detail), which the Streamlit
sidebar renders as an expandable citation under each answer.

A cheap intent classifier (`intent_classifier.py`) and an entity tracker
(`entity_tracker.py`, for resolving pronoun follow-ups like "what about
its momentum?") both bias the system prompt without ever gating which
tools are available. Repeat FAQ-style questions with no prior
conversation history are served from a response cache
(`src/data/cache.py`) keyed to the latest scan.

## Earnings Intelligence

`src/earnings/` ingests one ticker/quarter's transcript
(`provider_client.py` + `ingestion.py`, via api-ninjas.com) and runs it
through a pipeline of independent, idempotent passes:

1. **Chunking** (`chunking.py`) - splits by speaker turn into
   `prepared_remarks` / `qna` sections.
2. **Embedding** (`chunk_indexing.py`) - batch-embeds every chunk for
   semantic search.
3. **Summary, guidance, sentiment, and risk extraction** - independent
   LLM passes, each persisted to its own `earnings_insights` column.
4. **Quarter-over-quarter diff** (`qoq.py`) - a deterministic comparison
   against the ticker's prior ingested quarter, no LLM call.

Every pass degrades gracefully: a model failure is logged and swallowed
rather than blocking ingestion of the raw transcript. Results are
exposed via `GET /earnings/{ticker}` and the `get_earnings_insight` /
`search_earnings` chat tools.

## Embeddings

`src/embeddings/service.py` wraps [Voyage AI](https://www.voyageai.com/)'s
`voyage-large-2` model (1536-dimensional vectors) for both memo search
and earnings-chunk search. Batching (up to 128 texts/request) and retry
with backoff are handled internally.

## Scripts

Manual verification and eval scripts under `scripts/` (most require
`ANTHROPIC_API_KEY` and/or a Postgres with at least one completed scan):

| Script | Purpose |
|---|---|
| `manual_test_tool_calling.py` | Live check that Claude calls the right tool for a comparison question |
| `manual_test_memo_search.py` | Live check that qualitative questions route to memo search |
| `manual_test_earnings_provider.py` | Live api-ninjas.com transcript fetch |
| `manual_test_chat_service.py` / `manual_test_chat_memory.py` | Live Chat/Agent Service smoke tests |
| `manual_test_phase2_questions.py` | Live earnings-question smoke test |
| `eval_intent_classifier.py` | Accuracy + latency of intent classification against ~30 hand-written questions |
| `eval_conversation_memory.py` | Checks pronoun follow-ups resolve to the right ticker across turns |
| `eval_chat_service.py` | Golden Q&A regression eval for chat answer quality |
| `eval_guidance_extraction.py` | Guidance-extraction accuracy against known outcomes |
| `spot_check_sentiment.py` | Prints sentiment scores for known-tone Q&A exchanges, for human review |
| `backfill_earnings_transcripts.py` | One-time historical transcript backfill across the S&P 500 |
| `backfill_memo_embeddings.py` | Backfills memo embeddings for scans that predate memo search |
| `seed_eval_fixtures.py` | Seeds fixture data used by the eval scripts above |
| `report_llm_spend.py` | Prints aggregate LLM spend from the CLI |
| `backtest_signal.py` | Does a higher composite score predict higher forward returns? (IC, quintile spread, hit rate, per-rating returns) |

## Observability (Prometheus + Grafana)

The API exposes Prometheus metrics on `GET /metrics` (`src/observability/metrics.py`):
HTTP rate/latency/status per route, LLM calls/tokens/cost/latency per model and
call type, tool calls per tool and outcome, and chat turn outcomes (`answered`,
`cache_hit`, `llm_error`, `inconclusive`), round-trips per turn, and citations per answer.

```bash
docker compose --profile observability up -d   # Prometheus :9090, Grafana :3000 (admin/admin)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Grafana auto-provisions the **Titan Overview** dashboard from two datasources:
Prometheus (live request/LLM metrics) and Titan's Postgres (scan freshness,
scan duration, ticker coverage, daily LLM spend - data written by several
processes, so the DB is the source of truth). Alert rules live in
`observability/prometheus/alerts.yml`. If the API runs on a port other than
8000, update the `titan-api` target in `observability/prometheus/prometheus.yml`.

## Signal Backtest

```bash
python -m scripts.backtest_signal            # 7/14/28-day forward horizons
```

Uses the price stored with every `factor_scores` row to measure whether the
ranking has predictive power: Spearman IC, top-minus-bottom quintile spread,
top-quintile hit rate, and excess return per rating. Short history and
overlapping windows make it a regression metric, not proof of alpha.

## Benchmarks

Measured locally (Postgres + Redis in Docker, single uvicorn worker) on
2026-11-08 to 2026-11-10 and read back through Grafana. These come from
synthetic workloads described below, not production traffic.

### Read API

1,500 GET requests, 8 concurrent, with Zipf-skewed ticker popularity (a few
tickers requested far more often than the rest).

| Route | Cached | p50 | p95 | p99 |
|---|---|---|---|---|
| `/rankings` | yes | <10ms | 10-25ms | ~48ms |
| `/company/{ticker}` | yes | <10ms | 10-25ms | ~42ms |
| `/company/{ticker}/report` | no | <10ms | 10-25ms | ~41ms |
| `/compare` | no | <10ms | 10-25ms | ~38ms |

- **Redis hit rate: 82.6%** (945 hits / 199 misses).
- Uncached routes are as fast as cached ones locally - Postgres answers these
  queries in a few ms, so Redis matters mainly under higher load or with a
  remote database.
- Percentiles come from Prometheus histogram buckets, so they're ranges, not exact values.

### AI Chat

30 `/chat` requests: 22 unique questions (single-ticker, comparison,
thematic, earnings), 6 deliberate repeats, and 2 follow-ups within one
session. LLM latencies are exact, from the `llm_calls` table.

| Measure | p50 | p95 |
|---|---|---|
| Chat generation call (`claude-sonnet-5`, n=46) | 1.84s | 4.95s |
| Intent classification call (Haiku 4.5, n=24) | 0.60s | 0.86s |
| `/chat` end-to-end, answer generated (n=24) | 4.6s | 7.0s |
| `/chat` end-to-end, response-cache hit (n=6) | 24ms | 31ms |

- **Model round-trips per answer:** 1.92 on average (classifier, tool call, final answer).
- **Cost:** $0.36 for the run, about $0.015 per generated answer. Input tokens
  are 93% of it (147k in vs 9k out), because the system prompt and tool
  definitions are re-sent on every call. Prompt caching isn't used yet, which
  makes it the main lever for cutting cost.
- **Chat cache hit rate:** 20% in this run, but that reflects the deliberate
  repeats. Track the real rate on the Grafana "Response cache hit ratio" panel.
- **Grounding:** 17 of 22 unique questions were answered from Titan's data with
  citations. The 3 thematic questions failed because `VOYAGE_API_KEY` wasn't
  set. The 2 earnings questions correctly reported no ingested transcripts,
  and the sector question correctly reported there's no tool for it.

Reproduce: start the API and the observability stack, send chat traffic,
then query the `llm_calls` table:

```sql
SELECT call_type,
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
FROM llm_calls GROUP BY call_type;
```

### Signal Quality

`scripts/backtest_signal.py` over 20 full-universe scans (Jul 29 - Sep 17, 2026):

| Horizon | Mean IC | Top - bottom quintile | Top-quintile hit rate |
|---|---|---|---|
| 7 days | +0.028 | -1.34% | 51% |
| 14 days | +0.003 | -1.51% | 55% |
| 28 days | +0.028 | -2.10% | 56% |

The long-short spread is negative at every horizon, and SELL-rated names
beat STRONG BUY at 7 and 14 days and had the best excess return at 28 days.
On this short history the ranking shows no edge.

## Known Issues

- **Failed answers get cached.** When a tool call fails (e.g. memo search
  without `VOYAGE_API_KEY`), the "tool is unavailable" answer is still
  written to the response cache. It keeps being served until the next scan,
  even after the cause is fixed.
- **Follow-up answers have no citations.** A follow-up like "what about its
  momentum?" is often answered from conversation history without a tool call,
  so it cites real numbers but returns 0 sources. The UI shows no Sources
  expander for it, and Grafana counts it as ungrounded.
- **Empty retrieval data until keys are set.** Without `VOYAGE_API_KEY`,
  `memo_embeddings` stays empty and both search tools fail. After setting it,
  run `python -m scripts.backfill_memo_embeddings`. Without
  `EARNINGS_PROVIDER_API_KEY`, no earnings transcripts are ingested.
- **Stuck scan rows.** A crashed scan can leave a `scan_runs` row in `running`
  status forever. It stops blocking new scans after 2 hours but is never
  marked failed. The Grafana "Stuck 'running' scans" panel shows them.
- **Full scans always end `partial`.** 1-2 tickers fail to fetch from
  yfinance on every run.

## Tests

```bash
pytest
```

## Notes

- `app.py` disables SSL certificate verification globally as a
  workaround for "Certificate Verify Failed" errors some Mac/Python
  installs hit calling Yahoo Finance/Wikipedia. This is a local-dev
  convenience only - don't rely on it for anything security-sensitive.
- The "Thread Power" slider controls how many tickers are scanned
  concurrently (higher = faster, higher risk of being rate-limited by
  Yahoo Finance).
- This tool is for informational/educational purposes only and is not
  financial advice.
