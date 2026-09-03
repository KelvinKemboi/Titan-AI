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
│   │   ├── explain.py          # Factor Score Explanation Engine (per-factor score/weight/driver)
│   │   ├── memo_indexing.py    # Embeds + upserts each scan's analyst memos (memo_embeddings)
│   │   └── memo_search.py      # Cosine-similarity search over memo_embeddings
│   ├── api/
│   │   ├── main.py             # FastAPI app + router registration
│   │   ├── config.py           # Settings sourced from env vars (API_HOST, API_PORT, ...)
│   │   ├── deps.py             # Shared FastAPI dependencies (e.g. get_db)
│   │   ├── auth.py             # get_current_user: MVP API-key auth, reads API_KEYS
│   │   └── routes/
│   │       ├── health.py       # GET /health - DB connectivity check
│   │       ├── rankings.py     # GET /rankings
│   │       ├── company.py      # GET /company/{ticker}
│   │       └── compare.py      # GET /compare
│   ├── agents/
│   │   ├── chat_service.py     # Chat/Agent Service: Claude tool-calling loop + conversation memory
│   │   ├── entity_tracker.py   # extract_entities: tickers/factors mentioned, for pronoun resolution
│   │   ├── intent_classifier.py # classify_intent: cheap/fast routing hint (structured|qualitative|comparison)
│   │   └── tools/
│   │       ├── __init__.py     # Aggregates every tool submodule's TOOLS/DISPATCH into one registry
│   │       ├── base.py         # Shared tool contract: every tool returns source metadata
│   │       ├── factor_tools.py # get_factor_scores, compare_tickers Claude tool schemas
│   │       └── memo_tools.py   # search_memos Claude tool schema (qualitative retrieval)
│   ├── embeddings/
│   │   └── service.py          # embed_text/embed_texts: Voyage AI, batched + retried
│   ├── earnings/
│   │   └── provider_client.py  # get_transcript/search_transcripts: wraps api-ninjas.com (auth, retries, pagination)
│   └── data/
│       ├── models.py       # SQLAlchemy models: companies, scan_runs, factor_scores, memo_embeddings
│       ├── db.py           # Engine/session, reads DATABASE_URL
│       ├── cache.py        # Redis cache for /rankings + /company/{ticker}, reads REDIS_URL
│       └── migrations/     # Alembic migrations
├── scripts/
│   ├── manual_test_tool_calling.py  # Live Claude tool-use verification
│   └── manual_test_earnings_provider.py  # Live api-ninjas.com verification
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

The `db` service image is `pgvector/pgvector:pg16` - upstream `postgres:16`
with the [pgvector](https://github.com/pgvector/pgvector) extension
precompiled, needed for `vector(N)` columns and similarity search
(Phase 2's `earnings_chunks`, see
[docs/architecture.md](docs/architecture.md#4-database-schema-initial)). The
`c902a842d2a8_enable_pgvector_extension` migration runs `CREATE EXTENSION
IF NOT EXISTS vector` as part of `alembic upgrade head` above - no separate
step needed. If you're pointing `DATABASE_URL` at a Postgres instance other
than the `db` service, it must have pgvector installed for that migration
to succeed.

## Redis (caching)

`GET /rankings` and `GET /company/{ticker}` (`src/api/routes/`) cache their
reads through `src/data/cache.py`, invalidated on scan completion rather
than a fixed TTL (architecture.md §8) - `scanner_service.run_scan_and_persist`
calls `invalidate_scan_caches(tickers)` once a scan's results are committed,
clearing every `/rankings` sort-order key plus one `/company/{ticker}` key
per ticker the scan actually updated.

```bash
# Start local Redis (maps to host port 6380 to avoid clashing with any
# Redis you already have on 6379)
docker compose up -d redis
```

By default the app connects to `redis://localhost:6380/0`. Override with
the `REDIS_URL` environment variable. Redis is optional for correctness,
not just for local dev: every cache operation degrades to "no cache" (a
DB read) on any failure - unreachable, timeout, whatever - so an outage
means slower responses, never a 500. There's nothing to migrate or seed;
the first read after `docker compose up -d redis` just populates the
cache normally.

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

curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer alice-key-123" -H "Content-Type: application/json" \
  -d '{"message": "Why is AAPL rated a BUY?"}'
# {"response": "...", "sources": [...], "session_id": "<uuid>"}
# 401 without a valid Authorization header - see "Auth" below. Omit
# session_id to start a new conversation; pass a previous session_id back
# to continue it - as long as it's yours (see "Auth").
```

### Auth

Every `/chat` request needs `Authorization: Bearer <key>` - see
`src/api/auth.py:get_current_user`. This is a deliberately minimal MVP
scheme (a fixed set of keys sourced from one env var), not a placeholder
for something more built out in that same file later; swapping in a real
identity provider means replacing `get_current_user`'s body; every
route just depends on it and gets a `user_id` string back, unchanged.

```bash
# .env - "user_id:key" pairs, comma-separated
API_KEYS=alice:alice-key-123,bob:bob-key-456
```

Unset/empty `API_KEYS` means *no* key validates - the API fails closed
(every `/chat` request 401s) rather than open when misconfigured, never
silently allowing unauthenticated access.

The resolved `user_id` is what `chat_sessions.user_id` gets set to
(`src/data/chat_repository.py:get_or_create_session`) - replacing what
used to be an always-empty placeholder column, so sessions are now
actually scoped per caller rather than shared/anonymous. Passing a
`session_id` that exists but belongs to a *different* `user_id` is a 403,
not a silent read of someone else's conversation - one user must never be
able to see (or extend) another's chat history this way.

### Citation UI

Each assistant message in the sidebar chat (`app.py`) renders its
`sources` as a collapsed `Sources (N)` expander underneath the response -
present but out of the way, not a wall of citations under every message.
Expanding it renders each source's `detail` (see above) in a way
specific to its `type`, so a claim can actually be checked instead of
just naming the ticker/scan it came from:

- `factor_score` sources show the composite score/rating and, when
  present, the full per-factor breakdown (score, weight, contribution,
  driver) `get_factor_scores` computed; `compare_tickers`' sources carry
  each ticker's own five raw factor scores instead (no weight/
  contribution/driver - the comparison tool never computed those).
- `memo` sources show the actual memo text `search_memos` matched
  against - the qualitative claim's real evidence, not a paraphrase.
- Any source with no `detail` (or an unrecognized `type`) falls back to
  the ticker/scan/date line alone, so older cached responses or a future
  source type without `detail` still render instead of erroring.

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
[docs/technical-design.md §5](docs/technical-design.md). Each `Source`
also carries an optional `detail` - a bag of whatever verification data
the tool already computed (a factor_score source's composite/rating/
per-factor breakdown, a memo source's actual memo text) - so the
citation UI below can show it without a second fetch.

```python
from src.agents.tools import TOOLS, call_tool  # aggregates every tools/*.py submodule
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
`src/agents/tools/__init__.py` is the registry: it merges `TOOLS`/`DISPATCH`
from every submodule (`factor_tools.py`, `memo_tools.py` below, ...) so
adding a new tool category doesn't touch `chat_service.py`.

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

### Memo Search (qualitative retrieval)

`src/agents/tools/memo_tools.py` defines `search_memos(query)`: semantic
search over `memo_embeddings` - one embedded `RoboAnalyst.generate_memo()`
writeup per ticker per scan, Titan's first qualitative retrieval source
(alongside the numeric `get_factor_scores`/`compare_tickers` above). Use
it for questions with no single named ticker to key off of - "which
companies have a deep competitive moat" - where the structured tools
have nothing to look up.

`src/analytics/memo_indexing.py:index_memos()` embeds + upserts every
scan's memos automatically as the last step of
`scanner_service.run_scan_and_persist()` (embedding failures are logged
and swallowed there, not raised - a scan's factor scores still count as
persisted even if Voyage is briefly unreachable). For data scanned
before this existed, backfill the latest run instead of re-scanning:

```bash
export VOYAGE_API_KEY="..."
python -m scripts.backfill_memo_embeddings
```

Manual live-model verification that a qualitative question actually
routes to `search_memos` rather than the structured tools (requires
`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, and at least one scan with indexed
memos):

```bash
python -m scripts.manual_test_memo_search
```

### Earnings Transcript Provider

`src/earnings/provider_client.py` wraps api-ninjas.com's Earnings Call
Transcript API - selected over Financial Modeling Prep and Alpha Vantage;
see the written comparison in
[docs/technical-design.md §7](docs/technical-design.md). Ingestion code
(a later issue) only ever sees this module's own `Transcript`/
`TranscriptSearchResult` types and `get_transcript`/`search_transcripts`
functions, never the provider's request/response shapes - swapping
providers later means rewriting this module's internals, not its callers.

```python
from src.earnings.provider_client import get_transcript, search_transcripts

for hit in search_transcripts("AAPL"):  # paginated (offset/limit=50), most recent first
    print(hit.fiscal_year, hit.fiscal_quarter, hit.call_date)

transcript = get_transcript("AAPL", year=2024, quarter=2)  # None if the provider has nothing for this quarter
```

```bash
# .env
EARNINGS_PROVIDER_API_KEY=...  # api-ninjas.com key
```

Auth (`X-Api-Key` header) and rate limiting (429/5xx retried with
backoff, honoring a `Retry-After` header when the provider sends one) are
handled internally - `EarningsProviderAuthError` for a missing/rejected
key (fails the whole run rather than retrying, since a bad key fails
identically every time), `EarningsProviderError` for anything else.
`get_transcript` returns `None` - not an error - when the provider simply
has no transcript for a ticker/quarter, so one coverage gap degrades
gracefully instead of failing an entire ingestion run.

Manual live verification (requires `EARNINGS_PROVIDER_API_KEY`):

```bash
export EARNINGS_PROVIDER_API_KEY="..."
python -m scripts.manual_test_earnings_provider AAPL
```

### Intent Classification

`src/agents/intent_classifier.py:classify_intent(question)` is
technical-design.md §6 step 1: a cheap/fast model call
(`claude-haiku-4-5-20251001`, forced via `tool_choice` into a
`structured | qualitative | comparison` enum) that runs once per turn,
before the Chat/Agent Service decides which tools to foreground.

This is a **hint, not a gate**: `answer_question()` folds the result into
the system prompt (`INTENT_HINTS`) but always passes every tool
(`get_factor_scores`, `compare_tickers`, `search_memos`) regardless of
what was classified - per the risk technical-design.md §6 calls out,
a wrong guess must never make a tool unreachable. Any classifier failure
(API error, rate limit, a missing key) returns `None`, and the turn
proceeds with the plain system prompt rather than blocking.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python -m scripts.eval_intent_classifier
```

Runs `scripts/eval_intent_classifier.py`'s ~30 hand-written questions (10
per category, including the three phase-1 example questions from the
product brief) against the live classifier and reports per-category
accuracy plus latency (mean/median/min/max) - re-run this after any prompt
or model change.

### Conversation Memory (entity tracking)

`src/agents/entity_tracker.py:extract_entities(history)` is
technical-design.md §4's missing piece: the last-K-turns window
(`CHAT_HISTORY_WINDOW`) already gives the model raw context for free, but
nothing that reliably grounds a pronoun follow-up ("what about its
momentum?"). This scans a session's `history` for tickers - from each
assistant turn's already-verified `sources`, not regex/NLP over free text,
so a ticker Titan never actually looked up can't be mis-tracked - and
factors (matched by name against the closed `WEIGHTS` vocabulary). The
result is an `EntityState` (`last_ticker`, `last_factor`, plus full
mention history) that `answer_question()` folds into the system prompt as
a one-line grounding hint (`entity_hint()`) - the same advisory pattern as
intent classification above: it biases resolution, it never replaces the
model's own reasoning.

`EntityState` is a flat pydantic model - `model_dump_json()` is the whole
debug view - and `answer_question()` logs it every turn, so an incorrect
pronoun resolution is debuggable from the logs after the fact: what did
the system think "it" referred to, and why.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python -m scripts.eval_conversation_memory
```

Runs `scripts/eval_conversation_memory.py`'s 5 hand-written 3-turn
conversations (ticker question → pronoun follow-up → comparison) through
the real Chat/Agent Service with real Postgres persistence between turns,
checking each turn's *sources* - not just response text - cite the
ticker(s) the pronoun should have resolved to, and prints the entity
state before each turn for debugging.

### Chat Response Cache

FAQ-style questions ("what are the strongest momentum stocks today") get
re-asked by many different sessions between scans, so `answer_question()`
(`src/agents/chat_service.py`) caches the full response - text and
sources together - through `src/data/cache.py:chat_cache_key(question,
data_version)`, where `data_version` is the latest `scan_run_id`. A cache
hit returns immediately, before intent classification, entity extraction,
or any model call. Only a genuinely successful, model-completed answer is
cached - never `FALLBACK_RESPONSE` (a transient API failure) or
`INCONCLUSIVE_RESPONSE` (an unresolved tool-call loop); neither is a fact
worth reusing.

**Read this before touching the cache path**: it is keyed *only* on the
question's own text plus the data version - nothing session-specific.
That's only safe for a context-free question, i.e. the first turn of a
session with no prior `history`. A follow-up ("what about its momentum?")
depends on which ticker/factor came up earlier in that specific session
(entity tracking, above) - two sessions can phrase a follow-up
identically while meaning two different tickers, and a key built from
the text alone can't distinguish them. `answer_question()` therefore
never computes or uses this cache key when `history` is non-empty - full
stop, no exceptions, no attempt to make the key "smart enough" to
disambiguate. This is the one thing the caching design's own acceptance
criteria calls out as easy to get subtly wrong; see
`test_questions_with_history_never_touch_the_cache` in
`tests/unit/agents/test_chat_service.py` for the regression test that
locks it in.

Unlike the rankings/company cache above, there's no explicit invalidation
call: a question can be any string, so there's no fixed, enumerable key
set to bulk-delete the way there is for rankings' handful of sort orders.
Instead, `data_version` is embedded directly in the key - a new scan
changes it, and every key from the old version simply becomes
unreachable, aging out via the same TTL backstop as everything else in
`src/data/cache.py`.

## Embeddings

`src/embeddings/service.py` is the one place text gets turned into a
vector - both this milestone's pgvector setup and Phase 2's earnings-chunk
ingestion (docs/architecture.md #2, #6) call it rather than hitting an
embeddings API directly.

```python
from src.embeddings.service import embed_text, embed_texts

vector = embed_text("NVDA's Q3 guidance raised on strong datacenter demand")
vectors = embed_texts(chunks, input_type="document")  # batched automatically, any length
query_vector = embed_texts([question], input_type="query")[0]  # tuned for search, not storage
```

Model/dimension: [Voyage AI](https://www.voyageai.com/) - Anthropic's own
recommended embeddings partner, since Anthropic doesn't host embedding
models itself - `voyage-large-2`, chosen specifically because it natively
outputs 1536-dimensional vectors matching `vector(1536)` on
`earnings_chunks.embedding` (docs/architecture.md #4) with no
`output_dimension` override required. Requires a `VOYAGE_API_KEY`
environment variable (get one at
[dash.voyageai.com](https://dash.voyageai.com/)); batching (up to 128 texts
per request, the API's own cap) and retry (rate limits/timeouts, exponential
backoff) are handled internally.

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
