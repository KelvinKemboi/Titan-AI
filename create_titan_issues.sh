#!/usr/bin/env bash
set -euo pipefail

# Creates Titan AI GitHub labels, milestones, and issues from the project roadmap.
# Run this from inside the Titan-AI repo after: gh auth login

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI not found. Install gh first: https://cli.github.com/" >&2
  exit 1
fi

if ! gh repo view >/dev/null 2>&1; then
  echo "Run this script from inside a GitHub repo folder, or set GH_REPO=owner/repo." >&2
  exit 1
fi

ensure_label() {
  local name="$1"
  local color="$2"
  gh label create "$name" --color "$color" 2>/dev/null || true
}

ensure_milestone() {
  local title="$1"
  local description="$2"
  if ! gh api repos/:owner/:repo/milestones --jq ".[ ].title" | grep -Fxq "$title"; then
    gh api repos/:owner/:repo/milestones -f title="$title" -f description="$description" >/dev/null
  fi
}

echo "Creating labels..."
ensure_label api BFDADC
ensure_label backend 5319E7
ensure_label chat C5DEF5
ensure_label data F9D0C4
ensure_label database 006B75
ensure_label earnings D93F0B
ensure_label frontend FBCA04
ensure_label infra 5319E7
ensure_label observability 5319E7
ensure_label phase-1 1D76DB
ensure_label phase-2 D93F0B
ensure_label security D73A4A
ensure_label testing 0E8A16

echo "Creating milestones..."
ensure_milestone 'AI Investment Chat MVP' 'Initial Titan chat over rankings, factor scores, company fundamentals, and basic source attribution.'
ensure_milestone 'AI Investment Chat V2' 'RAG, embeddings, caching, entity tracking, auth, and improved source attribution.'
ensure_milestone 'Earnings Intelligence MVP' 'Transcript ingestion, summarization, guidance extraction, sentiment, risks, and earnings chat tools.'
ensure_milestone 'Earnings Intelligence V2' 'QoQ comparison, analyst reports, scheduled ingestion, backfills, evals, and observability.'

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

create_issue() {
  local title="$1"
  local labels="$2"
  local milestone="$3"
  local body_file="$4"
  echo "Creating: $title"
  gh issue create --title "$title" --body-file "$body_file" --label "$labels" --milestone "$milestone"
}

cat > "$tmpdir/issue_3.md" <<'EOF'
Difficulty: M

## Description
Wire the extracted scan function (#2) to persist results: one
`scan_runs` row per run, one `factor_scores` row per ticker, upserting
`companies` from the fundamentals already fetched in `RoboAnalyst.info`.

## Acceptance Criteria
- Running a scan creates exactly one `scan_runs` row with correct
  `started_at`/`completed_at`/`status`
- One `factor_scores` row per successfully-analyzed ticker, `raw_metrics`
  populated from `RoboAnalyst.metrics`
- `companies` rows created/updated with sector, industry, description
  from `yfinance` `info`
- Partial failures (some tickers fail `analyze()`) don't fail the whole
  run — `scan_runs.status` reflects partial success accurately

## Dependencies
#1, #2
EOF
create_issue 'Scanner Service writes scan results to Postgres' backend,database,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_3.md"

cat > "$tmpdir/issue_4.md" <<'EOF'
Difficulty: S

## Description
Run the Scanner Service on a recurring schedule (hourly, matching today's
`st.cache_data(ttl=3600)` cadence) instead of only on manual trigger.

## Acceptance Criteria
- Scheduled job (Celery beat, APScheduler, or platform cron) triggers a
  full scan on the configured interval
- Overlapping runs are prevented (a run in progress blocks a new one from
  starting)
- Failure of a scheduled run is logged/alertable, not silent

## Dependencies
#3
EOF
create_issue 'Schedule Scanner Service runs' infra,backend,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_4.md"

cat > "$tmpdir/issue_5.md" <<'EOF'
Difficulty: S

## Description
Stand up the API Gateway: project structure, config/env handling, health
check endpoint, DB connection pooling, and a place to add the routes in
the following issues.

## Acceptance Criteria
- `GET /health` returns 200 with DB connectivity check
- App runs locally via a documented command
- Config (DB URL, etc.) sourced from environment variables, not hardcoded

## Dependencies
#1
EOF
create_issue 'Scaffold FastAPI gateway service' api,backend,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_5.md"

cat > "$tmpdir/issue_6.md" <<'EOF'
Difficulty: S

## Description
Return the latest ranked list of tickers by composite score, optionally
filtered/sorted by a single factor.

## Acceptance Criteria
- `GET /rankings` returns latest `scan_runs`' `factor_scores`, sorted by
  `composite_score` desc
- `?factor=momentum` sorts by that factor's score instead
- Returns `scan_run_id` and `computed_at` so clients know data freshness
- Empty/no-scan-yet state returns an empty list, not an error

## Dependencies
#3, #5
EOF
create_issue '`GET /rankings` endpoint' api,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_6.md"

cat > "$tmpdir/issue_7.md" <<'EOF'
Difficulty: S

## Description
Return a company's profile plus its latest factor scores.

## Acceptance Criteria
- Returns `companies` row joined with latest `factor_scores` row
- Unknown ticker returns 404, not a 500 or empty 200
- Response includes `raw_metrics` (RSI, PEG, beta, etc.) for downstream
  explanation use

## Dependencies
#3, #5
EOF
create_issue '`GET /company/{ticker}` endpoint' api,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_7.md"

cat > "$tmpdir/issue_8.md" <<'EOF'
Difficulty: M

## Description
Implement the Comparison Engine (technical-design.md §3) as a service
function, exposed via `GET /compare?tickers=MSFT,GOOGL`.

## Acceptance Criteria
- Accepts 2+ tickers, returns aligned per-factor scores for each
- Response includes factor deltas sorted by `abs(delta)` descending
- Both tickers pinned to the same `scan_run_id`; if one is missing from
  the latest run, response says so explicitly rather than silently
  comparing mismatched runs
- 400 for fewer than 2 valid tickers

## Dependencies
#6
EOF
create_issue '`GET /compare` endpoint + Comparison Engine' api,backend,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_8.md"

cat > "$tmpdir/issue_9.md" <<'EOF'
Difficulty: M

## Description
Implement technical-design.md §2 as a callable function returning a
structured explanation (per-factor score, weight, contribution,
driving metric) — this becomes an LLM tool in #11.

## Acceptance Criteria
- Given a ticker, returns each factor's score, its `WEIGHTS` value, its
  contribution to composite, and a one-line description of what drove it
  (reusing the same thresholds `RoboAnalyst` scores against)
- Output is structured data (typed fields), not prose — prose generation
  is the LLM's job downstream, not this function's
- Unit tested against known `factor_scores` fixtures with expected
  explanation output

## Dependencies
#3
EOF
create_issue 'Factor Score Explanation Engine (tool)' backend,chat,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_9.md"

cat > "$tmpdir/issue_10.md" <<'EOF'
Difficulty: M

## Description
Define the Claude tool-use schemas for `get_factor_scores(ticker)` and
`compare_tickers(tickers)`, backed by #8 and #9.

## Acceptance Criteria
- Tool schemas defined with typed inputs/outputs matching #8/#9's return
  shape
- Each tool result includes source metadata (`scan_run_id`, `ticker`,
  `as_of`) per the source-attribution contract (technical-design.md §5)
- Manual test: prompting the model with a comparison question results in
  it calling `compare_tickers`, not fabricating numbers

## Dependencies
#8, #9
EOF
create_issue 'Wire Explanation Engine + Comparison Engine as LLM tools' chat,backend,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_10.md"

cat > "$tmpdir/issue_11.md" <<'EOF'
Difficulty: L

## Description
Build the service that owns a single-turn conversation: takes a user
message, calls the LLM with the tools from #10, executes tool calls,
feeds results back, returns a final response with attached sources.

## Acceptance Criteria
- Given a question answerable by one tool call, returns a correct,
  sourced response end-to-end
- System prompt enforces "don't use outside knowledge for anything Titan
  has data for" (technical-design.md §6) — verified with a question about
  a well-known public fact the model would otherwise answer from training
  data (e.g. "what sector is Apple in" should use the tool result, not
  general knowledge)
- Tool call failures (e.g. unknown ticker) produce a graceful response,
  not a raw exception surfaced to the user

## Dependencies
#10
EOF
create_issue 'Chat/Agent Service scaffold with tool-calling loop' chat,backend,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_11.md"

cat > "$tmpdir/issue_12.md" <<'EOF'
Difficulty: S

## Description
Expose the Chat/Agent Service (#11) via the API Gateway.

## Acceptance Criteria
- `POST /chat` with `{session_id?, message}` returns `{response, sources,
  session_id}`
- Omitted `session_id` creates a new session
- Response time budget documented (even if not yet optimized) so V2
  caching work (#20) has a baseline to improve on

## Dependencies
#5, #11
EOF
create_issue '`POST /chat` endpoint' api,chat,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_12.md"

cat > "$tmpdir/issue_13.md" <<'EOF'
Difficulty: M

## Description
Implement technical-design.md §4: persist every turn, inject the last-K
window into subsequent Chat/Agent Service calls.

## Acceptance Criteria
- Migration for `chat_sessions`, `chat_messages` (architecture.md §4)
- Every `/chat` call persists both the user message and assistant
  response, including `sources`
- A follow-up question in the same session correctly resolves context
  from prior turns (manual test: ask about NVDA, then ask "what about its
  momentum?")
- Window size (K) is a config value, not hardcoded inline

## Dependencies
#12
EOF
create_issue 'Conversation memory: `chat_sessions` / `chat_messages` persistence' database,chat,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_13.md"

cat > "$tmpdir/issue_14.md" <<'EOF'
Difficulty: M

## Description
Add a chat UI to the existing `app.py`, calling the new `/chat` API. This
is additive — the existing scanner UI/flow is untouched.

## Acceptance Criteria
- New sidebar or tab with a chat input + message history display
- Renders `sources` under each assistant response (even minimally, e.g.
  "Source: MSFT, scan #142")
- Session ID persisted in `st.session_state` across reruns within a
  browser session
- Existing "Initialize Market Scan" flow unaffected

## Dependencies
#12
EOF
create_issue 'Chat panel in Streamlit app' frontend,phase-1 'AI Investment Chat MVP' "$tmpdir/issue_14.md"

cat > "$tmpdir/issue_15.md" <<'EOF'
Difficulty: M

## Description
Enable `pgvector` on Postgres and build a reusable embedding
helper/service (model choice, batching, retry) that both this milestone
and Phase 2 ingestion will call.

## Acceptance Criteria
- Migration enables the `vector` extension
- Embedding helper function: text in, vector out, with batching for
  multiple inputs
- Documented embedding model/dimension choice (must match the `vector(N)`
  column width used in schema)

## Dependencies
#1
EOF
create_issue 'Add pgvector extension + embedding pipeline infrastructure' infra,database,phase-1 'AI Investment Chat V2' "$tmpdir/issue_15.md"

cat > "$tmpdir/issue_16.md" <<'EOF'
Difficulty: M

## Description
Embed and index the existing `generate_memo` output per ticker/scan as
the first qualitative retrieval source, and add a `search_memos(query)`
tool.

## Acceptance Criteria
- Memos embedded on scan completion (or backfilled for latest run)
- `search_memos(query)` tool returns top-k memo snippets with source
  metadata (`ticker`, `scan_run_id`)
- Manual test: a qualitative question ("which companies have deep
  competitive moats") returns relevant tickers via this path, not the
  structured tools

## Dependencies
#15, #3
EOF
create_issue 'Vector search over generated memos' backend,chat,phase-1 'AI Investment Chat V2' "$tmpdir/issue_16.md"

cat > "$tmpdir/issue_17.md" <<'EOF'
Difficulty: M

## Description
Implement technical-design.md §6 step 1: a fast/cheap model call that
classifies each incoming message before the Chat/Agent Service decides
which tools to foreground.

## Acceptance Criteria
- Classifier returns one of `structured | qualitative | comparison` with
  reasonable accuracy on a hand-written eval set (~30 example questions
  covering all three phase-1 example questions from the product brief)
- Structured tools remain callable regardless of classification (per the
  risk noted in technical-design.md §6) — classification informs routing,
  it doesn't hard-gate tool availability
- Classification latency doesn't dominate total response time (measure
  and document)

## Dependencies
#11, #16
EOF
create_issue 'Intent classifier for structured vs. qualitative vs. comparison routing' chat,backend,phase-1 'AI Investment Chat V2' "$tmpdir/issue_17.md"

cat > "$tmpdir/issue_18.md" <<'EOF'
Difficulty: M

## Description
Improve conversation memory (#13) to explicitly track entities (tickers,
factors) mentioned in-session so ellipsis/pronoun follow-ups resolve
reliably, beyond what a raw last-K-turns window gives for free.

## Acceptance Criteria
- Eval set of multi-turn conversations (ask about a ticker, then a
  pronoun follow-up, then a comparison) passes with correct entity
  resolution
- Entity state is inspectable/loggable for debugging incorrect
  resolutions

## Dependencies
#13
EOF
create_issue 'Multi-turn entity tracking for follow-up questions' chat,phase-1 'AI Investment Chat V2' "$tmpdir/issue_18.md"

cat > "$tmpdir/issue_19.md" <<'EOF'
Difficulty: S

## Description
Add Redis, cache `/rankings` and `/company/{ticker}` reads, invalidated
on new `scan_runs` completion (architecture.md §8).

## Acceptance Criteria
- Cache hit avoids a Postgres round-trip for repeated reads within the
  scan interval
- Scan completion (#3/#4) invalidates the relevant cache keys
- Cache unavailability degrades to direct DB reads, not an error

## Dependencies
#6, #7
EOF
create_issue 'Redis cache for rankings + factor scores' infra,backend,phase-1 'AI Investment Chat V2' "$tmpdir/issue_19.md"

cat > "$tmpdir/issue_20.md" <<'EOF'
Difficulty: M

## Description
Cache `/chat` responses keyed on `hash(question) + data_version` so FAQ-
style questions ("what are the strongest momentum stocks today") don't
re-run the full tool-calling loop every time within a scan interval.

## Acceptance Criteria
- Identical question + unchanged underlying data returns the cached
  response with cached `sources`
- Cache is invalidated when the `scan_run_id` (or relevant
  `earnings_insights`, once Phase 2 lands) it drew from changes
- Cache key strategy documented (this is easy to get subtly wrong —
  session-specific follow-ups must not be cached as if they were generic
  questions)

## Dependencies
#19, #13
EOF
create_issue 'LLM response cache for repeated chat questions' infra,chat,phase-1 'AI Investment Chat V2' "$tmpdir/issue_20.md"

cat > "$tmpdir/issue_21.md" <<'EOF'
Difficulty: M

## Description
Replace the placeholder `chat_sessions.user_id` with real identity so
sessions/history are actually scoped per user, not shared/anonymous.

## Acceptance Criteria
- API requires an auth token (even a simple API-key scheme for MVP is
  acceptable — full OAuth is not required at this stage)
- `chat_sessions.user_id` reflects the authenticated caller
- A user cannot read another user's `chat_sessions`/`chat_messages`

## Dependencies
#12
EOF
create_issue 'Basic auth / session identity for chat' api,security,phase-1 'AI Investment Chat V2' "$tmpdir/issue_21.md"

cat > "$tmpdir/issue_22.md" <<'EOF'
Difficulty: S

## Description
Improve #14's minimal source rendering into a proper citation UI (e.g.
expandable "Sources" section per message, linking factor/ticker/date).

## Acceptance Criteria
- Each assistant message shows its sources in a readable, non-intrusive
  way
- Clicking/expanding a source shows enough detail to verify the claim
  (e.g. the actual factor score value and scan date)

## Dependencies
#14, #10
EOF
create_issue 'Render source attribution in Streamlit chat panel' frontend,phase-1 'AI Investment Chat V2' "$tmpdir/issue_22.md"

cat > "$tmpdir/issue_23.md" <<'EOF'
Difficulty: M

## Description
`yfinance` does not provide earnings transcripts. Evaluate providers
(Financial Modeling Prep, AlphaVantage, a dedicated transcript vendor),
select one, and build the API client.

## Acceptance Criteria
- Written comparison of at least 2 candidate providers (coverage, cost,
  rate limits, S&P 500 completeness)
- Client wraps the chosen provider's API with our own interface (so
  swapping providers later doesn't ripple through the ingestion code)
- Client handles auth, rate limiting, and pagination for the chosen
  provider

## Dependencies
None
EOF
create_issue 'Select and integrate an earnings transcript data provider' infra,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_23.md"

cat > "$tmpdir/issue_24.md" <<'EOF'
Difficulty: S

## Description
Migration for the two tables per architecture.md §4.

## Acceptance Criteria
- Migration creates `earnings_transcripts`, `earnings_chunks` with the
  `vector` column sized to match #15's embedding dimension
- Foreign key from `earnings_chunks.transcript_id` to
  `earnings_transcripts.id`

## Dependencies
#1, #15
EOF
create_issue 'Add `earnings_transcripts` / `earnings_chunks` schema' database,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_24.md"

cat > "$tmpdir/issue_25.md" <<'EOF'
Difficulty: M

## Description
Job that, given a ticker + quarter, fetches the raw transcript via #23's
client and stores it in `earnings_transcripts`.

## Acceptance Criteria
- Successful fetch creates one `earnings_transcripts` row with raw text
  and source URL
- Missing/unavailable transcript for a given ticker+quarter is handled
  gracefully (logged, not a hard failure of any batch job)
- Idempotent: re-running for an already-ingested ticker+quarter doesn't
  create a duplicate row

## Dependencies
#23, #24
EOF
create_issue 'Transcript ingestion job (fetch + store raw)' backend,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_25.md"

cat > "$tmpdir/issue_26.md" <<'EOF'
Difficulty: M

## Description
Split raw transcript text into `earnings_chunks` rows, tagged
`prepared_remarks` or `qna`, per technical-design.md §7.

## Acceptance Criteria
- Chunker correctly separates prepared remarks from Q&A for the
  transcript format the chosen provider (#23) returns
- Each chunk is small enough for embedding (documented max size) while
  preserving speaker/section boundaries
- Unit tested against at least 2 real sample transcripts

## Dependencies
#25
EOF
create_issue 'Chunking pipeline (by speaker turn / section)' backend,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_26.md"

cat > "$tmpdir/issue_27.md" <<'EOF'
Difficulty: S

## Description
Wire #26's chunks through #15's embedding helper, populating the
`embedding` column.

## Acceptance Criteria
- Every chunk row gets an embedding on ingestion
- Batch embedding used for a transcript's chunks (not one API call per
  chunk) for cost/latency

## Dependencies
#15, #26
EOF
create_issue 'Embed `earnings_chunks` using the shared embedding pipeline' backend,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_27.md"

cat > "$tmpdir/issue_28.md" <<'EOF'
Difficulty: M

## Description
Implement technical-design.md §8: LLM summarization pass over a
transcript's chunks, run at ingestion time.

## Acceptance Criteria
- `earnings_insights.summary` populated for each newly-ingested
  transcript
- Long transcripts exceeding practical context size are chunk-then-
  reduced (summarize sections, then summarize summaries) rather than
  truncated silently
- Runs async, off any user-facing request path

## Dependencies
#26
EOF
create_issue 'Earnings Summary Generation' backend,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_28.md"

cat > "$tmpdir/issue_29.md" <<'EOF'
Difficulty: M

## Description
Implement technical-design.md §9: structured extraction of guidance
direction with a supporting quote.

## Acceptance Criteria
- `earnings_insights.guidance_direction` populated as one of `raised /
  maintained / lowered / none_given / unclear`
- Supporting quote stored alongside the classification
- Eval set of transcripts with known guidance outcomes used to sanity-
  check extraction accuracy before merge

## Dependencies
#28
EOF
create_issue 'Guidance Extraction' backend,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_29.md"

cat > "$tmpdir/issue_30.md" <<'EOF'
Difficulty: S

## Description
Implement technical-design.md §10: rubric-based sentiment scoring over
Q&A chunks.

## Acceptance Criteria
- `earnings_insights.sentiment_score` populated in range -1..1
- Scoring rubric documented (what -1 vs. 0 vs. 1 looks like) so the
  prompt is auditable, not a black box
- Manual spot-check against 5+ known transcripts for plausibility

## Dependencies
#28
EOF
create_issue 'Sentiment Analysis' backend,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_30.md"

cat > "$tmpdir/issue_31.md" <<'EOF'
Difficulty: S

## Description
Implement technical-design.md §11: extraction of risks management
actually mentioned in the call, with supporting quotes.

## Acceptance Criteria
- `earnings_insights.risks` populated as a list of `{risk, quote}`
  objects
- Extraction explicitly scoped to what was said in this call (not
  general/10-K-style risk factors — that's out of scope, see
  future-expansion.md)

## Dependencies
#28
EOF
create_issue 'Risk Extraction' backend,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_31.md"

cat > "$tmpdir/issue_32.md" <<'EOF'
Difficulty: M

## Description
Expose ingested earnings data via the API and add `get_earnings_insight
(ticker, quarter?)` / `search_earnings(query)` as LLM tools, following the
same source-attribution contract as the Phase 1 tools (#9, #16).

## Acceptance Criteria
- `GET /earnings/{ticker}` returns transcripts + insights for that ticker
- New tools registered with the Chat/Agent Service; source metadata
  includes `transcript_id`
- Manual test: an earnings example question from the product brief
  ("summarize NVDA's latest earnings call") produces a correct, sourced
  answer end-to-end

## Dependencies
#28, #29, #30, #31, #11
EOF
create_issue '`GET /earnings/{ticker}` endpoint + wire earnings tools into Chat/Agent Service' api,chat,earnings,phase-2 'Earnings Intelligence MVP' "$tmpdir/issue_32.md"

cat > "$tmpdir/issue_33.md" <<'EOF'
Difficulty: M

## Description
Implement technical-design.md §12: structured diff between a ticker's
current and prior `earnings_insights` rows.

## Acceptance Criteria
- `earnings_insights.qoq_changes` populated with guidance-direction
  change, sentiment delta, and new/resolved risks (set difference on
  #31's extracted risk list)
- First-quarter-ingested tickers with no prior insight return
  `insufficient_history` explicitly rather than an empty/misleading diff

## Dependencies
#29, #30, #31
EOF
create_issue 'Quarter-over-Quarter Comparison' backend,earnings,phase-2 'Earnings Intelligence V2' "$tmpdir/issue_33.md"

cat > "$tmpdir/issue_34.md" <<'EOF'
Difficulty: M

## Description
Implement technical-design.md §13: extend the existing
`RoboAnalyst.generate_memo` template with a conditional "Recent Earnings"
section sourced from `earnings_insights`.

## Acceptance Criteria
- Tickers with earnings data get an added section (summary, guidance
  direction, top risk) in the memo
- Tickers without earnings data render exactly as today (no regression to
  existing memo output — verify against current `generate_memo` snapshot)
- Report available both via API and in the Streamlit report view

## Dependencies
#28, #29, #31
EOF
create_issue 'Automatic Analyst-Style Report combining factor scores + earnings insights' backend,earnings,phase-2 'Earnings Intelligence V2' "$tmpdir/issue_34.md"

cat > "$tmpdir/issue_35.md" <<'EOF'
Difficulty: M

## Description
Replace MVP's manual/on-demand ingestion trigger (#25) with scheduling
tied to each company's actual earnings date.

## Acceptance Criteria
- Earnings calendar source identified (may be the same provider as #23
  or a separate calendar API)
- Ingestion job for a ticker fires automatically within a bounded window
  after its earnings date
- Missed/delayed transcript availability (provider hasn't published yet)
  retries on a backoff schedule rather than failing permanently

## Dependencies
#25
EOF
create_issue 'Earnings calendar integration for scheduled ingestion' infra,earnings,phase-2 'Earnings Intelligence V2' "$tmpdir/issue_35.md"

cat > "$tmpdir/issue_36.md" <<'EOF'
Difficulty: L

## Description
One-time batch job to ingest historical transcripts so QoQ comparison
(#33) and chat questions about "last quarter" work immediately for the
existing universe, not only going forward.

## Acceptance Criteria
- Batch job processes the full S&P 500 ticker list (from
  `titan/data.py`), ingesting up to 4 trailing quarters per ticker
- Provider rate limits respected (throttled/batched, not a thundering
  herd against #23's client)
- Progress/failure per ticker is logged so partial completion is
  resumable, not restart-from-zero

## Dependencies
#25, #26, #27, #28, #29, #30, #31
EOF
create_issue 'Backfill 4 trailing quarters for the S&P 500 universe' backend,earnings,data,phase-2 'Earnings Intelligence V2' "$tmpdir/issue_36.md"

cat > "$tmpdir/issue_37.md" <<'EOF'
Difficulty: S

## Description
Add `get_qoq_changes(ticker)` and `get_analyst_report(ticker)` as LLM
tools, completing the Phase 2 example questions from the product brief
("what changed from last quarter?").

## Acceptance Criteria
- Both tools registered with source metadata per the attribution
  contract
- Manual test: all six Phase 2 example questions from the product brief
  produce correct, sourced answers end-to-end

## Dependencies
#33, #34, #32
EOF
create_issue 'Wire QoQ Comparison + Analyst Report as chat tools' chat,earnings,phase-2 'Earnings Intelligence V2' "$tmpdir/issue_37.md"

cat > "$tmpdir/issue_38.md" <<'EOF'
Difficulty: M

## Description
Formalize the ad-hoc "manual test" acceptance criteria scattered across
prior issues into a repeatable eval set covering both phases' example
questions, run in CI against every Chat/Agent Service change.

## Acceptance Criteria
- Eval set includes all example questions from the original product
  brief (Phase 1 and Phase 2) plus edge cases surfaced during
  implementation (unknown ticker, insufficient history, stale data)
- Each case has an expected answer shape (not necessarily exact text —
  e.g. "cites MSFT and GOOGL factor scores from the same scan_run_id")
- CI fails on regression, not just on crash

## Dependencies
#37
EOF
create_issue 'Build a golden Q&A eval set for chat accuracy regression testing' testing,chat,phase-2 'Earnings Intelligence V2' "$tmpdir/issue_38.md"

cat > "$tmpdir/issue_39.md" <<'EOF'
Difficulty: M

## Description
By this point the Chat/Agent Service calls multiple LLMs (intent
classifier, generation, earnings extraction) across multiple tools per
turn — add tracing so a bad answer or a cost spike is debuggable.

## Acceptance Criteria
- Every `/chat` request's tool calls, model calls, and token usage are
  logged/traceable end-to-end for that request
- Per-session and aggregate LLM spend is queryable (even a simple
  dashboard/report, not necessarily real-time)
- A sample "bad answer" can be traced back to which tool call(s)
  produced the incorrect grounding data

## Dependencies
#37
EOF
create_issue 'Observability: tool-call tracing + LLM spend tracking per session' infra,observability,phase-2 'Earnings Intelligence V2' "$tmpdir/issue_39.md"

echo "Done creating Titan AI issues."
