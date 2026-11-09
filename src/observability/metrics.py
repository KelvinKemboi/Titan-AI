"""
Prometheus metrics: the real-time, aggregate counterpart to tracing.py's
per-request Postgres rows. tracing.py answers "what exactly happened on
request X"; this module answers "what is the p95 chat latency / error rate /
spend rate right now" - scraped by Prometheus from GET /metrics and charted
in Grafana (see observability/ in the repo root).

Every helper here is cheap, in-process, and never raises.

Multi-worker note: prometheus_client keeps metrics per process. If you run
uvicorn with --workers > 1, set PROMETHEUS_MULTIPROC_DIR (see
https://prometheus.github.io/client_python/multiprocess/) or scrape each
worker separately.
"""
import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Buckets tuned for this app: cached reads are ms, chat turns with several
# tool round-trips can take tens of seconds.
_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120)

# HTTP
HTTP_REQUESTS = Counter(
    "titan_http_requests_total", "HTTP requests handled", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "titan_http_request_duration_seconds", "HTTP request latency", ["method", "route"],
    buckets=_LATENCY_BUCKETS,
)

# LLM
LLM_CALLS = Counter(
    "titan_llm_calls_total", "Anthropic messages.create calls", ["model", "call_type", "outcome"]
)
LLM_TOKENS = Counter(
    "titan_llm_tokens_total", "LLM tokens consumed", ["model", "call_type", "direction"]
)
LLM_COST = Counter(
    "titan_llm_cost_usd_total", "Estimated LLM spend in USD", ["model", "call_type"]
)
LLM_LATENCY = Histogram(
    "titan_llm_latency_seconds", "LLM call latency", ["model", "call_type"], buckets=_LATENCY_BUCKETS
)

# Agent / chat
TOOL_CALLS = Counter("titan_tool_calls_total", "Chat agent tool calls", ["tool", "success"])
CHAT_TURNS = Counter(
    "titan_chat_turns_total", "Chat turns by how they ended",
    ["outcome"],  # answered | cache_hit | llm_error | inconclusive
)
CHAT_TOOL_ITERATIONS = Histogram(
    "titan_chat_tool_iterations", "Model round-trips needed to answer one chat turn",
    buckets=(1, 2, 3, 4, 5, 6),
)
CHAT_SOURCES = Histogram(
    "titan_chat_sources_per_answer", "Citations attached to a chat answer (0 = ungrounded)",
    buckets=(0, 1, 2, 3, 5, 10, 20),
)


def observe_llm_call(*, model, call_type, outcome, latency_s=None,
                     input_tokens=None, output_tokens=None, cost_usd=None):
    try:
        LLM_CALLS.labels(model, call_type, outcome).inc()
        if latency_s is not None:
            LLM_LATENCY.labels(model, call_type).observe(latency_s)
        if input_tokens:
            LLM_TOKENS.labels(model, call_type, "input").inc(input_tokens)
        if output_tokens:
            LLM_TOKENS.labels(model, call_type, "output").inc(output_tokens)
        if cost_usd:
            LLM_COST.labels(model, call_type).inc(float(cost_usd))
    except Exception:  # metrics must never break the request they observe
        pass


def observe_tool_call(tool, success):
    try:
        TOOL_CALLS.labels(tool, str(bool(success)).lower()).inc()
    except Exception:
        pass


def observe_chat_turn(outcome, *, iterations=None, sources=None):
    try:
        CHAT_TURNS.labels(outcome).inc()
        if iterations is not None:
            CHAT_TOOL_ITERATIONS.observe(iterations)
        if sources is not None:
            CHAT_SOURCES.observe(sources)
    except Exception:
        pass


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Records count + latency per request, labelled by the route *template*
    (/company/{ticker}), not the raw path, so label cardinality stays bounded."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            route_label = getattr(route, "path", "unmatched")
            HTTP_REQUESTS.labels(request.method, route_label, str(status)).inc()
            HTTP_LATENCY.labels(request.method, route_label).observe(time.perf_counter() - start)


def metrics_endpoint(_: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
