import json
import logging
import os
import time
import uuid
from typing import List, Optional

import anthropic
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.agents.entity_tracker import entity_hint, extract_entities
from src.agents.intent_classifier import INTENT_HINTS, classify_intent
from src.agents.tools import TOOLS, call_tool
from src.agents.tools.base import Source
from src.data.cache import cache_get, cache_set, chat_cache_key
from src.data.models import ChatMessage, FactorScore
from src.observability.tracing import record_llm_call, record_tool_call

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

# Safety net against a pathological tool-call loop that never converges.
MAX_TOOL_ITERATIONS = 5

# How many prior chat_messages rows to inject as context for a follow-up.
CHAT_HISTORY_WINDOW = int(os.environ.get("CHAT_HISTORY_WINDOW", "10"))

SYSTEM_PROMPT = """You are Titan's investment research assistant. Answer questions \
using Titan's own data, retrieved through the tools available to you. Do not answer from \
your general knowledge or training data.

For any question about a ticker's factor scores, rating, or how tickers compare, \
call the relevant tool and base your answer only on what it returns. State \
numbers, ratings, and comparisons exactly as the tool reports them, and mention \
which ticker and scan they came from.

If a question needs information Titan doesn't have a tool for, say so plainly \
rather than answering from general knowledge - even for facts you would \
otherwise know, like a company's sector or headquarters.

If a tool call fails (for example, an unknown ticker), explain what went wrong \
in plain language instead of guessing at an answer."""

FALLBACK_RESPONSE = (
    "Sorry, I ran into a problem answering that and can't give a reliable "
    "response right now. Please try again."
)

INCONCLUSIVE_RESPONSE = (
    "I wasn't able to finish answering that after several tool calls. Please "
    "try rephrasing your question."
)

class ChatAnswer(BaseModel):
    """Response envelope for one turn: the model's final text plus every
    source its tool calls produced, for citation in the UI."""

    response: str
    sources: List[Source]


def _run_tool(db: Session, block, request_id: str) -> tuple:
    """
    Executes one tool_use block and returns (tool_result content block,
    sources). A failure is turned into an `is_error` tool_result instead
    of raising, so the model can explain the problem in plain language
    rather than the request surfacing as a raw 500. Every call, success
    or failure, is traced under `request_id` with its exact input so a
    bad answer can be traced back to the tool call that produced it.
    """
    try:
        result = call_tool(db, block.name, block.input)
    except Exception as exc:
        record_tool_call(
            db, request_id=request_id, tool_name=block.name, tool_input=block.input,
            success=False, error=str(exc),
        )
        return (
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps({"error": str(exc)}),
                "is_error": True,
            },
            [],
        )
    record_tool_call(
        db, request_id=request_id, tool_name=block.name, tool_input=block.input,
        success=True, sources=[s.model_dump(mode="json") for s in result.sources],
    )
    return (
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result.model_dump_json(),
        },
        result.sources,
    )


def answer_question( db: Session, question: str, *, history: Optional[List[ChatMessage]] = None,
                    client: Optional[anthropic.Anthropic] = None, request_id: Optional[str] = None,) -> ChatAnswer:
    """
    Owns a single-turn conversation: sends `question` to the model with
    the available tools, executes any tool calls against `db`, feeds the
    results back, and repeats until the model responds with text instead
    of a tool call. Returns that text with the sources collected along
    the way.

    When `history` is empty, `question` is looked up in the FAQ-style
    response cache first (keyed on the question plus the latest
    scan_run_id, so a new scan never serves a stale cached answer) and a
    hit is returned immediately with no tracing, since nothing new was
    actually called. `request_id` groups every LLM/tool call made this
    turn for later tracing; pass the same id the caller will store on
    this turn's chat_messages row, or omit it to have one generated.
    """
    request_id = request_id or str(uuid.uuid4())
    client = client or anthropic.Anthropic()

    cache_key = None
    if not history:
        data_version = db.query(func.max(FactorScore.scan_run_id)).scalar()
        cache_key = chat_cache_key(question, data_version)
        cached = cache_get(cache_key)
        if cached is not None:
            return ChatAnswer.model_validate_json(cached)

    intent = classify_intent(question, db=db, request_id=request_id)
    system_prompt = SYSTEM_PROMPT
    if intent in INTENT_HINTS:
        system_prompt = (
            f"{SYSTEM_PROMPT}\n\n{INTENT_HINTS[intent]} "
            "This is a hint, not a restriction - use whichever tool actually answers the question."
        )

    # Grounds pronoun/ellipsis follow-ups ("what about its momentum?") in
    # whichever ticker/factor was last discussed.
    entities = extract_entities(history or [])
    logger.info("chat entity state for this turn: %s", entities.model_dump_json())
    hint = entity_hint(entities)
    if hint:
        system_prompt = f"{system_prompt}\n\n{hint}"

    messages = [{"role": m.role, "content": m.content} for m in (history or [])]
    messages.append({"role": "user", "content": question})
    sources: List[Source] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        start = time.monotonic()
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError:
            return ChatAnswer(response=FALLBACK_RESPONSE, sources=[])
        record_llm_call(
            db, request_id=request_id, call_type="chat_generation", model=MODEL,
            response=response, latency_ms=int((time.monotonic() - start) * 1000),
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            text = "".join(b.text for b in response.content if b.type == "text")
            answer = ChatAnswer(response=text, sources=sources)
            if cache_key is not None:
                cache_set(cache_key, answer.model_dump_json())
            return answer

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            result_block, block_sources = _run_tool(db, block, request_id)
            tool_results.append(result_block)
            sources.extend(block_sources)
        messages.append({"role": "user", "content": tool_results})

    return ChatAnswer(response=INCONCLUSIVE_RESPONSE, sources=sources)
