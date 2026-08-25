import json
import os
from typing import List, Optional

import anthropic
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.agents.intent_classifier import INTENT_HINTS, classify_intent
from src.agents.tools import TOOLS, call_tool
from src.agents.tools.base import Source
from src.data.models import ChatMessage

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

# Safety net against a pathological tool-call loop (e.g. the model repeatedly retrying a failing call)
MAX_TOOL_ITERATIONS = 5

# Conversation memory: how many prior chat_messages
# rows to inject as context for a follow-up question. "last 10 turns" MVP default
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

# block is a tool_use block from the model's response, and _run_tool executes it against the database.
def _run_tool(db: Session, block) -> tuple:
    """
    Executes one tool_use block and returns (tool_result content block, sources).
    Any failure - a ValueError from call_tool (e.g. unknown ticker) or an
    unexpected one (e.g. a transient DB error, or search_memos' Voyage API
    call failing) - is turned into an `is_error` tool_result instead of
    raising, so the model sees the failure and can respond to the user in
    plain language rather than the request surfacing as a raw 500.
    """
    try:
        result = call_tool(db, block.name, block.input) # dispatches to the matching tool implementation (registered in src.agents.tools)
    except Exception as exc: # any tool failure, expected (ValueError) or not
        return (
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps({"error": str(exc)}),
                "is_error": True,
            },
            [],
        )
    return (#returns a tuple containing the tool_result content block and the sources for the tool call
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result.model_dump_json(),
        },
        result.sources,
    )


def answer_question( db: Session, question: str, *, history: Optional[List[ChatMessage]] = None,
                    client: Optional[anthropic.Anthropic] = None,) -> ChatAnswer:
    """
    Owns a single-turn conversation: sends `question` to the model with the
    factor-score tools, executes any tool calls against `db`, feeds the
    results back, and repeats until the model responds with text instead of a
    tool call. Returns that text with the sources collected along the way
    """
    client = client or anthropic.Anthropic() # if no client is provided, create a new instance of the Anthropics API client
    intent = classify_intent(question) # classifies the user's question to determine the intent, which may influence how the system prompt is constructed
    system_prompt = SYSTEM_PROMPT
    if intent in INTENT_HINTS: # if the intent is one of the known intents, append the corresponding hint to the system prompt
        system_prompt = (
            f"{SYSTEM_PROMPT}\n\n{INTENT_HINTS[intent]} "
            "This is a hint, not a restriction - use whichever tool actually answers the question."
        )

    messages = [{"role": m.role, "content": m.content} for m in (history or [])] # initialize the messages list with the chat history as a list of dictionaries
    messages.append({"role": "user", "content": question})
    sources: List[Source] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        try: # Call the model with the system prompt, tools, and messages
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError:
            return ChatAnswer(response=FALLBACK_RESPONSE, sources=[])

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            text = "".join(b.text for b in response.content if b.type == "text")
            return ChatAnswer(response=text, sources=sources) # if there are no tool_use blocks, return the model's text response along with the collected sources

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            result_block, block_sources = _run_tool(db, block)
            tool_results.append(result_block)
            sources.extend(block_sources)
        messages.append({"role": "user", "content": tool_results})

    return ChatAnswer(response=INCONCLUSIVE_RESPONSE, sources=sources)
