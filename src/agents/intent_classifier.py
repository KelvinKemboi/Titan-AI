"""
Intent classification: model call that classifies each incoming chat message before the Chat/Agent
Service (chat_service.py) decides which tools to foreground
"""
from typing import Literal, Optional

import anthropic

# Haiku-class model
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
CLASSIFIER_MAX_TOKENS = 20

Intent = Literal["structured", "qualitative", "comparison"]
INTENTS = ("structured", "qualitative", "comparison")

_SYSTEM_PROMPT = """Classify an investment-research question into exactly one category, \
based on what kind of data would answer it - not on whether you personally know the answer:

- structured: about ONE named ticker's own factor scores, rating, or ranking (e.g. \
"why is AAPL rated a BUY", "explain MSFT's score", "what's NVDA's biggest risk factor").
- comparison: TWO OR MORE named tickers being compared or ranked against each other \
(e.g. "is AMD or INTC stronger", "compare MSFT and GOOGL", "why is X ranked above Y").
- qualitative: no single ticker named, or a subjective/thematic question only answerable \
from prose commentary rather than a number (e.g. "which companies have a deep competitive \
moat", "find high-quality compounders", "what looks undervalued right now").

Call classify_intent with your answer."""

_CLASSIFY_INTENT_TOOL = {
    "name": "classify_intent",
    "description": "Records the classified intent for one investment-research question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(INTENTS),
            },
        },
        "required": ["intent"],
    },
}

# Injected into chat_service's system prompt as a routing hint
INTENT_HINTS = {
    "structured": (
        "This question looks like it's about one named ticker's own factor scores - "
        "prefer get_factor_scores if exactly one ticker is named."
    ),
    "comparison": (
        "This question looks like a comparison between tickers - prefer compare_tickers "
        "if 2 or more tickers are named."
    ),
    "qualitative": (
        "This question looks qualitative/thematic with no single ticker to look up - "
        "prefer search_memos."
    ),
}


def classify_intent(question: str, *, client: Optional[anthropic.Anthropic] = None) -> Optional[Intent]:
    """
    Classifies `question` into one of Intent's three values via one forced
    tool-call to CLASSIFIER_MODEL. `tool_choice` forces the model to call
    classify_intent (rather than reply with text), and the tool schema's
    `enum` constrains `intent` to exactly one of the three values
    """
    client = client or anthropic.Anthropic()
    try:
        # Call the model with a system prompt that instructs it to classify the question into one of the three intents
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            tools=[_CLASSIFY_INTENT_TOOL],
            tool_choice={"type": "tool", "name": "classify_intent"},
            messages=[{"role": "user", "content": question}],
        )
    except Exception:
        return None

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None) # find the tool_use block in the model's response, if any
    if tool_use_block is None:
        return None

    intent = tool_use_block.input.get("intent")
    return intent if intent in INTENTS else None
