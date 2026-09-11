"""
Eval harness for extract_guidance: a small set of
earnings-call excerpts with known guidance outcomes, run against the real
classifier to measure accuracy per category.

Requires ANTHROPIC_API_KEY.

Run:
    python -m scripts.eval_guidance_extraction
"""
from dotenv import load_dotenv

from src.earnings.guidance import extract_guidance

load_dotenv()

# (label, transcript_excerpt, expected_direction, is_verbatim)
EVAL_SET = [
    (
        "TRMB Q4 2024 - raised",
        "Suhasini Chandramouli: Good afternoon, and welcome to the call.\n\n"
        "Phil Sawarynski: Relative to our initial fiscal 2025 guidance we provided in December "
        "at our Investor Day, on a constant currency basis, we have raised our revenue and EPS "
        "guidance.\n",
        "raised",
        True,
    ),
    (
        "AES Q3 2024 - maintained",
        "Operator: Welcome to the AES Corporation third quarter 2024 earnings call.\n\n"
        "Steve Coughlin: We are also reaffirming our adjusted EBITDA guidance range of 2.6 "
        "billion to 2.9 billion. While I'm pleased with our execution this year on our growth "
        "objectives, several large drivers have impacted results, primarily at our legacy "
        "businesses, and we now expect to end the year towards the lower end of our guidance "
        "range.\n",
        "maintained",
        True,
    ),
    (
        "Macy's Q3 2024 - lowered",
        "Operator: Welcome to the Macy's third quarter 2024 earnings call.\n\n"
        "Adrian Mitchell: We are updating our annual adjusted diluted EPS outlook of $2.25 to "
        "$2.50 which compares to an adjusted prior outlook of $2.34 to $2.69. We are also "
        "updating our adjusted EBITDA as a percent of total revenue outlook to 8% to 8.4%, "
        "compared to a prior outlook of 8.2% to 8.7%.\n",
        "lowered",
        True,
    ),
    (
        "Alphabet-style - none_given (constructed, grounded in real, documented practice)",
        "Operator: We'll take our next question from an analyst.\n\n"
        "Analyst: Could you share your revenue outlook for next quarter?\n\n"
        "CFO: As you know, we don't provide formal quarterly guidance. What I can share is that "
        "we remain focused on our long-term investment priorities and are pleased with the "
        "momentum across our businesses this quarter.\n",
        "none_given",
        False,
    ),
    (
        "constructed - unclear (mixed signal across segments)",
        "Operator: Welcome to the call.\n\n"
        "CFO: For the segment overall, we're not adjusting our formal outlook at this time. "
        "That said, our hardware division is tracking meaningfully ahead of where we expected, "
        "while the services division has seen some softness we're still evaluating. We'll have "
        "more to share as the quarter develops.\n",
        "unclear",
        False,
    ),
]


def main():
    correct = 0
    print(f"Running eval on {len(EVAL_SET)} examples against the live API...\n")
    for label, transcript, expected, is_verbatim in EVAL_SET:
        direction, quote = extract_guidance(transcript)
        ok = direction == expected
        correct += ok
        status = "PASS" if ok else "FAIL"
        source = "verbatim" if is_verbatim else "constructed"
        print(f"[{status}] expected={expected:<12} got={direction:<12} ({source}) {label}")
        print(f"       quote: {quote!r}")

    accuracy = correct / len(EVAL_SET)
    print(f"\nOverall accuracy: {correct}/{len(EVAL_SET)} ({accuracy:.0%})")


if __name__ == "__main__":
    main()
