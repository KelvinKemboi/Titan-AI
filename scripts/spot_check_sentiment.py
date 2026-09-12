"""
Manual spot-check for score_sentiment: prints the rubric-based score for 5 real Q&A exchanges with an
independently-known tone

All 5 are quotes from real, public earnings calls,
reflowed into api-ninjas.com's "Speaker: text" convention (see
src/earnings/chunking.py) - each with a note on why its tone is
independently known, not just asserted:
  - NVDA Q3 2024: Jensen Huang unequivocally affirms multi-year growth
    across every demand vector when pressed on 2025 Data Center growth -
    expect clearly positive.
  - MSFT Q3 2024: Satya Nadella confidently defends AI capex ROI with no
    hedging - expect positive, less unrestrained than NVDA.
  - M (Macy's) Q3 2024: Tony Spring gives a flat, data-forward answer
    about promotional levels - expect neutral.
  - INTC Q2 2024: Pat Gelsinger responds evasively to a pointed question
    about whether Intel has "accurately diagnosed" its problems, pivoting
    to a future "phase 2" rather than answering directly - expect negative.
  - BA (Boeing) Q4 2024: Kelly Ortberg opens with an unprompted, explicit
    "the quarter was disappointing" - expect clearly negative.

Requires ANTHROPIC_API_KEY.

Run:
    python -m scripts.spot_check_sentiment
"""
from dotenv import load_dotenv

from src.earnings.sentiment import score_sentiment

load_dotenv()

# (label, transcript_excerpt, expected_tone_description)
SPOT_CHECK_SET = [
    (
        "NVDA Q3 2024 - Tim Arcuri (UBS) / Jensen Huang",
        "Operator: We will go ahead and take our next question from Tim Arcuri with UBS.\n\n"
        "Tim Arcuri: Do you think that Data Center can grow even in 2025?\n\n"
        "Jensen Huang: Absolutely believe the Data Center can grow through 2025. And there are, "
        "of course, several reasons for that. We are expanding our supply quite significantly. "
        "We're seeing the waves of generative AI starting from the start-ups and CSPs, moving to "
        "consumer Internet companies, moving to enterprise software platforms, moving to "
        "enterprise companies.\n",
        "clearly positive",
    ),
    (
        "MSFT Q3 2024 - Keith Weiss (Morgan Stanley) / Satya Nadella",
        "Operator: Our first question comes from the line of Keith Weiss with Morgan Stanley.\n\n"
        "Keith Weiss: It looks like Microsoft is on track to ramp capex over 50% year on year "
        "this year to over $50 billion. Could you give us some color on how you as the "
        "management team try to quantify the potential opportunities that underlie these "
        "investments?\n\n"
        "Satya Nadella: At a high level, the way we, as a management team, talk about it is "
        "there are two sides to this. There is training and inference. We want to be able to "
        "allocate the capital required to essentially be training these large foundation models "
        "and stay in the leadership position there. And we've done that successfully all the way "
        "today, and you've seen it flow through our P&L.\n",
        "positive, more measured than NVDA",
    ),
    (
        "M (Macy's) Q3 2024 - Paul Lejuez (Citi) / Tony Spring",
        "Operator: Our next question comes from Paul Lejuez with Citi.\n\n"
        "Paul Lejuez: Curious if promotions have been running higher in the quarter-to-date "
        "period? And if so, how much of the higher sales do you attribute to the higher "
        "promotions year over year?\n\n"
        "Tony Spring: Quarter to date, the promotions are approximately comparable to last year. "
        "So, an interesting factor is that our discount rate is year over year about the same as "
        "it was. The difference when you get into the impact to margin is the mix of business.\n",
        "neutral, matter-of-fact",
    ),
    (
        "INTC Q2 2024 - Vivek Arya (BofA) / Pat Gelsinger",
        "Operator: Our next question comes from Vivek Arya with Bank of America Securities.\n\n"
        "Vivek Arya: Pat, big picture, are the challenges - the product issue, market issue, "
        "strategic issue, execution issue - I'm just wondering, have the core issues been "
        "accurately diagnosed?\n\n"
        "Pat Gelsinger: This first phase of the recovery, restoration, and rebuilding plan is now "
        "well underway. That said, with that foundation in place, it's time for us to focus on "
        "phase 2, building a more financially sustainable model for the company for the future.\n",
        "negative, defensive/evasive",
    ),
    (
        "BA (Boeing) Q4 2024 - Sheila Kahyaoglu (Jefferies) / Kelly Ortberg",
        "Operator: Our next question comes from Sheila Kahyaoglu with Jefferies.\n\n"
        "Sheila Kahyaoglu: Kelly, maybe on the fixed price development programs within BDS. It "
        "seems like the timing to stabilize those keeps getting pushed to the right. How are you "
        "actively managing those programs, and what are you looking to change?\n\n"
        "Kelly Ortberg: OK, Sheila, I'll go first and then ask Brian to follow up. Yes. So, "
        "obviously, the quarter was disappointing here on the fixed price development programs.\n",
        "clearly negative",
    ),
]


def main():
    print(f"Running spot-check on {len(SPOT_CHECK_SET)} real transcript excerpts against the live API...\n")
    for label, transcript, expected_tone in SPOT_CHECK_SET:
        score = score_sentiment(transcript)
        print(f"score={score!s:<6} expected~{expected_tone:<32} {label}")

    print(
        "\nEyeball check: scores should roughly rank NVDA > MSFT > M > INTC > BA, with M near 0 "
        "and NVDA/BA near the +1/-1 extremes. This is a plausibility check, not a pass/fail grade."
    )


if __name__ == "__main__":
    main()
