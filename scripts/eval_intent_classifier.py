"""
Eval harness for classify_intent: ~30 hand-written questions, each labeled with the expected intent, run against
the real classifier to measure accuracy and per-call latency

Run:
    python -m scripts.eval_intent_classifier
"""
import statistics
import time

from dotenv import load_dotenv

from src.agents.intent_classifier import classify_intent

load_dotenv()

# (question, expected_intent)
EVAL_SET = [
    # structure: one named ticker asking about its own factor scores/rating
    ("Why is AAPL rated a BUY?", "structured"),
    ("Explain Apple's factor score.", "structured"),
    ("What's driving MSFT's composite score?", "structured"),
    ("Is NVDA overvalued right now?", "structured"),
    ("What's Tesla's momentum score?", "structured"),
    ("Why did AMZN's rating change?", "structured"),
    ("How risky is META as an investment?", "structured"),
    ("What are Nvidia's biggest risks?", "structured"), # product-brief example
    ("Break down GOOGL's quality score.", "structured"),
    ("What's driving Netflix's volatility rating?", "structured"),

    # comparison: two or more named tickers
    ("Why is Microsoft ranked above Google?", "comparison"), # product-brief example
    ("Compare AMD and Intel.", "comparison"),
    ("Is AAPL or MSFT the stronger buy?", "comparison"),
    ("MSFT vs GOOGL - which has better momentum?", "comparison"),
    ("Rank NVDA, AMD, and INTC by quality score.", "comparison"),
    ("Why does TSLA score higher than F?", "comparison"),
    ("Which is safer, JNJ or PFE?", "comparison"),
    ("How do AAPL and GOOGL stack up on solvency?", "comparison"),
    ("Between COST and WMT, which is the better value pick?", "comparison"),
    ("Compare the top 3 tech stocks by composite score.", "comparison"),

    # qualitative: no single ticker named, thematic/subjective 
    ("Which companies have a deep competitive moat?", "qualitative"), # product-brief example
    ("Find me high-quality compounders.", "qualitative"),
    ("What looks undervalued right now?", "qualitative"),
    ("Which stocks have strong pricing power?", "qualitative"),
    ("Show me companies with low debt and strong margins.", "qualitative"),
    ("What are some defensive names to consider?", "qualitative"),
    ("Which companies are showing bullish momentum trends?", "qualitative"),
    ("Find me stocks that look like value traps.", "qualitative"),
    ("What sectors look strongest in this scan?", "qualitative"),
    ("Suggest some low-volatility picks.", "qualitative"),
]


def main():
    correct = 0
    latencies = []
    by_intent = {}

    print(f"Running eval on {len(EVAL_SET)} examples against the live API...\n")
    for question, expected in EVAL_SET:
        t0 = time.monotonic() # start timer
        predicted = classify_intent(question) # call the real classifier
        elapsed = time.monotonic() - t0 # stop timer
        latencies.append(elapsed) # record latency

        ok = predicted == expected
        correct += ok # increment correct count if prediction matches expected
        by_intent.setdefault(expected, [0, 0]) # initialize counts for this intent if not already present
        by_intent[expected][0] += ok # increment correct count
        by_intent[expected][1] += 1 # increment total count

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {elapsed * 1000:6.0f}ms  expected={expected:<12} got={predicted!s:<12} {question}") # print result

    # Overall accuracy
    accuracy = correct/len(EVAL_SET)
    print(f"\nOverall accuracy: {correct}/{len(EVAL_SET)} ({accuracy:.0%})")
    # Per-intent accuracy
    for intent, (n_correct, n_total) in by_intent.items():
        print(f"  {intent:<12} {n_correct}/{n_total} ({n_correct / n_total:.0%})")

    # Latency stats
    print(
        f"\nLatency (ms): avg={statistics.mean(latencies) * 1000:.0f} " # mean latency in milliseconds
        f"median={statistics.median(latencies) * 1000:.0f} " # median latency in milliseconds
        f"min={min(latencies) * 1000:.0f} max={max(latencies) * 1000:.0f}" # min/max latency in milliseconds
    )


if __name__ == "__main__":
    main()
