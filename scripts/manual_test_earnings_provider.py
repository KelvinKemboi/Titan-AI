"""
Manual test: exercises src/earnings/provider_client.py against the real
API Ninjas endpoints -
pagination via search_transcripts, then a full-text fetch via
get_transcript for the first result found.

Run:
    python -m scripts.manual_test_earnings_provider [TICKER]
"""
import sys

from dotenv import load_dotenv

from src.earnings.provider_client import EarningsProviderError, get_transcript, search_transcripts

load_dotenv()


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\n--- search_transcripts({ticker!r}) (pagination) ---")
    try:
        results = list(search_transcripts(ticker))
    except EarningsProviderError as exc:
        print(f"FAIL: search_transcripts raised {exc}")
        return

    if not results:
        print(f"FAIL: no transcripts found for {ticker} - try a different ticker.")
        return
    print(f"PASS: found {len(results)} transcripts, most recent first:")
    for r in results[:5]:
        print(f"  {r.fiscal_year} {r.fiscal_quarter} (call_date={r.call_date})")

    latest = results[0]
    print(f"\n--- get_transcript({ticker!r}, year={latest.fiscal_year}, quarter={latest.fiscal_quarter[1:]}) ---")
    try:
        transcript = get_transcript(ticker, year=latest.fiscal_year, quarter=int(latest.fiscal_quarter[1:]))
    except EarningsProviderError as exc:
        print(f"FAIL: get_transcript raised {exc}")
        return

    if transcript is None:
        print("FAIL: get_transcript returned None for a quarter search_transcripts just listed.")
        return
    print(f"PASS: fetched {len(transcript.raw_text)} chars of transcript text for {transcript.ticker}.")
    print("First 200 chars:", transcript.raw_text[:200])


if __name__ == "__main__":
    main()
