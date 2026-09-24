"""
Simple LLM spend report:
aggregate spend, a per-call-type breakdown, and the top sessions by
cost, over src/observability/tracing.py's llm_calls/chat_messages rows.
"""
import argparse
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from src.data.db import SessionLocal
from src.observability.spend import get_aggregate_spend, get_top_sessions_by_spend

load_dotenv()


def _fmt_cost(cost_usd, call_count=None):
    if cost_usd is not None:
        return f"${cost_usd:.4f}"
    return "$0.0000" if call_count == 0 else "n/a (unpriced model)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since-days", type=int, default=None, help="Restrict to calls in the last N days")
    parser.add_argument("--top", type=int, default=10, help="How many top-spending sessions to list")
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.since_days) if args.since_days else None

    db = SessionLocal()
    try:
        summary = get_aggregate_spend(db, since=since)
        top_sessions = get_top_sessions_by_spend(db, limit=args.top)
    finally:
        db.close()

    window = f"last {args.since_days} day(s)" if args.since_days else "all time"
    print(f"=== LLM spend report ({window}) ===\n")
    print(f"Total calls:        {summary.call_count}")
    print(f"Total input tokens: {summary.input_tokens:,}")
    print(f"Total output tokens:{summary.output_tokens:,}")
    print(f"Total cost:         {_fmt_cost(summary.cost_usd, summary.call_count)}")

    if summary.by_call_type:
        print("\nBy call type:")
        for bucket in summary.by_call_type:
            print(
                f"  {bucket.call_type:<24} {bucket.call_count:>5} calls  "
                f"{bucket.input_tokens:>10,} in  {bucket.output_tokens:>10,} out  "
                f"{_fmt_cost(bucket.cost_usd, bucket.call_count)}"
            )

    print(f"\nTop {len(top_sessions)} sessions by cost:")
    for row in top_sessions:
        print(
            f"  {row['session_id']}  {_fmt_cost(row['cost_usd'], row['call_count']):>18}  "
            f"{row['call_count']:>4} calls  {row['input_tokens']:>10,} in  {row['output_tokens']:>10,} out"
        )
    if not top_sessions:
        print("  (no traced sessions yet)")


if __name__ == "__main__":
    main()
