import random
import time
from concurrent.futures import ThreadPoolExecutor

from titan.analyst import RoboAnalyst


def _analyze_ticker(ticker):
    # Polite Delay to prevent IP Bans (Dynamic Throttling)
    time.sleep(random.uniform(0.1, 1.0))

    analyst = RoboAnalyst(ticker)
    if analyst.analyze():
        analyst.generate_memo()
        return analyst
    return None


def run_scan(tickers, concurrency=5, on_progress=None):
    """
    Runs the factor-scoring scan over `tickers` and returns the valid
    RoboAnalyst results, unsorted.

    UI-agnostic: callers drive their own progress display via
    `on_progress(index, total)`, invoked with the same 0-based index and
    total count for every ticker as it completes.
    """
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_analyze_ticker, t) for t in tickers]

        for i, future in enumerate(futures):
            res = future.result()
            if res and res.valid:
                results.append(res)

            if on_progress:
                on_progress(i, len(futures))

    return results
