# Titan AI: Market Scanner

Quantitative Stock Intelligence Platform

A Streamlit app that scans the S&P 500, scores each company on Value,
Momentum, Quality, Solvency, and Volatility factors, and surfaces the
top-ranked picks with a Wall Street-style writeup and radar chart.

## Roadmap

Titan is evolving into an AI-powered investment research platform on top
of the scanner above:

- **Phase 1 — AI Investment Chat**: ask natural-language questions
  ("Why is Microsoft ranked above Google?", "What are Nvidia's biggest
  risks?") and get answers grounded in Titan's own factor scores and
  rankings, with source citations — not generic LLM knowledge.
- **Phase 2 — Earnings Intelligence**: AI-generated earnings call
  summaries, guidance-direction tracking, sentiment, risk extraction, and
  quarter-over-quarter comparisons, feeding into the same chat.

The full design — system architecture, per-feature technical specs,
GitHub milestones/issues, target repo structure, and longer-term
expansion (Portfolio Builder, Backtesting, Multi-Agent Analyst, Titan
Copilot, Real-time Alerts) — lives in [`docs/`](docs/README.md). The
scanner in this repo keeps working as-is; Phase 1 is additive, not a
rewrite.

## Project structure

```
.
├── app.py                 # Streamlit entrypoint (UI + scan orchestration)
├── titan/
│   ├── config.py          # Scoring weights
│   ├── data.py             # S&P 500 ticker universe (Wikipedia scrape + fallback)
│   └── analyst.py          # RoboAnalyst: per-ticker fundamental/technical scoring
├── .streamlit/
│   └── config.toml        # Theme + Streamlit runtime settings
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Click **Initialize Market Scan** in the UI to pull the current S&P 500
constituent list, fetch price history + fundamentals for each ticker via
`yfinance`, and rank the results.

## Notes

- `app.py` disables SSL certificate verification globally
  (`ssl._create_default_https_context`) as a workaround for the
  "Certificate Verify Failed" error some Mac/Python installs hit when
  calling Yahoo Finance / Wikipedia. This is a known tradeoff for local
  convenience — do not rely on it for anything security-sensitive.
- Ticker scans run concurrently via `ThreadPoolExecutor`; the "Thread
  Power" slider in the sidebar controls how many tickers are analyzed at
  once (higher = faster scans, higher risk of being rate-limited by Yahoo
  Finance).
- This tool is for informational/educational purposes only and is not
  financial advice.
