import io

import pandas as pd
import requests
import streamlit as st

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Emergency Fallback List (Top ~30 by weight) used if Wikipedia is unreachable
FALLBACK_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
    'JPM', 'V', 'XOM', 'UNH', 'MA', 'PG', 'COST', 'JNJ', 'HD', 'MRK', 'CVX', 'ABBV',
    'CRM', 'BAC', 'WMT', 'AMD', 'ACN', 'LIN', 'NFLX', 'MCD', 'DIS', 'KO', 'PEP',
]


def _fetch_from_wikipedia():
    """Scrapes Wikipedia for the S&P 500 list."""
    # Wikipedia returns 403 to requests with no User-Agent header, so
    # pd.read_html(url) alone (no header control) no longer works.
    resp = requests.get(
        WIKIPEDIA_URL,
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0 (compatible; TitanAI/1.0)"},
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df['Symbol'].tolist()
    return [t.replace('.', '-') for t in tickers]


@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers():
    """
    Returns the current S&P 500 ticker list.

    Scrapes Wikipedia for the list. Falls back to a static list if
    Wikipedia is unreachable.
    """
    try:
        return _fetch_from_wikipedia()
    except Exception:
        return FALLBACK_TICKERS
