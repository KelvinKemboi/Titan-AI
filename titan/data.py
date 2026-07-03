import pandas as pd
import streamlit as st

# Emergency Fallback List (Top ~30 by weight) used if Wikipedia is unreachable
FALLBACK_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
    'JPM', 'V', 'XOM', 'UNH', 'MA', 'PG', 'COST', 'JNJ', 'HD', 'MRK', 'CVX', 'ABBV',
    'CRM', 'BAC', 'WMT', 'AMD', 'ACN', 'LIN', 'NFLX', 'MCD', 'DIS', 'KO', 'PEP',
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers():
    """Scrapes Wikipedia for the S&P 500 list. Includes fallback protection."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Fix Ticker Symbol nuances (BRK.B -> BRK-B)
        return [t.replace('.', '-') for t in tickers]
    except Exception:
        return FALLBACK_TICKERS
