import ssl

import streamlit as st
import plotly.graph_objects as go

from titan.config import WEIGHTS
from titan.data import get_sp500_tickers
from src.analytics.scanner_service import ScanAlreadyRunningError, run_scan_and_persist

# CONFIGURATION & SETUP
st.set_page_config(page_title="Titan: AI Hedge Fund", layout="wide", initial_sidebar_state="collapsed")

# SSL Bypass for Mac/PC (Fixes "Certificate Verify Failed" errors)
ssl._create_default_https_context = ssl._create_unverified_context

# UI
st.title("Titan AI: Market Scanner")
st.markdown("`Status: Online` | `Model: v4.2 (Value/Momentum)` | `Universe: S&P 500`")

# Hidden Sidebar for "Power Users" (You)
with st.sidebar:
    st.header(" Simulation Settings")
    st.info("The AI runs autonomously. Settings are optimized for the current VIX environment.")
    concurrency = st.slider("Thread Power (Speed vs Safety)", 1, 20, 5)

if st.button("Initialize Market Scan"):

    # Get Universe
    with st.status("Connecting to Market Data Streams...", expanded=True) as status:
        st.write("Downloading S&P 500 Index constituents...")
        tickers = get_sp500_tickers()
        st.write(f"Universe Identified: {len(tickers)} equities.")

        # Scanning Loop
        st.write("Spinning up AI Analyst Swarm...")
        progress_bar = st.progress(0)
        def on_progress(i, total):
            # Update UI every 5 ticks (and on the final tick) to save resources
            if i % 5 == 0 or i == total - 1:
                progress_bar.progress((i + 1) / total)
        try:
            results = run_scan_and_persist(tickers, concurrency=concurrency, on_progress=on_progress)
        except ScanAlreadyRunningError:
            results = None
            status.update(label="A scan is already in progress — try again shortly.", state="error", expanded=False)
        else:
            status.update(label="Scan Complete!", state="complete", expanded=False)

    # Results Display
    if results:
        # Sort by Score
        results.sort(key=lambda x: x.score, reverse=True)
        top_picks = results[:5]

        st.divider()
        st.subheader("🏆 The Alpha List (Top 5)")

        cols = st.columns(5)
        for i, stock in enumerate(top_picks):
            with cols[i]:
                st.metric(
                    label=stock.ticker,
                    value=f"${stock.metrics['Price']:.2f}",
                    delta=f"Score: {int(stock.score)}",
                )
                st.caption(f"{stock.metrics['Trend']}")

        st.divider()
        st.subheader("Detailed Analyst Reports")

        # Only show top 30 to keep browser fast
        for stock in results[:30]:
            with st.expander(f"**{stock.ticker}** | Score: {int(stock.score)}"):
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    st.markdown(stock.memo)
                with col2:
                    # Radar Chart Visualization
                    categories = list(WEIGHTS.keys())
                    values = stock.metrics['Scores']

                    fig = go.Figure(data=go.Scatterpolar(
                        r=values, theta=categories, fill='toself',
                        line=dict(color='#00CC96'),
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False,
                        margin=dict(t=20, b=20, l=20, r=20),
                        height=250,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"radar-{stock.ticker}")
    elif results is not None:
        st.error("Scan failed. Check your internet connection or try again later.")
