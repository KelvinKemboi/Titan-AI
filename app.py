import ssl
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import plotly.graph_objects as go
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from titan.config import WEIGHTS
from titan.data import get_sp500_tickers
from titan.analyst import RoboAnalyst

# 1. CONFIGURATION & SETUP
st.set_page_config(page_title="Titan: AI Hedge Fund", layout="wide", initial_sidebar_state="collapsed")

# SSL Bypass for Mac/PC (Fixes "Certificate Verify Failed" errors)
ssl._create_default_https_context = ssl._create_unverified_context

# 2. UI
st.title("Titan AI: Market Scanner")
st.markdown("`Status: Online` | `Model: v4.2 (Value/Momentum)` | `Universe: S&P 500`")

# Hidden Sidebar for "Power Users" (You)
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    st.info("The AI runs autonomously. Settings are optimized for the current VIX environment.")
    concurrency = st.slider("Thread Power (Speed vs Safety)", 1, 20, 5)

if st.button("Initialize Market Scan"):

    # 1. Get Universe
    with st.status("📡 Connecting to Market Data Streams...", expanded=True) as status:
        st.write("Downloading S&P 500 Index constituents...")
        tickers = get_sp500_tickers()
        st.write(f"Universe Identified: {len(tickers)} equities.")

        # 2. Scanning Loop
        st.write("Spinning up AI Analyst Swarm...")
        results = []
        progress_bar = st.progress(0)

        # Streamlit's script context must be propagated to worker threads
        # explicitly, since ThreadPoolExecutor threads are created outside
        # the main script run.
        ctx = get_script_run_ctx()

        # The Worker Function
        def process_ticker(ticker):
            add_script_run_ctx(threading.current_thread(), ctx)

            # Polite Delay to prevent IP Bans (Dynamic Throttling)
            time.sleep(random.uniform(0.1, 1.0))

            analyst = RoboAnalyst(ticker)
            if analyst.analyze():
                analyst.generate_memo()
                return analyst
            return None

        # Threaded Execution
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all jobs
            futures = [executor.submit(process_ticker, t) for t in tickers]

            # Iterate as they complete
            for i, future in enumerate(futures):
                res = future.result()
                if res and res.valid:
                    results.append(res)

                # Update UI every 5 ticks (and on the final tick) to save resources
                if i % 5 == 0 or i == len(futures) - 1:
                    progress_bar.progress((i + 1) / len(tickers))

        status.update(label="Scan Complete!", state="complete", expanded=False)

    # 3. Results Display
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
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Scan failed. Check your internet connection or try again later.")
