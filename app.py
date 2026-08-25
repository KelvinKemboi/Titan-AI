import logging
import os
import ssl

import requests
import streamlit as st
import plotly.graph_objects as go

from titan.config import WEIGHTS
from titan.data import get_sp500_tickers
from src.analytics.scanner_service import ScanAlreadyRunningError, run_scan_and_persist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# CONFIGURATION & SETUP
st.set_page_config(page_title="Titan: AI Hedge Fund", layout="wide", initial_sidebar_state="collapsed")

# Streamlit has no built-in "sidebar on the right" option - the sidebar and
# main content are flex siblings (stAppViewContainer, flex-direction: row),
# so swapping their visual order does it. The expand toggle (shown only
# while the sidebar is collapsed) lives separately in the top header/toolbar,
# not inside the sidebar itself, so it needs its own rule to follow along;
# the collapse toggle (shown while expanded) is inside the sidebar and moves
# with it automatically.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { order: 1; }
    /* Streamlit collapses the sidebar by translating it -100% (off the LEFT edge of its
       own box) - since its own width shrinks to 0 at the same time, a relative-% override
       here would resolve against that same 0, so this uses a viewport-relative offset
       instead (robust regardless of the sidebar's configured/resized width). */
    [data-testid="stSidebar"][aria-expanded="false"] { transform: translateX(100vw) !important; }
    /* The expand toggle lives 3 levels deep in unlabeled flex wrapper divs inside the
       header, alongside the Deploy/menu buttons - rather than depend on that nested
       structure, detach it from flow and pin it to the header's top-right corner. */
    [data-testid="stExpandSidebarButton"] {
        position: fixed;
        top: 0.6rem;
        right: 8rem; /* clears the Deploy button + menu icon group pinned to the far right */
        left: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# The Streamlit UI talks to the FastAPI gateway (`uvicorn src.api.main:app`) over HTTP
CHAT_API_URL = os.environ.get("CHAT_API_URL", f"http://localhost:{os.environ.get('API_PORT', '8000')}/chat")

# SSL Bypass for Mac/PC (Fixes "Certificate Verify Failed" errors)
ssl._create_default_https_context = ssl._create_unverified_context

# UI
st.title("Titan AI: Market Scanner")
st.markdown("`Status: Online` | `Model: v4.2 (Value/Momentum)` | `Universe: S&P 500`")

if "chat_session_id" not in st.session_state:
    st.session_state["chat_session_id"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
# None = no scan run yet this session; persisted across reruns (e.g. a chat
# message's st.rerun()) so the results below don't vanish on the next rerun.
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None

# Hidden Sidebar for "Power Users" (You)
with st.sidebar:
    st.header(" Simulation Settings")
    st.info("The AI runs autonomously. Settings are optimized for the current VIX environment.")
    concurrency = st.slider("Thread Power (Speed vs Safety)", 1, 20, 5)

    st.divider()
    st.header("Titan Analyst Chat")
    st.caption("Ask about a ticker's factor scores or compare tickers, backed by the latest scan.")

    chat_log = st.container(height=400) # container to hold the chat messages
    with chat_log:
        for turn in st.session_state["chat_history"]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                for source in turn.get("sources", []):
                    st.caption(f"Source: {source['ticker']}, scan #{source['ref_id']}")

    chat_prompt = st.chat_input("Ask Titan a question about the market or a specific stock...")
    if chat_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": chat_prompt, "sources": []})
        try: # Call the FastAPI backend to get a response from the AI model
            api_response = requests.post(
                CHAT_API_URL,
                json={"session_id": st.session_state["chat_session_id"], "message": chat_prompt},
                timeout=30,
            )
            api_response.raise_for_status()
            payload = api_response.json()
        #handle network errors and inform the user
        except requests.RequestException as exc:
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": f"Sorry, I couldn't reach Titan's chat service ({exc}).",
                    "sources": [],
                }
            )
        else:
            st.session_state["chat_session_id"] = payload["session_id"]
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": payload["response"],
                    "sources": payload.get("sources", []),
                }
            )
        st.rerun()

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
            # Nothing new happened - leave any previously-scanned results on screen rather than blanking them.
            status.update(label="A scan is already in progress - try again shortly.", state="error", expanded=False)
        except Exception:
            # Ditto: a failed attempt shouldn't erase a still-valid earlier scan's results.
            # Logged here since this is the only place the actual exception is visible -
            # the UI message intentionally doesn't leak internals to the browser.
            logger.exception("Initialize Market Scan failed")
            status.update(label="Scan failed - check the app logs for details.", state="error", expanded=False)
        else:
            status.update(label="Scan Complete!", state="complete", expanded=False)
            st.session_state["scan_results"] = results

# Rendered from session_state (not the button block above) so results persist
# across reruns triggered elsewhere on the page (e.g. the chat's st.rerun()).
results = st.session_state["scan_results"]
if results:
    # Sort by Score
    results.sort(key=lambda x: x.score, reverse=True)
    top_picks = results[:5]

    st.divider()
    st.subheader("The Alpha List (Top 5)")

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
