import json
import os
import time
import threading
from collections import deque
from typing import Deque, Dict, List

import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SYMBOLS = [s.strip().upper() for s in os.environ.get("SYMBOLS", "AAPL,MSFT,NVDA").split(",") if s.strip()]

st.set_page_config(page_title="AI Stock Sentiment & Forecast", layout="wide")

st.sidebar.title("Controls")
symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=0)
refresh = st.sidebar.slider("Refresh (s)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

col1, col2 = st.columns([2, 1])

price_hist: Deque[float] = deque(maxlen=360)
time_hist: Deque[float] = deque(maxlen=360)
senti_hist: Deque[float] = deque(maxlen=360)

placeholder_chart = col1.empty()
placeholder_stats = col2.empty()


def fetch_state(sym: str) -> Dict:
    try:
        r = requests.get(f"{API_URL}/state", params={"symbol": sym}, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}


last_ts = 0.0

while True:
    data = fetch_state(symbol)
    if not data:
        st.warning("Waiting for API...")
        time.sleep(refresh)
        continue

    ts = data.get("ts", time.time())
    if ts != last_ts:
        last_ts = ts
        price = data.get("last_price", 0.0)
        snap = data.get("snapshot", {})
        fc = data.get("forecast", {})
        price_hist.append(price)
        time_hist.append(ts)
        senti_hist.append(float(snap.get("senti_ewma", 0.0)))

        # Build price + fan
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(time_hist), y=list(price_hist), mode='lines', name='Price'))
        # fan samples: plot a subset for performance
        samples = fc.get("samples", [])
        max_paths = min(30, len(samples))
        for i in range(max_paths):
            path = samples[i]
            # align path x starting at last time
            xs = [time_hist[-1] + (j * refresh) for j in range(len(path))]
            fig.add_trace(go.Scatter(x=xs, y=path, mode='lines', line=dict(width=1, color='rgba(0,100,200,0.15)'), showlegend=False))
        fig.update_layout(height=500, title=f"{symbol} Price & Forecast")
        placeholder_chart.plotly_chart(fig, use_container_width=True)

        with placeholder_stats.container():
            st.metric("Last Price", f"{price:.2f}")
            st.metric("Sentiment (EWMA)", f"{snap.get('senti_ewma', 0.0):.3f}")
            st.metric("Vol(5)", f"{snap.get('vol5', 0.0):.4f}")
            st.metric("Vol(15)", f"{snap.get('vol15', 0.0):.4f}")
            st.write("Forecast returns:")
            st.json({k: v for k, v in fc.items() if k != 'samples'})

    time.sleep(refresh)
