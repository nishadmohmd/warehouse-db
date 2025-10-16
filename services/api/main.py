from __future__ import annotations
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from services.ingestor.prices_synthetic import SyntheticPriceStream
from services.ingestor.prices_yfinance_replay import YFinanceReplayStream
from services.ingestor.social_synthetic import SyntheticSocialStream
from services.ingestor.social_reddit import RedditStream
from services.ingestor.news_newsapi import NewsPoller
from services.features.sentiment import SentimentScorer
from services.features.build_window import RollingWindow
from services.model.inference import Forecaster


class State(BaseModel):
    symbol: str
    last_price: float
    snapshot: Dict[str, float]
    forecast: Dict[str, Any]


@dataclass
class SymbolRuntime:
    window: RollingWindow
    last_price: float = 0.0


def get_symbols() -> List[str]:
    env = os.environ.get("SYMBOLS", "AAPL,MSFT,NVDA")
    return [s.strip().upper() for s in env.split(",") if s.strip()]


app = FastAPI(title="AI Stock Sentiment & Forecast")


class Runtime:
    def __init__(self) -> None:
        self.symbols = get_symbols()
        self.refresh_seconds = float(os.environ.get("REFRESH_SECONDS", "1"))
        self.use_synth = os.environ.get("USE_SYNTHETIC", "1") == "1"
        self.senti = SentimentScorer(domain="social")
        self.forecaster = Forecaster(window=60, horizon=5, samples=100)
        self.by_symbol: Dict[str, SymbolRuntime] = {s: SymbolRuntime(window=RollingWindow(window=60)) for s in self.symbols}
        self.clients: List[WebSocket] = []
        self._bg_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._bg_task is None:
            self._bg_task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        # Choose price source (synthetic or yfinance replay)
        if self.use_synth:
            price_stream = SyntheticPriceStream(self.symbols, refresh_seconds=self.refresh_seconds)
        else:
            price_stream = YFinanceReplayStream(self.symbols, period="5d", interval="1m", speedup=60.0, loop=True)
        # Social/news sources
        social_synth = SyntheticSocialStream(self.symbols, refresh_seconds=2.0)
        reddit = RedditStream(self.symbols, interval=10.0)
        news = NewsPoller(self.symbols, interval=20.0)
        price_iter = price_stream.stream().__aiter__()
        social_iter = social_synth.stream().__aiter__()
        reddit_iter = reddit.stream().__aiter__() if reddit.is_available else None
        news_iter = news.stream().__aiter__() if news.is_available else None
        while True:
            # gather one price tick per symbol per second, plus possibly a social event
            tick = await price_iter.__anext__()
            sym_rt = self.by_symbol[tick.symbol]
            sym_rt.window.add_price(tick.ts, tick.price, tick.volume)
            sym_rt.last_price = tick.price
            # opportunistically take social event if available without blocking
            senti_event = None
            try:
                senti_event = await asyncio.wait_for(social_iter.__anext__(), timeout=0.0)
            except Exception:
                pass
            if senti_event is not None:
                scores = self.senti.score_texts([senti_event.text])
                if scores:
                    s = scores[0]
                    sym_rt.window.add_sentiment(senti_event.ts, s["sentiment"], s["confidence"])
            # Reddit event
            if reddit_iter is not None:
                try:
                    r_event = await asyncio.wait_for(reddit_iter.__anext__(), timeout=0.0)
                    scores = self.senti.score_texts([r_event.text])
                    if scores:
                        s2 = scores[0]
                        self.by_symbol[r_event.symbol].window.add_sentiment(r_event.ts, s2["sentiment"], s2["confidence"])
                except Exception:
                    pass
            # News event
            if news_iter is not None:
                try:
                    n_event = await asyncio.wait_for(news_iter.__anext__(), timeout=0.0)
                    text = f"{n_event.title} {n_event.description}"
                    scores = self.senti.score_texts([text])
                    if scores:
                        s3 = scores[0]
                        self.by_symbol[n_event.symbol].window.add_sentiment(n_event.ts, s3["sentiment"], s3["confidence"])
                except Exception:
                    pass
            # compute features and forecast
            X, snap = sym_rt.window.features()
            fc = self.forecaster.forecast(X, sym_rt.last_price)
            payload = {
                "type": "state",
                "symbol": tick.symbol,
                "last_price": sym_rt.last_price,
                "snapshot": snap,
                "forecast": fc,
                "ts": time.time(),
            }
            # broadcast to all clients (simple, no per-sub filtering)
            living = []
            for ws in self.clients:
                try:
                    await ws.send_text(json.dumps(payload))
                    living.append(ws)
                except Exception:
                    pass
            self.clients = living
            await asyncio.sleep(max(0.0, self.refresh_seconds))


runtime = Runtime()


@app.on_event("startup")
async def on_start() -> None:
    await runtime.start()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/state")
def get_state(symbol: Optional[str] = None) -> Dict[str, Any]:
    if not symbol:
        symbol = runtime.symbols[0]
    symbol = symbol.upper()
    rt = runtime.by_symbol.get(symbol)
    if rt is None:
        return {"error": "unknown symbol"}
    X, snap = rt.window.features()
    fc = runtime.forecaster.forecast(X, rt.last_price)
    return {
        "symbol": symbol,
        "last_price": rt.last_price,
        "snapshot": snap,
        "forecast": fc,
        "ts": time.time(),
    }


class InjectPayload(BaseModel):
    symbol: str
    text: str


@app.post("/inject")
def inject_event(payload: InjectPayload) -> Dict[str, Any]:
    symbol = payload.symbol.upper()
    rt = runtime.by_symbol.get(symbol)
    if rt is None:
        return {"error": "unknown symbol"}
    scores = runtime.senti.score_texts([payload.text])
    if scores:
        s = scores[0]
        rt.window.add_sentiment(time.time(), s["sentiment"], s["confidence"])
    X, snap = rt.window.features()
    fc = runtime.forecaster.forecast(X, rt.last_price)
    return {
        "ok": True,
        "symbol": symbol,
        "snapshot": snap,
        "forecast": fc,
    }


@app.websocket("/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    runtime.clients.append(ws)
    try:
        while True:
            # Keep the connection alive; no recv needed
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            runtime.clients.remove(ws)
        except ValueError:
            pass
