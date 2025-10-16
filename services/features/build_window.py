from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PricePoint:
    ts: float
    price: float
    volume: float = 0.0


@dataclass
class SentimentPoint:
    ts: float
    sentiment: float
    confidence: float


class RollingWindow:
    """Maintains rolling feature window for a single symbol.

    Window granularity is seconds. Features include:
      - log return
      - rolling volatility (5s, 15s)
      - EWMA of sentiment
      - count of sentiment events in last 60s
    """

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.prices: Deque[PricePoint] = deque(maxlen=window + 1)
        self.sentiments: Deque[SentimentPoint] = deque(maxlen=window * 10)

    def add_price(self, ts: float, price: float, volume: float = 0.0) -> None:
        self.prices.append(PricePoint(ts=ts, price=price, volume=volume))

    def add_sentiment(self, ts: float, sentiment: float, confidence: float) -> None:
        self.sentiments.append(SentimentPoint(ts=ts, sentiment=sentiment, confidence=confidence))
        # prune old
        cutoff = ts - self.window
        while self.sentiments and self.sentiments[0].ts < cutoff:
            self.sentiments.popleft()

    def _compute_returns(self) -> np.ndarray:
        if len(self.prices) < 2:
            return np.zeros(1)
        prices = np.array([p.price for p in self.prices], dtype=float)
        prices = np.maximum(prices, 1e-8)
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        return returns[-self.window :]

    def _rolling_std(self, arr: np.ndarray, k: int) -> float:
        if len(arr) < 2:
            return 0.0
        k = max(1, min(k, len(arr)))
        window = arr[-k:]
        return float(np.std(window))

    def _sentiment_features(self, now_ts: float) -> Tuple[float, float, float]:
        if not self.sentiments:
            return 0.0, 0.0, 0.0
        # EWMA with half-life ~15s
        alpha = 2.0 / (15.0 + 1.0)
        ewma = 0.0
        weight_sum = 0.0
        count = 0
        bull = 0
        bear = 0
        for s in list(self.sentiments):
            ewma = alpha * s.sentiment + (1 - alpha) * ewma
            weight_sum += alpha if s.sentiment != 0 else 0
            count += 1
            if s.sentiment > 0:
                bull += 1
            elif s.sentiment < 0:
                bear += 1
        bb_ratio = (bull / max(1, bear)) if bear > 0 else float(bull)
        return float(ewma), float(count), float(bb_ratio)

    def features(self) -> Tuple[np.ndarray, Dict[str, float]]:
        now_ts = time.time()
        returns = self._compute_returns()
        vol5 = self._rolling_std(returns, 5)
        vol15 = self._rolling_std(returns, 15)
        senti_ewma, senti_count, bb_ratio = self._sentiment_features(now_ts)
        # Align lengths: use last self.window returns; pad if needed
        if len(returns) < self.window:
            pad = np.zeros(self.window - len(returns))
            returns = np.concatenate([pad, returns])
        # Build per-step features where sentiment is treated as current-state scalar features
        # We repeat scalar sentiment features across the window; model can learn their impact
        senti_feats = np.array([senti_ewma, senti_count, bb_ratio, vol5, vol15], dtype=float)
        X = np.stack([returns] * 1, axis=1)  # shape (window, 1)
        # Append broadcasted scalars as extra channels
        broadcast = np.repeat(senti_feats.reshape(1, -1), self.window, axis=0)  # (window, 5)
        X = np.concatenate([X, broadcast], axis=1)  # (window, 6)
        snapshot = {
            "vol5": vol5,
            "vol15": vol15,
            "senti_ewma": senti_ewma,
            "senti_count": senti_count,
            "bb_ratio": bb_ratio,
        }
        return X.astype(float), snapshot
