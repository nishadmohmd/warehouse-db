from __future__ import annotations
import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional


@dataclass
class PriceTick:
    ts: float
    symbol: str
    price: float
    volume: float


class SyntheticPriceStream:
    def __init__(self, symbols: List[str], refresh_seconds: float = 1.0) -> None:
        self.symbols = symbols
        self.refresh_seconds = refresh_seconds
        self._state: Dict[str, float] = {s: 100.0 + 10.0 * random.random() for s in symbols}

    async def _tick_symbol(self, symbol: str) -> PriceTick:
        last = self._state[symbol]
        # geometric brownian motion step
        dt = self.refresh_seconds / 60.0  # treat 60s as 1 unit
        mu = 0.0
        sigma = 0.02
        epsilon = random.gauss(0.0, 1.0)
        ret = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * epsilon
        price = max(1e-3, last * math.exp(ret))
        self._state[symbol] = price
        vol = abs(epsilon) * 10.0
        return PriceTick(ts=time.time(), symbol=symbol, price=price, volume=vol)

    async def stream(self) -> AsyncIterator[PriceTick]:
        while True:
            start = time.time()
            for s in self.symbols:
                yield await self._tick_symbol(s)
            spent = time.time() - start
            await asyncio.sleep(max(0.0, self.refresh_seconds - spent))
