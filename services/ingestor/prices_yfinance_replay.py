from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

import yfinance as yf


@dataclass
class PriceTick:
    ts: float
    symbol: str
    price: float
    volume: float


_INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
}


class YFinanceReplayStream:
    """Replays historical minute bars as a live stream.

    If data ends and loop=True, it restarts from the beginning.
    Speed can be increased using `speedup` (e.g., 60 => 1m bar per second).
    """

    def __init__(
        self,
        symbols: List[str],
        period: str = "5d",
        interval: str = "1m",
        speedup: float = 60.0,
        loop: bool = True,
    ) -> None:
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.speedup = max(1.0, speedup)
        self.loop = loop
        self._data: Dict[str, List[Dict[str, float]]] = {}
        self._idx: Dict[str, int] = {s: 0 for s in symbols}
        self._load()

    def _load(self) -> None:
        for s in self.symbols:
            try:
                df = yf.download(s, period=self.period, interval=self.interval, progress=False, threads=False)
                if df is None or df.empty:
                    self._data[s] = []
                    continue
                rows: List[Dict[str, float]] = []
                for idx, row in df.iterrows():
                    price = float(row.get("Close", row.get("Adj Close", 0.0)))
                    volume = float(row.get("Volume", 0.0))
                    rows.append({"price": price, "volume": volume})
                self._data[s] = rows
            except Exception:
                self._data[s] = []

    async def stream(self) -> AsyncIterator[PriceTick]:
        interval_sec = _INTERVAL_SECONDS.get(self.interval, 60)
        sleep_s = interval_sec / self.speedup
        while True:
            start = time.time()
            for s in self.symbols:
                data = self._data.get(s, [])
                if not data:
                    # No data; skip
                    continue
                i = self._idx[s]
                row = data[i]
                tick = PriceTick(ts=time.time(), symbol=s, price=row["price"], volume=row["volume"])
                yield tick
                i += 1
                if i >= len(data):
                    if self.loop:
                        i = 0
                    else:
                        i = len(data) - 1
                self._idx[s] = i
            spent = time.time() - start
            await asyncio.sleep(max(0.0, sleep_s - spent))
