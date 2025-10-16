from __future__ import annotations
import asyncio
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

import requests


@dataclass
class NewsEvent:
    ts: float
    symbol: str
    title: str
    description: str
    url: str


class NewsPoller:
    """Polls NewsAPI for each symbol on an interval and yields events.

    Requires NEWSAPI_KEY. Uses simple per-symbol queries.
    """

    def __init__(self, symbols: List[str], interval: float = 15.0) -> None:
        self.symbols = symbols
        self.interval = interval
        self.api_key = os.environ.get("NEWSAPI_KEY")
        self._seen: Dict[str, set[str]] = {s: set() for s in symbols}

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _query_symbol(self, symbol: str) -> List[NewsEvent]:
        if not self.api_key:
            return []
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": self.api_key,
        }
        events: List[NewsEvent] = []
        try:
            r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
            if not r.ok:
                return []
            data = r.json()
            for art in data.get("articles", []):
                url = art.get("url") or ""
                if not url or url in self._seen[symbol]:
                    continue
                self._seen[symbol].add(url)
                title = art.get("title") or ""
                desc = art.get("description") or ""
                events.append(NewsEvent(ts=time.time(), symbol=symbol, title=title, description=desc, url=url))
        except Exception:
            return []
        return events

    async def stream(self) -> AsyncIterator[NewsEvent]:
        while True:
            start = time.time()
            for s in self.symbols:
                for ev in self._query_symbol(s):
                    yield ev
            spent = time.time() - start
            await asyncio.sleep(max(0.0, self.interval - spent))
