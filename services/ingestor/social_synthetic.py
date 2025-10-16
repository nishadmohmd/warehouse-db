from __future__ import annotations
import asyncio
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, List


POS = [
    "beats earnings",
    "strong guidance",
    "innovation breakthrough",
    "analyst upgrade",
    "record revenue",
]
NEG = [
    "misses estimates",
    "supply chain issues",
    "regulatory probe",
    "data breach",
    "downgrade",
]
NEU = [
    "product release",
    "market open",
    "executive interview",
    "industry conference",
    "partnership announcement",
]


@dataclass
class SocialEvent:
    ts: float
    symbol: str
    text: str


class SyntheticSocialStream:
    def __init__(self, symbols: List[str], refresh_seconds: float = 2.0) -> None:
        self.symbols = symbols
        self.refresh_seconds = refresh_seconds

    async def stream(self) -> AsyncIterator[SocialEvent]:
        while True:
            now = time.time()
            symbol = random.choice(self.symbols)
            bucket = random.choices([POS, NEG, NEU], weights=[0.4, 0.3, 0.3])[0]
            phrase = random.choice(bucket)
            text = f"{symbol} {phrase}"
            yield SocialEvent(ts=now, symbol=symbol, text=text)
            await asyncio.sleep(self.refresh_seconds)
