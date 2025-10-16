from __future__ import annotations
import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List, Optional

try:
    import praw  # type: ignore
except Exception:
    praw = None  # type: ignore


@dataclass
class SocialEvent:
    ts: float
    symbol: str
    text: str


class RedditStream:
    """Streams Reddit submissions from a few finance subreddits and maps to symbols.

    Requires Reddit API credentials. If unavailable or import fails, no events will be yielded.
    """

    def __init__(self, symbols: List[str], subreddits: Optional[List[str]] = None, interval: float = 10.0) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.subreddits = subreddits or ["stocks", "wallstreetbets"]
        self.interval = interval
        self._reddit = None
        if praw is not None and os.environ.get("REDDIT_CLIENT_ID"):
            try:
                self._reddit = praw.Reddit(
                    client_id=os.environ.get("REDDIT_CLIENT_ID"),
                    client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
                    user_agent=os.environ.get("REDDIT_USER_AGENT", "stock-ai-app"),
                )
            except Exception:
                self._reddit = None
        self._sym_re = re.compile(r"\b(" + "|".join(re.escape(s) for s in self.symbols) + r")\b", re.IGNORECASE)

    @property
    def is_available(self) -> bool:
        return self._reddit is not None

    def _iter_texts(self) -> Iterable[SocialEvent]:
        if self._reddit is None:
            return []
        for sub in self.subreddits:
            try:
                subreddit = self._reddit.subreddit(sub)
                for post in subreddit.new(limit=25):
                    text = f"{post.title} {post.selftext or ''}"
                    m = self._sym_re.search(text)
                    if not m:
                        continue
                    sym = m.group(1).upper()
                    yield SocialEvent(ts=time.time(), symbol=sym, text=text[:500])
            except Exception:
                continue

    async def stream(self) -> AsyncIterator[SocialEvent]:
        while True:
            any_yield = False
            for ev in self._iter_texts():
                any_yield = True
                yield ev
            await asyncio.sleep(self.interval if any_yield else max(2.0, self.interval))
