"""Rate limiting and request size guards."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock

from fastapi import HTTPException, Request


@dataclass
class RateLimiter:
    max_per_minute: int
    _events: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: Lock = field(default_factory=Lock)

    def check(self, client_key: str) -> None:
        if self.max_per_minute <= 0:
            return
        now = time.monotonic()
        window = 60.0
        with self._lock:
            bucket = self._events[client_key]
            while bucket and now - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= self.max_per_minute:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            bucket.append(now)


def client_key_from_request(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def enforce_body_size(request: Request, max_bytes: int) -> None:
    body = await request.body()
    if len(body) > max_bytes:
        raise HTTPException(status_code=413, detail="Request body too large")
