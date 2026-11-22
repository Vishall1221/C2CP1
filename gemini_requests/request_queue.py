"""
request_queue.py — rate-limited async request queue (stdlib only; API call is injected).

Handles what the SDK's built-in retry does badly:
  * token bucket for requests/minute shared by all workers
  * 429 minute-limit  -> shared cooldown that honours the server's retryDelay
  * 429 day-limit     -> stop cleanly, remaining jobs are `deferred` and finish on resume
  * 5xx / timeouts    -> exponential backoff with jitter
  * invalid output    -> up to N repair retries with the validation error appended
  * slow request      -> after `hedge_after` s a duplicate is sent (if the rate cap has room)
                         and the first answer wins; ~1 in 5 requests hung to the 45 s timeout
  * every success is stored immediately (SQLite) so an interrupted run resumes for free
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class ValidationFailure(ValueError):
    """The model answered but the JSON did not validate. The queue retries with a repair hint."""


@dataclass
class Job:
    key: str            # cache key (content addressed)
    outfit_id: str
    priority: float     # higher = sent first when the daily budget is short
    payload: Any = None


@dataclass
class Outcome:
    job: Job
    status: str                         # ok | failed | deferred
    response_json: str | None = None
    error: str | None = None
    attempts: int = 0
    cached: bool = False


@dataclass
class Failure:
    kind: str                           # rate_minute | rate_day | transient | fatal | fatal_global
    retry_after: float | None
    message: str


# ── Classification ────────────────────────────────────────────────────────────

_DELAY = re.compile(r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.I)
_IN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I)


def classify(exc: BaseException) -> Failure:
    """Duck-typed: works with google.genai.errors.APIError and with plain exceptions."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = " ".join(str(x) for x in (exc, getattr(exc, "message", ""), getattr(exc, "details", "")) if x)
    delay = None
    m = _DELAY.search(text) or _IN.search(text)
    if m:
        delay = float(m.group(1))
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if delay is None and headers:
        try:
            delay = float(headers.get("Retry-After"))
        except (TypeError, ValueError):
            pass
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        # Classify by quota id, not by the (sometimes misleading) delay.
        return Failure("rate_day" if "PerDay" in text else "rate_minute", delay, text[:300])
    if code in (401, 403, 404):
        return Failure("fatal_global", None, text[:300])
    if any(k in type(exc).__name__ for k in ("DefaultCredentialsError", "RefreshError")):
        return Failure("fatal_global", None,
                       "Google Cloud credentials missing or expired: run `gcloud auth application-default "
                       f"login` or set GOOGLE_APPLICATION_CREDENTIALS in config.env. ({text[:200]})")
    if code == 400:
        return Failure("fatal", None, text[:300])
    if code in (408, 500, 502, 503, 504) or isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
        return Failure("transient", delay, text[:300])
    name = type(exc).__name__.lower()
    if any(k in name for k in ("timeout", "transport", "connect", "readerror")):
        return Failure("transient", delay, text[:300])
    return Failure("fatal", None, text[:300])


# ── Token bucket ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Requests-per-minute token bucket, a hard cap of `rpm` requests in any 60 s, and a
    shared cooldown.

    The bucket alone let a full burst (rpm) through and then kept refilling, so ~19
    requests went out in the first 35 s at rpm=12. Gemini counts requests per minute
    window, answered 429 and imposed a ~45 s cooldown (47 s of an 83 s stage in a test run).
    The 60 s window cap keeps the client under the server's count instead.
    """

    def __init__(self, rpm: float, *, burst: float | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable] = asyncio.sleep):
        self.rate = max(rpm, 0.001) / 60.0
        self.capacity = float(burst if burst is not None else max(1.0, rpm))
        self.tokens = self.capacity
        self._clock, self._sleep = clock, sleep
        self._t = clock()
        self._blocked_until = 0.0
        self._lock: asyncio.Lock | None = None
        self.window_cap = max(1, int(rpm))
        self._grants: deque = deque()            # times of requests in the last 60 s

    def cooldown(self, seconds: float) -> None:
        self._blocked_until = max(self._blocked_until, self._clock() + max(0.0, seconds))

    def try_acquire(self) -> bool:
        """Take a request slot only if one is free right now (used for hedged duplicates)."""
        now = self._clock()
        if now < self._blocked_until:
            return False
        while self._grants and now - self._grants[0] >= 60.0:
            self._grants.popleft()
        self.tokens = min(self.capacity, self.tokens + (now - self._t) * self.rate)
        self._t = now
        if len(self._grants) >= self.window_cap or self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        self._grants.append(now)
        return True

    async def acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:                      # FIFO among waiters
            while True:
                now = self._clock()
                if now < self._blocked_until:
                    await self._sleep(self._blocked_until - now)
                    continue
                while self._grants and now - self._grants[0] >= 60.0:
                    self._grants.popleft()
                if len(self._grants) >= self.window_cap:
                    await self._sleep(self._grants[0] + 60.0 - now + 0.05)
                    continue
                self.tokens = min(self.capacity, self.tokens + (now - self._t) * self.rate)
                self._t = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self._grants.append(now)
                    return
                await self._sleep((1.0 - self.tokens) / self.rate)


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_jobs(
    jobs: list[Job],
    call: Callable[[Job, str | None], Awaitable[str]],
    store,
    limiter: RateLimiter,
    *,
    concurrency: int = 4,
    rpd: int = 450,
    max_attempts: int = 6,
    repair_retries: int = 2,
    on_done: Callable[[Outcome, int, int], None] | None = None,
    on_retry: Callable[..., None] | None = None,      # (kind, wait_s, message)
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
    rng: random.Random | None = None,
    hedge_after: float | None = None,
) -> list[Outcome]:
    """
    `call(job, repair_hint)` performs ONE request and returns the validated JSON text,
    raising ValidationFailure for bad output or any API exception.
    Returns outcomes in the order of `jobs`.
    """
    rng = rng or random.Random()
    outcomes: dict[str, Outcome] = {}
    total = len(jobs)

    def finish(o: Outcome) -> None:
        outcomes[o.job.key] = o
        if on_done:
            on_done(o, len(outcomes), total)

    todo: list[Job] = []
    for job in jobs:
        cached = store.get_response(job.key)
        if cached is not None:
            finish(Outcome(job, "ok", cached, attempts=0, cached=True))
        elif job.key not in outcomes:
            todo.append(job)

    todo.sort(key=lambda j: -j.priority)
    budget = max(0, rpd - store.daily_count())
    stop = asyncio.Event()
    stop_reason = {"text": ""}
    queue: asyncio.Queue = asyncio.Queue()
    for job in todo:
        queue.put_nowait(job)

    async def attempt(job: Job, repair: str | None) -> str:
        """One logical attempt; if it is still unanswered after `hedge_after` s, race a duplicate."""
        if hedge_after is None:
            return await call(job, repair)
        first = asyncio.ensure_future(call(job, repair))
        done, _ = await asyncio.wait({first}, timeout=hedge_after)
        if done or stop.is_set() or store.daily_count() >= rpd or not limiter.try_acquire():
            return await first
        store.bump_daily()
        if on_retry:
            on_retry("hedge", 0.0, f"no answer after {hedge_after:.0f} s")
        pending = {first, asyncio.ensure_future(call(job, repair))}
        error: BaseException | None = None
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is None:
                    for other in pending:
                        other.cancel()
                    return task.result()
                error = task.exception()
        raise error

    def defer(job: Job, why: str) -> None:
        store.put_failure(job.key, job.outfit_id, "deferred", why)
        finish(Outcome(job, "deferred", error=why))

    async def handle(job: Job) -> None:
        repair: str | None = None
        repairs = 0
        attempts = 0
        while attempts < max_attempts:
            if stop.is_set():
                return defer(job, stop_reason["text"] or "stopped")
            if store.daily_count() >= rpd:
                stop.set()
                stop_reason["text"] = "Daily request budget reached; run again after the quota resets."
                return defer(job, stop_reason["text"])
            await limiter.acquire()
            attempts += 1
            store.bump_daily()
            try:
                text = await attempt(job, repair)
            except ValidationFailure as exc:
                if repairs >= repair_retries:
                    store.put_failure(job.key, job.outfit_id, "failed", f"invalid output: {exc}")
                    return finish(Outcome(job, "failed", error=f"invalid output: {exc}", attempts=attempts))
                repairs += 1
                repair = str(exc)
                if on_retry:
                    on_retry("invalid", 0.0, repair)
                continue
            except Exception as exc:  # noqa: BLE001
                f = classify(exc)
                if f.kind == "rate_day":
                    stop.set()
                    stop_reason["text"] = "Gemini daily quota exhausted; remaining outfits resume after reset."
                    return defer(job, stop_reason["text"])
                if f.kind == "fatal_global":
                    stop.set()
                    stop_reason["text"] = f"Gemini rejected the request: {f.message}"
                    store.put_failure(job.key, job.outfit_id, "failed", f.message)
                    return finish(Outcome(job, "failed", error=f.message, attempts=attempts))
                if f.kind == "fatal":
                    store.put_failure(job.key, job.outfit_id, "failed", f.message)
                    return finish(Outcome(job, "failed", error=f.message, attempts=attempts))
                wait = f.retry_after if f.retry_after is not None else min(60.0, 2.0 ** attempts)
                wait += rng.uniform(0.0, 1.0)
                if on_retry:
                    on_retry(f.kind, wait, f.message)
                if f.kind == "rate_minute":
                    limiter.cooldown(wait)          # everyone waits, not just this job
                else:
                    await sleep(wait)
                continue
            store.put_response(job.key, job.outfit_id, text)
            return finish(Outcome(job, "ok", text, attempts=attempts))
        err = f"gave up after {attempts} attempts"
        store.put_failure(job.key, job.outfit_id, "failed", err)
        finish(Outcome(job, "failed", error=err, attempts=attempts))

    admitted = 0

    async def worker() -> None:
        nonlocal admitted
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if admitted >= budget:
                defer(job, "Over the daily request budget; will run on the next day.")
                continue
            admitted += 1
            await handle(job)

    await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    return [outcomes[j.key] for j in jobs if j.key in outcomes]
