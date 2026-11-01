"""Spend ledger: reserve the worst case atomically before every paid call, settle after.

Why reserve-then-settle: with check-then-charge, N concurrent turns all pass the check and
all charge, so caps overshoot by N times a turn. Here `reserve` takes the worst-case cost of
the call (all input tokens uncached, the full output allowance) against every cap at once,
under one lock, and refuses if any cap would be exceeded. `settle` replaces the reservation
with what the model actually billed. A reservation nobody settles (a crashed container)
expires after RESERVATION_TTL_S and is kept at its worst case: the ledger errs on the side of
spending less, never more.

Where the lock lives: one process. Production runs LedgerCore inside a single-container Modal
class (api/app.py, `SpendLedger`, max_containers=1) that every API container calls, and saves
a snapshot to a Modal Dict after each change so a restart keeps the day's totals. Tests and
development use MemoryLedger, the same core in-process.

Caps checked in one reservation:
- the day's LLM spend across all visitors (LLM_USD_PER_DAY),
- one visitor's spend on one record (LLM_USD_PER_RUN; the run id is derived on the server),
- turns per visitor per day and turns per IP per day,
- a token bucket per IP against bursts (IP_BURST turns, refilled at IP_REFILL_PER_S).
"""

from __future__ import annotations

import datetime as dt
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Limits:
	# Measured Oct 27 on muse-spark-1.3 (12 tours, docs/decisions/guide-limits.md): a tour costs
	# $0.033 on average and $0.115 at most, at the guide's price table (muse_spark.PRICE_*, cited
	# but unconfirmed). Source: derived, three of the costliest tours on one record plus one
	# worst-case reservation (5.4k input + 2,000 output tokens, about $0.015).
	usd_per_run: float = 0.40
	# source: contract open question ($10 LLM per day for Nov 5-8), pending Chiya's answer
	usd_per_day: float = 10.0
	# Measured Oct 27: 7.6 rounds per tour on average, 21 at most. Source: derived, three of the
	# longest tours (63), rounded up.
	turns_per_visitor_day: int = 64
	# source: guess, four visitors behind one shared campus or office address
	turns_per_ip_day: int = 256
	# Measured Oct 27: rounds arrive about every 1.7 s (p50 model latency) and at most 4 come back
	# to back before the visitor acts. Source: derived, a burst of 12 and one round per 2 s let
	# the longest measured tour (21 rounds) through without a refusal.
	ip_burst: int = 12
	ip_refill_per_s: float = 0.5
	# source: derived, well past the 12 s model timeout, so a reservation always outlives its call
	reservation_ttl_s: float = 90.0


Refusal = Literal["daily_cap", "run_cap", "visitor_turns", "ip_turns", "ip_burst"]


@dataclass
class Reservation:
	id: str
	visitor: str
	run: str
	ip: str
	usd: float
	at: float
	#: Seconds before an unsettled reservation closes at its worst case; None is the ledger's own.
	ttl: float | None = None


@dataclass
class Decision:
	ok: bool
	reservation: str | None = None
	reason: Refusal | None = None
	retry_after_s: float | None = None


def today_et(now: float | None = None) -> str:
	moment = dt.datetime.fromtimestamp(now if now is not None else time.time(), ET)
	return moment.date().isoformat()


@dataclass
class _Day:
	day: str
	usd: float = 0.0
	turns_visitor: dict[str, int] = field(default_factory=dict)
	turns_ip: dict[str, int] = field(default_factory=dict)
	run_usd: dict[str, float] = field(default_factory=dict)


class LedgerCore:
	"""All state behind one lock. Pure Python; the clock is passed in so tests control time."""

	def __init__(self, limits: Limits | None = None) -> None:
		self.limits = limits or Limits()
		self._lock = threading.Lock()
		self._d = _Day(today_et())
		self._open: dict[str, Reservation] = {}
		self._bucket: dict[str, tuple[float, float]] = {}  # ip -> (tokens, at)

	# ---- internals (call with the lock held)

	def _roll(self, now: float) -> None:
		day = today_et(now)
		if day != self._d.day:
			self._d = _Day(day)
			# reservations from yesterday stay open until settled; they count against today too

	def _expire(self, now: float) -> None:
		ttl = self.limits.reservation_ttl_s
		for rid in [r.id for r in self._open.values() if now - r.at > (r.ttl if r.ttl is not None else ttl)]:
			self._close(rid, None)  # kept at worst case

	def _outstanding(self, *, run: str | None = None) -> float:
		return sum(r.usd for r in self._open.values() if run is None or r.run == run)

	def _take_token(self, ip: str, now: float) -> float | None:
		"""Take one token from the IP's bucket; None if taken, else seconds until one is free."""
		cap, rate = self.limits.ip_burst, self.limits.ip_refill_per_s
		tokens, at = self._bucket.get(ip, (float(cap), now))
		tokens = min(float(cap), tokens + (now - at) * rate)
		if tokens < 1.0:
			self._bucket[ip] = (tokens, now)
			# a bucket that never refills (rate 0) reports a day's wait
			return (1.0 - tokens) / rate if rate > 0 else 86_400.0
		self._bucket[ip] = (tokens - 1.0, now)
		return None

	def _close(self, rid: str, usd: float | None) -> None:
		r = self._open.pop(rid, None)
		if r is None:
			return
		spent = r.usd if usd is None else max(0.0, usd)
		self._d.usd += spent
		self._d.run_usd[r.run] = self._d.run_usd.get(r.run, 0.0) + spent

	# ---- public

	def reserve(
		self,
		*,
		visitor: str,
		run: str,
		ip: str,
		usd: float,
		now: float | None = None,
		run_cap: float | None = None,
		ttl_s: float | None = None,
	) -> Decision:
		"""`run_cap` replaces the per-run cap for this run (the fix loop's per-address budget);
		`ttl_s` keeps a reservation open longer than a guide turn (a fix race runs minutes)."""
		now = time.time() if now is None else now
		lim = self.limits
		with self._lock:
			self._roll(now)
			self._expire(now)
			d = self._d
			if d.usd + self._outstanding() + usd > lim.usd_per_day:
				return Decision(False, reason="daily_cap")
			cap = lim.usd_per_run if run_cap is None else run_cap
			if d.run_usd.get(run, 0.0) + self._outstanding(run=run) + usd > cap:
				return Decision(False, reason="run_cap")
			if d.turns_visitor.get(visitor, 0) >= lim.turns_per_visitor_day:
				return Decision(False, reason="visitor_turns")
			if d.turns_ip.get(ip, 0) >= lim.turns_per_ip_day:
				return Decision(False, reason="ip_turns")
			wait = self._take_token(ip, now)
			if wait is not None:
				return Decision(False, reason="ip_burst", retry_after_s=round(wait, 1))
			rid = secrets.token_urlsafe(12)
			self._open[rid] = Reservation(rid, visitor, run, ip, usd, now, ttl_s)
			d.turns_visitor[visitor] = d.turns_visitor.get(visitor, 0) + 1
			d.turns_ip[ip] = d.turns_ip.get(ip, 0) + 1
			return Decision(True, reservation=rid)

	def settle(self, reservation: str, usd: float) -> None:
		"""Replace a reservation with the billed amount. Idempotent; an unknown id is ignored."""
		with self._lock:
			self._close(reservation, usd)

	def release(self, reservation: str) -> None:
		"""The call never happened (the model was unreachable): refund the money and the turn."""
		with self._lock:
			r = self._open.pop(reservation, None)
			if r is None:
				return
			for counts, key in ((self._d.turns_visitor, r.visitor), (self._d.turns_ip, r.ip)):
				if counts.get(key, 0) > 0:
					counts[key] -= 1

	def spent_today(self, now: float | None = None) -> float:
		now = time.time() if now is None else now
		with self._lock:
			self._roll(now)
			self._expire(now)
			return self._d.usd + self._outstanding()

	def snapshot(self) -> dict:
		with self._lock:
			return {
				"day": asdict(self._d),
				# A reservation's own ttl is kept beside the rows, not in them, so a ledger from
				# before ttl existed can still restore this snapshot if the API is rolled back.
				"open": [{k: v for k, v in asdict(r).items() if k != "ttl"} for r in self._open.values()],
				"open_ttl": {r.id: r.ttl for r in self._open.values() if r.ttl is not None},
				"bucket": {k: list(v) for k, v in self._bucket.items()},
			}

	@classmethod
	def restore(cls, snap: dict, limits: Limits | None = None) -> LedgerCore:
		core = cls(limits)
		day = snap.get("day") or {}
		core._d = _Day(
			day=day.get("day", today_et()),
			usd=float(day.get("usd", 0.0)),
			turns_visitor=dict(day.get("turns_visitor", {})),
			turns_ip=dict(day.get("turns_ip", {})),
			run_usd=dict(day.get("run_usd", {})),
		)
		ttls = snap.get("open_ttl") or {}
		core._open = {
			r["id"]: Reservation(**{**r, "ttl": ttls.get(r["id"], r.get("ttl"))})
			for r in snap.get("open", [])
		}
		core._bucket = {k: (float(v[0]), float(v[1])) for k, v in (snap.get("bucket") or {}).items()}
		return core


class Ledger(Protocol):
	async def reserve(
		self,
		*,
		visitor: str,
		run: str,
		ip: str,
		usd: float,
		run_cap: float | None = None,
		ttl_s: float | None = None,
	) -> Decision: ...
	async def settle(self, reservation: str, usd: float) -> None: ...
	async def release(self, reservation: str) -> None: ...
	async def spent_today(self) -> float: ...


class MemoryLedger:
	"""In-process: correct for one container (tests, development)."""

	def __init__(self, core: LedgerCore | None = None) -> None:
		self.core = core or LedgerCore()

	async def reserve(
		self,
		*,
		visitor: str,
		run: str,
		ip: str,
		usd: float,
		run_cap: float | None = None,
		ttl_s: float | None = None,
	) -> Decision:
		return self.core.reserve(visitor=visitor, run=run, ip=ip, usd=usd, run_cap=run_cap, ttl_s=ttl_s)

	async def settle(self, reservation: str, usd: float) -> None:
		self.core.settle(reservation, usd)

	async def release(self, reservation: str) -> None:
		self.core.release(reservation)

	async def spent_today(self) -> float:
		return self.core.spent_today()


class ModalLedger:
	"""Calls the single-container SpendLedger service (api/app.py) from any API container."""

	def __init__(self, app_name: str = "preflight-api", cls_name: str = "SpendLedger") -> None:
		import modal  # imported here so tests and development never need it

		self._svc = modal.Cls.from_name(app_name, cls_name)()

	async def reserve(
		self,
		*,
		visitor: str,
		run: str,
		ip: str,
		usd: float,
		run_cap: float | None = None,
		ttl_s: float | None = None,
	) -> Decision:
		raw = await self._svc.reserve.remote.aio(
			visitor=visitor, run=run, ip=ip, usd=usd, run_cap=run_cap, ttl_s=ttl_s
		)
		return Decision(**raw)

	async def settle(self, reservation: str, usd: float) -> None:
		await self._svc.settle.remote.aio(reservation, usd)

	async def release(self, reservation: str) -> None:
		await self._svc.release.remote.aio(reservation)

	async def spent_today(self) -> float:
		return float(await self._svc.spent_today.remote.aio())
