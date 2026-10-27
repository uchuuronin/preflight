"""Guide spend limits: turns per visitor per day, tokens per turn and per run, and the
global daily LLM cap. Exceeding any of them switches the visitor to the rule-based tour,
never to an error page.

In-memory for now; the shared ledger (gap 2, A) replaces MemoryLedger behind the same
interface so limits hold across API containers.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from typing import Protocol
from zoneinfo import ZoneInfo

TURNS_PER_VISITOR_DAY = 30
MAX_OUTPUT_TOKENS_PER_TURN = 1200
MAX_INPUT_TOKENS_PER_TURN = 60_000
LLM_USD_PER_RUN = 0.40
LLM_USD_PER_DAY = 10.0
ALERT_AT = 0.8

ET = ZoneInfo("America/New_York")


def today_et() -> str:
	return dt.datetime.now(ET).date().isoformat()


@dataclass
class Decision:
	allowed: bool
	reason: str | None = None


class Ledger(Protocol):
	def check(self, visitor: str, record: str | None) -> Decision: ...
	def charge(self, visitor: str, record: str | None, usd: float) -> None: ...
	def spent_today(self) -> float: ...


class MemoryLedger:
	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._day = today_et()
		self._turns: dict[str, int] = {}
		self._run_usd: dict[str, float] = {}
		self._day_usd = 0.0

	def _roll(self) -> None:
		day = today_et()
		if day != self._day:
			self._day, self._turns, self._day_usd = day, {}, 0.0

	def check(self, visitor: str, record: str | None) -> Decision:
		with self._lock:
			self._roll()
			if self._day_usd >= LLM_USD_PER_DAY:
				return Decision(False, "daily_cap")
			if self._turns.get(visitor, 0) >= TURNS_PER_VISITOR_DAY:
				return Decision(False, "visitor_turns")
			if record and self._run_usd.get(record, 0.0) >= LLM_USD_PER_RUN:
				return Decision(False, "run_cap")
			return Decision(True)

	def charge(self, visitor: str, record: str | None, usd: float) -> None:
		with self._lock:
			self._roll()
			self._turns[visitor] = self._turns.get(visitor, 0) + 1
			self._day_usd += usd
			if record:
				self._run_usd[record] = self._run_usd.get(record, 0.0) + usd

	def spent_today(self) -> float:
		with self._lock:
			self._roll()
			return self._day_usd
