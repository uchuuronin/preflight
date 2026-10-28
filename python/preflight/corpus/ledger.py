"""Wall-clock spend for machines that bill for uptime, not per config (Review, Oct 27: the AMD droplet).

A droplet bills from creation to destruction, idle time and restarts included, so the sweep's spend is
the time since the droplet started, times its hourly rate, whatever ran in between. Each start, config
and stop is appended to a JSONL ledger next to the results, so the spend can be checked afterwards.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path


class UptimeLedger:
	def __init__(
		self,
		path: str | Path,
		started_at: datetime,
		usd_per_hour: float,
		clock: Callable[[], float] = time.time,
	):
		if started_at.tzinfo is None:
			raise ValueError("give the droplet's start time with a time zone, e.g. 2026-10-28T09:00:00-04:00")
		self.path = Path(path)
		self.started = started_at.timestamp()
		self.rate = usd_per_hour
		self.clock = clock

	def spent(self) -> float:
		"""Dollars billed so far: hours since the droplet started, times its rate."""
		return max(0.0, self.clock() - self.started) / 3600 * self.rate

	def record(self, event: str, **fields) -> None:
		self.path.parent.mkdir(parents=True, exist_ok=True)
		now = self.clock()
		row = {
			"t": datetime.fromtimestamp(now, UTC).isoformat(timespec="seconds"),
			"event": event,
			"uptime_s": round(max(0.0, now - self.started), 1),
			"spent_usd": round(self.spent(), 4),
			**fields,
		}
		with self.path.open("a") as f:
			f.write(json.dumps(row, sort_keys=True) + "\n")

	def rows(self) -> list[dict]:
		if not self.path.exists():
			return []
		return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
