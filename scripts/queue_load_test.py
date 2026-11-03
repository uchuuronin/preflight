"""Load test of the probe queue's per-visitor cap, against a copy of the job store (never production).

	PYTHONPATH=python python scripts/queue_load_test.py [--visitors 20] [--seed 1]

The copy is `runner.queue.InMemoryJobStore`, the class whose logic the deployed job store wraps, on a simulated
clock (its clock is injectable), so the leasing, the per-visitor running limit, the daily job limit and the
in-flight budget hold are the production code's own. 20 visitors arrive over the first 10 minutes (a link shared
in a channel), each wanting the daily allowance of 3 probe jobs and submitting the next as soon as the queue
takes it (a refusal is retried after its Retry-After). Eight workers (MAX_RUNNING) lease jobs and run them for a
drawn duration; CPU-only jobs, so no GPU waits are modelled. Job length is a guess: lognormal, median 200 s,
95th percentile about 600 s (today's check-ins ran 1.5 to 8 minutes end to end, Nov 2). Prints, per cap: queue
wait of the jobs taken (p50, p95, max), refusals, and how long the last job finished after the first arrival.
"""

import argparse
import math
import random
import statistics
import sys

from preflight.runner import queue as Q

TICK_S = 1.0
ARRIVAL_WINDOW_S = 600.0
JOB_MEDIAN_S, JOB_SIGMA = 200.0, 0.67  # guess: lognormal, exp(1.645 * 0.67) * 200 is about 600 s at p95


def run(cap: int, visitors: int, seed: int) -> dict:
	rng = random.Random(seed)
	now = [1_762_000_000.0]  # a fixed day, so the ET daily counters behave as on one day
	store = Q.InMemoryJobStore(clock=lambda: now[0], max_queued_per_visitor=cap)
	arrive = sorted(rng.uniform(0, ARRIVAL_WINDOW_S) for _ in range(visitors))
	want = {f"v{i}": Q.VISITOR_DAILY_JOBS for i in range(visitors)}
	next_try = {f"v{i}": t for i, t in enumerate(arrive)}
	submitted: dict[str, float] = {}
	waits, refused, running = [], 0, {}
	sent = {v: 0 for v in want}
	t0, t, last_end = now[0], 0.0, 0.0
	while (any(want.values()) or store.queued() or running) and t < 6 * 3600:
		now[0] = t0 + t
		for v in want:
			if want[v] and next_try[v] <= t:
				try:
					s = store.submit({"source": {"kind": "sample", "uri": "sample://zoo/x"}, "n": sent[v]}, visitor=v,
						idempotency_key=f"{v}-{sent[v]}")  # fmt: skip
					submitted[s.job_id] = t
					sent[v] += 1
					want[v] -= 1
					next_try[v] = t  # the next one at once, if the queue takes it
				except Q.QueueFull as exc:
					refused += 1
					if exc.code != "queue_full":  # a daily limit: this visitor is done for the day
						want[v] = 0
					next_try[v] = t + max(1, exc.retry_after_s)
		for jid, (lease, end) in list(running.items()):
			if end <= t:
				store.complete(lease, record_id=f"r-{jid}", spent_usd=0.0)
				del running[jid]
				last_end = t
			else:
				store.heartbeat(lease)
		while len(running) < Q.MAX_RUNNING:
			lease = store.lease(f"w{len(running)}")
			if lease is None:
				break
			waits.append(t - submitted[lease.job_id])
			running[lease.job_id] = (lease, t + rng.lognormvariate(math.log(JOB_MEDIAN_S), JOB_SIGMA))
		t += TICK_S
	waits.sort()
	p = lambda q: waits[min(len(waits) - 1, int(q * len(waits)))]  # noqa: E731
	return {"cap": cap, "jobs": len(waits), "wait_p50_s": p(0.5), "wait_p95_s": p(0.95), "wait_max_s": waits[-1],
		"refusals": refused, "last_done_s": last_end}  # fmt: skip


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--visitors", type=int, default=20)
	ap.add_argument("--seeds", type=int, default=20)
	a = ap.parse_args()
	print(f"{a.visitors} visitors x {Q.VISITOR_DAILY_JOBS} jobs, {Q.MAX_RUNNING} workers, {a.seeds} seeds (medians over seeds)")
	for cap in (1, 2, 3, 5):
		rows = [run(cap, a.visitors, s) for s in range(a.seeds)]
		med = {k: statistics.median(r[k] for r in rows) for k in rows[0] if k != "cap"}
		print(f"cap {cap}: {med['jobs']:.0f} jobs, queue wait p50 {med['wait_p50_s']:.0f} s, p95 {med['wait_p95_s']:.0f} s, "
			f"max {med['wait_max_s']:.0f} s; refusals {med['refusals']:.0f}; last job done at {med['last_done_s'] / 60:.0f} min")  # fmt: skip
	return 0


if __name__ == "__main__":
	sys.exit(main())
