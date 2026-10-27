"""Totals for an agent or API run from a sample of its tasks: cost, tokens and the runaway rate.

A runs 20-50 of the run's tasks for real (`AgentSample`); this projects the full run.
The interval is a percentile bootstrap of the sample mean, times the task count, with
the upper end raised for runaway tasks the sample may have missed (A's D3 rerun,
docs/decisions/probe-plan.md):

	hi += population x (upper bound on the runaway rate - observed rate) x runaway cost

The upper bound is a one-sided Clopper-Pearson bound at 80%, applied whether or not a
runaway was seen; a runaway is a task above 8x the sample median, and when none was
seen it is priced at 20x the median. When most tasks cost nothing (cached), the median
is 0 and A's rule would call every paid task a runaway; B uses the median of the paid
tasks instead. A plain bootstrap covers 71% at 50 tasks; this
covers 89% (A's D3; B's re-check with this code is S6 in docs/decisions/engine-settings.md).

Pure and Pyodide-safe: no I/O; the bootstrap is seeded, so the same sample always
gives the same interval.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

MODEL_ID = "agent-bootstrap-cp@0.1.0"
LEVEL = 0.9
RUNAWAY_FACTOR = 8.0  # a task above 8x the sample median is a runaway
UNSEEN_RUNAWAY_MULT = 20.0  # price of a runaway nobody saw: 20x the median
RUNAWAY_BOUND = 0.80  # one-sided Clopper-Pearson level (D3 rerun: 89.1% coverage at 50 tasks)
N_BOOT = 2000
MIN_SAMPLE = 10  # below this the bootstrap is noise; abstain and say how many more to run
BOOT_CHUNK_CELLS = 1_000_000  # bootstrap draws held at once: about 8 MB, whatever the sample size


# ---------------------------------------------------------------- binomial bounds


def _binom_cdf(x: int, n: int, p: float) -> float:
	"""P(X <= x) for X ~ Binomial(n, p), summed in log space so large n stays stable."""
	if p <= 0.0:
		return 1.0
	if p >= 1.0:
		return 1.0 if x >= n else 0.0
	lp, lq = math.log(p), math.log1p(-p)
	terms = [
		math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
		for i in range(x + 1)
	]
	top = max(terms)
	return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def cp_upper(x: int, n: int, conf: float) -> float:
	"""One-sided Clopper-Pearson upper bound on a binomial rate: x successes in n trials."""
	if n <= 0:
		return 1.0
	if x >= n:
		return 1.0
	lo, hi = x / n, 1.0
	for _ in range(60):
		mid = (lo + hi) / 2
		lo, hi = (mid, hi) if _binom_cdf(x, n, mid) > 1 - conf else (lo, mid)
	return hi


def cp_lower(x: int, n: int, conf: float) -> float:
	"""One-sided Clopper-Pearson lower bound."""
	if x <= 0 or n <= 0:
		return 0.0
	return 1.0 - cp_upper(n - x, n, conf)


# ---------------------------------------------------------------- the estimate


@dataclass(frozen=True)
class TotalEstimate:
	point: float | None
	lo: float | None
	hi: float | None
	unit: str
	status: str  # measured | predicted | extrapolated | abstain
	model_id: str = MODEL_ID
	reason: str | None = None
	level: float = LEVEL
	detail: dict = field(default_factory=dict)  # sample size, runaways, the tail added, excluded tasks

	def to_estimate(self):
		"""The schema's Estimate (source exact: computed for this run, not read off a grid)."""
		from preflight.schema.prediction import Estimate

		return Estimate(
			point=self.point,
			lo=self.lo,
			hi=self.hi,
			unit=self.unit,
			status=self.status,
			model_id=self.model_id,
			source="exact",
			reason=self.reason,
			level=self.level,
		)


def _abstain(unit: str, why: str, **detail) -> TotalEstimate:
	return TotalEstimate(None, None, None, unit, "abstain", reason=why, detail=detail)


def _clean(values: Iterable[float | None]) -> tuple[np.ndarray, int]:
	arr, dropped = [], 0
	for v in values:
		if v is None or not math.isfinite(v) or v < 0:
			dropped += 1
		else:
			arr.append(float(v))
	return np.asarray(arr, dtype=float), dropped


def runaway_threshold(values: np.ndarray, factor: float = RUNAWAY_FACTOR) -> float:
	"""8x the median; when most tasks cost nothing (cached), 8x the median of the ones that cost something."""
	med = float(np.median(values)) if values.size else 0.0
	if med > 0:
		return factor * med
	pos = values[values > 0]
	return factor * float(np.median(pos)) if pos.size else math.inf


def estimate_total(
	values: Iterable[float | None],
	population: int | None,
	unit: str = "usd",
	*,
	level: float = LEVEL,
	runaway_factor: float = RUNAWAY_FACTOR,
	unseen_runaway_mult: float = UNSEEN_RUNAWAY_MULT,
	runaway_bound: float = RUNAWAY_BOUND,
	n_boot: int = N_BOOT,
	min_sample: int = MIN_SAMPLE,
	seed: int = 0,
) -> TotalEstimate:
	"""The full run's total of a per-task quantity (cost, tokens), from the sampled tasks."""
	s, dropped = _clean(values)
	n = int(s.size)
	if not population or population <= 0:
		return _abstain(unit, "the full run's task count isn't known", n=n)
	if n < min_sample:
		why = f"only {n} task{'s' if n != 1 else ''} measured; run at least {min_sample} for a total"
		return _abstain(unit, why, n=n, dropped=dropped)
	with np.errstate(over="ignore"):
		too_big = not math.isfinite(float(s.sum()) * population * UNSEEN_RUNAWAY_MULT)
	if too_big:
		return _abstain(unit, "the task values are too large to add up", n=n)
	if n > population:
		why = f"{n} tasks measured but the run is said to have only {population}; check the task count"
		return _abstain(unit, why, n=n, population=population)
	if n == population:
		total = float(s.sum())  # every task ran: the total is measured
		if not math.isfinite(total):
			return _abstain(unit, "the task values are too large to add up", n=n)
		return TotalEstimate(total, total, total, unit, "measured", detail={"n": n, "population": population})

	med = float(np.median(s))
	thr = runaway_threshold(s, runaway_factor)
	mask = s > thr
	x = int(mask.sum())
	pos = s[s > 0]
	base = med if med > 0 else (float(np.median(pos)) if pos.size else 0.0)
	run_cost = float(s[mask].mean()) if x else unseen_runaway_mult * base

	rng = np.random.default_rng(seed)
	per = max(1, BOOT_CHUNK_CELLS // n)  # bounded memory for any sample size
	boots = (
		np.concatenate(
			[
				s[rng.integers(0, n, size=(min(per, n_boot - i), n))].mean(axis=1)
				for i in range(0, n_boot, per)
			]
		)
		* population
	)
	a = (1 - level) / 2
	lo, hi = (float(q) for q in np.quantile(boots, [a, 1 - a]))
	point = float(s.mean()) * population
	upper_rate = cp_upper(x, n, runaway_bound)
	tail = population * max(0.0, upper_rate - x / n) * run_cost
	hi = max(hi, point) + tail

	if x:
		why = (
			f"{x} of {n} sampled tasks ran away (over {runaway_factor:g}x the median); "
			f"the upper end allows for up to {upper_rate:.0%} of the run doing so"
		)
	else:
		why = (
			f"no sampled task ran away, but up to {upper_rate:.0%} of the run could; "
			f"the upper end prices them at {unseen_runaway_mult:g}x the median task"
		)
	if dropped:
		why += f"; {dropped} task{'s' if dropped != 1 else ''} without a usable value left out"
	if not all(math.isfinite(v) for v in (point, lo, hi)):
		return _abstain(unit, "the task values are too large to add up", n=n)
	return TotalEstimate(
		point,
		min(lo, point),
		hi,
		unit,
		"predicted",
		reason=why,
		level=level,
		detail={
			"n": n,
			"population": population,
			"dropped": dropped,
			"runaways": x,
			"runaway_threshold": thr,
			"runaway_cost": run_cost,
			"runaway_rate_upper": upper_rate,
			"tail_added": tail,
		},
	)


def runaway_rate(
	values: Iterable[float | None],
	level: float = LEVEL,
	min_sample: int = MIN_SAMPLE,
	runaway_factor: float = RUNAWAY_FACTOR,
) -> TotalEstimate:
	"""Share of the run's tasks that run away: observed rate with a two-sided Clopper-Pearson interval."""
	s, dropped = _clean(values)
	n = int(s.size)
	if n < min_sample:
		return _abstain("fraction", f"only {n} tasks measured; run at least {min_sample}", n=n)
	x = int((s > runaway_threshold(s, runaway_factor)).sum())
	side = (1 + level) / 2
	return TotalEstimate(
		x / n,
		cp_lower(x, n, side),
		cp_upper(x, n, side),
		"fraction",
		"predicted",
		reason=None
		if x
		else f"none of {n} sampled tasks ran away; the upper end is what {n} tasks can rule out",
		level=level,
		detail={"n": n, "runaways": x, "dropped": dropped},
	)


# ---------------------------------------------------------------- from A's AgentSample


def _tokens(t) -> int:
	# OpenAI and Anthropic count reasoning inside output tokens, so it isn't added again (A to pin this)
	return int(t.input_tokens + t.output_tokens)


def agent_totals(
	sample,
	price: Callable[[object], float | None] | None = None,
	population: int | None = None,
	**kw,
) -> dict[str, TotalEstimate]:
	"""`AgentSample` -> {"cost.total", "tokens.total", "risk.runaway"}.

	Cost per task is `cost_usd` when the sample carries it, else `price(task)` (tokens times
	the model's rates; D owns the API price table). A task with neither is left out and counted.
	"""
	tasks = list(getattr(sample, "tasks", None) or [])
	pop = population if population is not None else getattr(sample, "population", None)
	costs = [t.cost_usd if t.cost_usd is not None else (price(t) if price else None) for t in tasks]
	tokens = [_tokens(t) for t in tasks]
	out = {
		"cost.total": estimate_total(costs, pop, "usd", **kw),
		"tokens.total": estimate_total(tokens, pop, "tokens", **kw),
		"risk.runaway": runaway_rate(
			costs if any(c is not None for c in costs) else tokens,
			level=kw.get("level", LEVEL),
			min_sample=kw.get("min_sample", MIN_SAMPLE),
			runaway_factor=kw.get("runaway_factor", RUNAWAY_FACTOR),
		),
	}
	return out


# ---------------------------------------------------------------- sample size and the top-up (Chiya, Oct 27)

DEFAULT_SAMPLE = 30  # tasks run first, with the 80% runaway bound (S6: 90.3% coverage)
TOP_UP_TASKS = (
	20  # more tasks the guide offers, with the user's click, when the cost range straddles the budget
)
MAX_SAMPLE = DEFAULT_SAMPLE + TOP_UP_TASKS


@dataclass(frozen=True)
class BudgetCheck:
	verdict: str  # "under" | "over" | "straddles" | "unknown"
	offer_top_up: bool
	top_up_tasks: int
	reason: str


def budget_check(estimate: TotalEstimate, budget_usd: float | None) -> BudgetCheck:
	"""Where the run's cost range sits against the budget, and whether to offer the 20-task top-up.

	The top-up is offered once: only while the sample is still below 50 tasks and the run has
	tasks left to sample. The guide asks the user before running it.
	"""
	if budget_usd is None or budget_usd <= 0:
		return BudgetCheck("unknown", False, 0, "no budget set")
	if estimate.status == "abstain" or estimate.lo is None or estimate.hi is None:
		return BudgetCheck("unknown", False, 0, estimate.reason or "no cost estimate yet")
	if estimate.hi <= budget_usd:
		return BudgetCheck("under", False, 0, f"the whole cost range is under the ${budget_usd:,.2f} budget")
	if estimate.lo > budget_usd:
		return BudgetCheck("over", False, 0, f"the whole cost range is over the ${budget_usd:,.2f} budget")
	n = int(estimate.detail.get("n", 0))
	pop = int(estimate.detail.get("population", 0))
	more = max(0, min(TOP_UP_TASKS, pop - n)) if n < MAX_SAMPLE else 0
	why = f"the cost range straddles the ${budget_usd:,.2f} budget"
	if more:
		why += f"; running {more} more tasks narrows it"
	return BudgetCheck("straddles", more > 0, more, why)


def combine_samples(first, extra):
	"""The first sample plus a top-up, as one `AgentSample` (a task id seen twice counts once)."""
	from preflight.schema.record import AgentSample

	seen: set[str] = set()
	tasks = []
	for t in [*first.tasks, *extra.tasks]:
		if t.task_id not in seen:
			seen.add(t.task_id)
			tasks.append(t)
	pop = first.population if first.population is not None else extra.population
	added = len(tasks) - len(first.tasks)
	selection = f"{first.selection or 'sample'} + top-up of {added}"
	data = {**first.model_dump(), "tasks": [t.model_dump() for t in tasks], "population": pop}
	return AgentSample.model_validate({**data, "selection": selection[:200]})  # validated, unlike model_copy
