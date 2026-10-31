"""Campaign planner: a grid of runs priced, segmented, checked and dated from the engine's own intervals.

Every number starts from engine.api: each cell's run time and cost come from `predict` and `run_cost`, with
their 90% intervals. The planner adds no accuracy of its own, so it is exactly as accurate as the engine; its card
says so. Pure Python and numpy, no I/O, so it runs in Pyodide like the engine.

Pieces (main plan, Feature set: Campaign mode):
- grid pricing: each cell's GPU-hours and dollars, and the grid's total with an interval;
- Young/Daly checkpoint interval from the measured save time and the cluster's mean time to interrupt;
- Slurm wall-time segments: how many resubmits each run needs under the job time limit;
- quota check: checkpoints kept by the runs that are going at once, against the disk quota;
- ASHA pruning plan: rungs, survivors and the GPU-hours it saves against running every cell in full;
- Monte Carlo completion dates: p10, p50 and p90 of when the last run finishes, interruptions and resubmits
  included, run by run in the team's lanes.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from preflight.engine import api, knobs
from preflight.schema.prediction import Estimate
from preflight.schema.record import Campaign, CampaignCell, FlightRecord, ModelCard

MODEL_ID = "campaign-mc@0.1.0"
Z90 = 1.6448536269514722  # the 90% two-sided normal quantile; the engine's bands are 90%
CARD_NOTE = (
	"Every run's time and cost come from the engine's own 90% intervals, so this plan is exactly as accurate as "
	"the engine: it adds scheduling, interruptions and resubmits on top, never accuracy. Runs are assumed to err "
	"together (the same script and the same model), which makes the grid's range as wide as it can honestly be."
)
RESTART_S = 60.0  # guess: a requeue plus reload takes at least a minute; the engine uses the same floor
MAX_CELLS = 512  # a grid bigger than this is a sweep for a scheduler, not something to draw on a calendar


@dataclass(frozen=True)
class Cluster:
	"""Where the campaign runs. Unset fields come from the record's constraints."""

	slots: int = 1  # runs going at once (GPUs or nodes the user may hold)
	lanes: dict[str, int] | None = None  # team lanes: name -> slots; overrides `slots`
	wall_time_cap_s: float | None = None  # the scheduler's limit per job, e.g. Slurm's --time
	mtti_s: float | None = None  # mean time to interrupt one run
	restart_s: float = RESTART_S
	queue_wait_s: float = 0.0  # mean wait before each resubmitted segment starts
	checkpoint_interval_s: float | None = None  # what the runs will use; default: the Young/Daly interval
	disk_quota_bytes: int | None = None
	keep_checkpoints: int = 2  # checkpoints each run keeps on disk
	budget_usd: float | None = None
	provider: str | None = None
	billing: str | None = None
	region: str | None = None


@dataclass(frozen=True)
class Asha:
	"""Successive halving: every cell runs `min_steps`, the best 1/eta go on to eta times as many, and so on."""

	min_steps: int
	eta: int = 3
	max_steps: int | None = None  # default: the run's full length


@dataclass(frozen=True)
class CellPlan:
	id: str
	knobs: dict
	time: Estimate  # the engine's time.total
	gpu_hours: Estimate | None
	usd: Estimate | None
	segments: int | None = None
	lane: str | None = None
	cut: bool = False
	reason: str = ""


@dataclass(frozen=True)
class Rung:
	steps: int
	runs: int


@dataclass(frozen=True)
class AshaPlan:
	rungs: tuple[Rung, ...]
	gpu_hours: Estimate
	saved_gpu_hours: Estimate
	reason: str


@dataclass(frozen=True)
class QuotaCheck:
	needed_bytes: Estimate
	quota_bytes: int | None
	fits: bool | None  # None when there is no quota to check against
	reason: str


@dataclass(frozen=True)
class CampaignPlan:
	cells: tuple[CellPlan, ...]
	gpu_hours: Estimate  # the runs that go ahead (after any budget cut), redone work included
	usd: Estimate
	finish_s: Estimate  # seconds from the start until the last run ends
	completion: dict[str, datetime]  # p10, p50, p90
	p_by_deadline: float | None
	checkpoint_interval: Estimate
	quota: QuotaCheck
	asha: AshaPlan | None
	lanes: tuple[str, ...]
	cut_list: tuple[str, ...]
	notes: tuple[str, ...] = ()
	grid_gpu_hours: Estimate | None = None  # every cell, before any cut
	grid_usd: Estimate | None = None
	card: ModelCard = field(
		default_factory=lambda: ModelCard(
			id=MODEL_ID,
			kind="simulation",
			version=MODEL_ID.split("@")[1],
			training_data="none: built from the engine's predictions for this record",
			notes=CARD_NOTE,
		)
	)

	def to_campaign(self) -> Campaign:
		"""The record's Campaign block (contract), from this plan."""
		return Campaign(
			cells=[
				CampaignCell(
					id=c.id,
					knobs=c.knobs,
					gpu_hours=c.gpu_hours.point if c.gpu_hours else None,
					usd=c.usd.point if c.usd else None,
					wall_time_s=c.time.point,
					segments=c.segments,
					lane=c.lane,
					cut=c.cut,
				)
				for c in self.cells
			],
			lanes=list(self.lanes),
			cut_list=list(self.cut_list),
			completion_quantiles=dict(self.completion),
			checkpoint_interval_s=self.checkpoint_interval.point,
		)


# ---------------------------------------------------------------- small helpers


def _est(point, lo, hi, unit, status="predicted", reason=None) -> Estimate:
	return Estimate(
		point=point, lo=lo, hi=hi, unit=unit, status=status, model_id=MODEL_ID, source="exact", reason=reason
	)


def _abstain(unit: str, reason: str) -> Estimate:
	return Estimate(
		point=None,
		lo=None,
		hi=None,
		unit=unit,
		status="abstain",
		model_id=MODEL_ID,
		source="exact",
		reason=reason,
	)


def _ok(e: Estimate | None) -> bool:
	return e is not None and e.status != "abstain" and e.point is not None


def _sigmas(e: Estimate) -> tuple[float, float]:
	"""Log-space sigmas below and above the point, from the engine's 90% band."""
	lo = max(e.lo or e.point, 1e-12)
	hi = max(e.hi or e.point, 1e-12)
	return math.log(e.point / lo) / Z90, math.log(hi / e.point) / Z90


def _scale(z: np.ndarray, e: Estimate) -> np.ndarray:
	"""The estimate at standard-normal draws z: log-normal on each side of the point, as the band is."""
	down, up = _sigmas(e)
	return e.point * np.exp(np.where(z < 0, z * down, z * up))


def cells_of(grid: dict[str, list]) -> list[dict]:
	"""Every combination of the grid's values, in the order given."""
	keys = list(grid)
	return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))]


def _cell_id(knobs: dict) -> str:
	return ",".join(f"{k}={v}" for k, v in knobs.items()) or "base"


# ---------------------------------------------------------------- Young/Daly


def daly_interval(save_s: float, mtti_s: float) -> float:
	"""Daly's higher-order optimum checkpoint interval (Daly 2006, eq. 37); Young's sqrt(2 d M) is its first term.
	A save as long as twice the interruption interval leaves nothing to gain: checkpoint every M."""
	if save_s >= 2 * mtti_s:
		return mtti_s
	r = math.sqrt(save_s / (2 * mtti_s))
	return math.sqrt(2 * save_s * mtti_s) * (1 + r / 3 + r * r / 9) - save_s


def waste_fraction(interval_s: float, save_s: float, mtti_s: float, restart_s: float) -> float:
	"""Expected share of wall time spent saving or redoing work (Young's first-order model); minimised at
	Young's interval sqrt(2 d M)."""
	return save_s / interval_s + (interval_s + save_s) / (2 * mtti_s) + restart_s / mtti_s


def _save_seconds(record: FlightRecord) -> float | None:
	"""Seconds per checkpoint save, as a probe measured it."""
	for p in record.probes:
		if p.checkpoint and p.checkpoint.seconds_mean:
			return float(p.checkpoint.seconds_mean)
	return None


def _checkpoint_bytes(record: FlightRecord, knob_values: dict) -> Estimate:
	"""Bytes per checkpoint: measured by a probe, else from the memory model. A LoRA or QLoRA checkpoint holds the
	adapters and their optimizer state, not the frozen base (PEFT saves only what trains); a full fine-tune holds
	the weights, the master copy and the optimizer state."""
	for p in record.probes:
		if p.checkpoint and p.checkpoint.bytes_mean:
			b = float(p.checkpoint.bytes_mean)
			return _est(b, b, b, "bytes", "measured", "measured by a probe's checkpoint save")
	pred = api.predict(record, knob_values).root
	try:
		_, setup, _ = knobs.setup_from(record.job.script_map, api._values(record, knob_values))
		adapters = setup.method in ("lora", "qlora")
	except (ValueError, AttributeError):
		adapters = False
	# the adapters are as big as their gradients, which the memory model already counts
	names = (
		("gradients", "master_weights", "optimizer")
		if adapters
		else ("weights", "master_weights", "optimizer")
	)
	terms = [pred.get(f"memory.{t}") for t in names]
	terms = [t for t in terms if _ok(t)]
	if not terms:
		return _abstain(
			"bytes", "no probe saved a checkpoint, and the memory model has no weights for this job"
		)
	what = "adapters and their optimizer state" if adapters else "weights, master copy and optimizer state"
	return _est(
		sum(t.point for t in terms),
		sum(t.lo for t in terms),
		sum(t.hi for t in terms),
		"bytes",
		"predicted",
		f"no probe saved a checkpoint: {what} from the memory model",
	)


# ---------------------------------------------------------------- Slurm segments


def banked_per_segment(cap_s: float, interval_s: float, save_s: float, restart_s: float) -> int:
	"""Whole checkpoint intervals one job can finish and save before the time limit kills it."""
	return max(0, math.floor((cap_s - restart_s) / (interval_s + save_s)))


def segments_needed(
	work_s: float, cap_s: float, interval_s: float, save_s: float, restart_s: float
) -> int | None:
	"""Jobs a run of `work_s` wall seconds (saves included) splits into under a `cap_s` time limit. Work after a
	segment's last checkpoint is redone, so each job banks only whole intervals and their saves. None: it can't
	finish at all."""
	if work_s <= cap_s - restart_s:
		return 1
	k = banked_per_segment(cap_s, interval_s, save_s, restart_s)
	if k == 0:
		return None
	return math.ceil(work_s / (k * (interval_s + save_s)))


# ---------------------------------------------------------------- interruptions


MAX_EXPECTED_FAILURES = 1e5  # past this the run effectively never finishes: interruptions outpace checkpoints


def redone_seconds(
	rng: np.random.Generator,
	work_s: np.ndarray,
	interval_s: float | None,
	mtti_s: float,
	restart_s: float,
) -> np.ndarray:
	"""Wall seconds lost to Poisson interruptions, per draw, when each one sends the run back to its last checkpoint.

	The run is n pieces of length L (the checkpoint interval, or the whole run without checkpoints). A piece
	succeeds with probability exp(-L/M), so the failures before n successes are negative binomial, and each failure
	loses a truncated-exponential stretch of time plus the restart. Interruptions during redone work are counted,
	so a long run without checkpoints grows like (M + R)(e^(T/M) - 1), not linearly. inf: it won't finish."""
	work = np.asarray(work_s, dtype=float)
	L = np.minimum(interval_s, work) if interval_s else work
	n = np.maximum(1, np.ceil(work / np.maximum(L, 1e-9)))
	x = L / mtti_s
	expected = n * np.expm1(x)
	out = np.full(work.shape, np.inf)
	ok = expected < MAX_EXPECTED_FAILURES
	if not ok.any():
		return out
	p = np.exp(-x[ok])
	fails = rng.negative_binomial(n[ok], p)
	# a failed piece lasts t ~ Exp(M) given t < L: mean M - L/(e^x - 1), second moment 2M^2 - (L^2 + 2ML)/(e^x - 1)
	em1 = np.expm1(x[ok])
	Lk, M = L[ok], mtti_s
	mean = M - Lk / em1
	var = np.maximum(2 * M * M - (Lk * Lk + 2 * M * Lk) / em1 - mean * mean, 0.0)
	lost = fails * (mean + restart_s) + np.sqrt(fails * var) * rng.standard_normal(fails.shape)
	out[ok] = np.maximum(lost, 0.0)
	return out


# ---------------------------------------------------------------- ASHA


def asha_rungs(n: int, min_steps: int, eta: int, max_steps: int) -> list[Rung]:
	"""Successive-halving rungs (Li et al., ASHA): rung k runs n / eta^k configs to min_steps * eta^k steps."""
	if n < 1 or min_steps < 1 or eta < 2 or max_steps < min_steps:
		raise ValueError("ASHA needs n >= 1, min_steps >= 1, eta >= 2 and max_steps >= min_steps")
	rungs, k = [], 0
	while True:
		steps = min(min_steps * eta**k, max_steps)
		runs = max(1, n // eta**k)
		if runs == 1 and steps < max_steps:  # a lone survivor goes straight to the full length
			steps = max_steps
		rungs.append(Rung(steps, runs))
		if steps >= max_steps:  # the last survivors always run the full length
			break
		k += 1
	return rungs


def asha_steps(rungs: list[Rung]) -> int:
	"""Total steps run: each survivor only runs the steps beyond the rung it was promoted from."""
	total, prev = 0, 0
	for r in rungs:
		total += r.runs * (r.steps - prev)
		prev = r.steps
	return total


# ---------------------------------------------------------------- the plan


def _dur(seconds: float) -> str:
	return f"{seconds / 3600:g} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def _grown(e: Estimate | None, f: float) -> Estimate | None:
	"""A cost times f, the expected share of billed time once redone work and restarts are in."""
	if not _ok(e):
		return e
	why = (
		e.reason + "; " if e.reason else ""
	) + f"plus {f - 1:.0%} redone after interruptions and job limits"
	return _est(e.point * f, e.lo * f, e.hi * f, e.unit, e.status, why)


def _replace(c: CellPlan, **kw) -> CellPlan:
	return CellPlan(**{**c.__dict__, **kw})


def _sum(es: list[Estimate | None], unit: str, why: str) -> Estimate:
	"""Totals with every run's error shared (comonotone): lows and highs add, the widest honest range."""
	if not es:
		return _abstain(unit, "no runs")
	if not all(_ok(e) for e in es):
		return _abstain(unit, f"{sum(1 for e in es if not _ok(e))} of {len(es)} runs have no {unit}")
	return _est(
		sum(e.point for e in es), sum(e.lo for e in es), sum(e.hi for e in es), unit, "predicted", why
	)


def plan(
	record: FlightRecord,
	grid: dict[str, list],
	cluster: Cluster | None = None,
	asha: Asha | None = None,
	start: datetime | None = None,
	draws: int = 4000,
	seed: int = 0,
) -> CampaignPlan:
	"""Price, segment, check and date every cell of `grid` (knob -> values) for this record's job."""
	cl = cluster or Cluster()
	cons = record.job.constraints
	start = start or datetime.now(UTC)
	if start.tzinfo is None:
		start = start.replace(tzinfo=UTC)
	knob_sets = cells_of(grid)
	if not knob_sets or len(knob_sets) > MAX_CELLS:
		raise ValueError(f"a campaign needs between 1 and {MAX_CELLS} cells, got {len(knob_sets)}")
	cap = cl.wall_time_cap_s or cons.wall_time_cap_s
	mtti = cl.mtti_s or (cons.preemption.mtti_s if cons.preemption.kind == "poisson" else None)
	budget = cl.budget_usd if cl.budget_usd is not None else cons.budget_usd
	quota = cl.disk_quota_bytes or (cons.quota.disk_bytes if cons.quota else None)
	notes: list[str] = []
	lane_slots = {k: int(v) for k, v in (cl.lanes or {"cluster": max(1, cl.slots)}).items() if int(v) > 0}
	if cl.lanes and len(lane_slots) < len(cl.lanes):
		notes.append(
			f"lanes with no slots get no runs: {', '.join(k for k in cl.lanes if k not in lane_slots)}"
		)
	if not lane_slots:
		raise ValueError("every lane has 0 slots, so nothing can run")

	# ---- grid pricing: the engine's time and cost for each cell
	declared = {k.id for k in record.knobs}
	labels = [k for k in grid if k not in declared]
	if labels:
		# a seed or a learning rate the script doesn't expose as a knob: the engine sees no difference, so those
		# cells cost what the base run costs; the axis still multiplies the runs
		verb = "isn't a knob" if len(labels) == 1 else "aren't knobs"
		notes.append(
			f"{', '.join(labels)} {verb} the engine models, so each value costs what the run costs at its other settings"
		)
	engine_sets = [{k: v for k, v in ks.items() if k in declared} for ks in knob_sets]
	cells: list[CellPlan] = []
	steps_of: list[Estimate | None] = []
	for ks, eks in zip(knob_sets, engine_sets, strict=True):
		pred = api.predict(record, eks).root
		time = pred.get("time.total") or _abstain("s", "the engine has no run time for this job")
		steps_of.append(pred.get("time.step"))
		_, jc = api.run_cost(record, eks, provider=cl.provider, billing=cl.billing, region=cl.region)
		usd = jc.total_usd if jc.total_usd is not None else jc.market_equivalent_usd
		reason = "" if _ok(time) else (time.reason or "no run time")
		cells.append(CellPlan(_cell_id(ks), ks, time, jc.gpu_hours, usd, reason=reason))
	if any(not _ok(c.time) for c in cells):
		first = next(c for c in cells if not _ok(c.time))
		notes.append(
			f"{sum(1 for c in cells if not _ok(c.time))} of {len(cells)} runs have no run time from the engine ({first.id}: {first.reason})"
		)

	# ---- checkpoints: what each run will do (your interval, else the script's), and Young/Daly's advice
	save_s = _save_seconds(record)
	save = save_s or 0.0
	taus: list[float | None] = []
	tau_why = "your interval"
	for eks, st in zip(engine_sets, steps_of, strict=True):
		tau = cl.checkpoint_interval_s
		if tau is None:
			every = api.checkpoint_every(record, eks)
			if every and _ok(st):
				tau, tau_why = float(every) * st.point, f"the script saves every {every:g} steps"
		taus.append(tau)
	tau0 = taus[0]
	if mtti and save_s:
		best = daly_interval(save_s, mtti)
		why = (
			f"Young/Daly: save {save_s:.0f} s, one interruption every {mtti / 3600:.1f} h per run, so checkpoint "
			f"every {best / 60:.0f} min ({waste_fraction(best, save_s, mtti, cl.restart_s):.1%} of wall time lost)"
		)
		if tau0:
			why += f"; {tau_why} ({tau0 / 60:.0f} min) loses {waste_fraction(tau0, save_s, mtti, cl.restart_s):.1%}"
		else:
			why += "; the runs don't checkpoint, so an interruption loses everything so far"
		interval = _est(best, best, best, "s", "predicted", why)
	else:
		missing_why = (
			"set the cluster's mean time between interruptions"
			if not mtti
			else "no probe timed a checkpoint save"
		)
		interval = _abstain("s", f"no Young/Daly interval: {missing_why}")
	if any(taus) and not save_s:
		notes.append("no probe timed a checkpoint save, so saves are counted as free")

	# ---- Slurm segments: from the high end of each run's time, so a slow run still fits its jobs
	for i, c in enumerate(cells):
		if not _ok(c.time):
			continue
		if not cap:
			cells[i] = _replace(c, segments=1)
			continue
		segs = segments_needed(c.time.hi, cap, taus[i] or c.time.hi, save, cl.restart_s)
		why = c.reason
		if segs is None:
			why = (
				why + "; " if why else ""
			) + f"can't finish under the {_dur(cap)} job limit without checkpoints: none lands inside it"
		cells[i] = _replace(c, segments=segs, reason=why)

	# ---- each run's wall time per draw: the engine's band (one shared error), interruptions, job-limit tails
	rng = np.random.default_rng(seed)
	z = rng.standard_normal(draws)
	runnable = [_ok(c.time) and c.segments is not None for c in cells]
	wall = np.zeros((len(cells), draws))
	billed = np.zeros((len(cells), draws))
	never: dict[int, str] = {}
	for i, c in enumerate(cells):
		if not runnable[i]:
			continue
		base = _scale(z, c.time)
		extra = np.zeros(draws)
		if mtti:
			lost = redone_seconds(rng, base, taus[i], mtti, cl.restart_s)
			if not np.isfinite(lost).any():
				never[i] = "interruptions come faster than it can checkpoint, so it never finishes"
				continue
			extra = extra + np.where(np.isfinite(lost), lost, np.nan)
		waits = np.zeros(draws)
		if c.segments and c.segments > 1:
			k = banked_per_segment(cap, taus[i] or c.time.hi, save, cl.restart_s)
			tail = cap - k * ((taus[i] or c.time.hi) + save)  # each killed job's unsaved tail and its restart
			extra = extra + (c.segments - 1) * tail
			if cl.queue_wait_s:
				waits = rng.gamma(c.segments - 1, cl.queue_wait_s, draws)
		billed[i] = base + extra  # GPUs bill redone work and restarts, not time waiting in the queue
		wall[i] = billed[i] + waits
	for i, why in never.items():
		cells[i] = _replace(cells[i], reason=(cells[i].reason + "; " if cells[i].reason else "") + why)
		runnable[i] = False

	# ---- money and GPU-hours with the redone work in: each run's engine cost times its expected billed share
	for i, c in enumerate(cells):
		if not runnable[i] or (not mtti and not (c.segments and c.segments > 1)):
			continue
		f = float(np.nanmean(billed[i]) / np.nanmean(_scale(z, c.time)))
		if f > 1.0001:
			cells[i] = _replace(c, usd=_grown(c.usd, f), gpu_hours=_grown(c.gpu_hours, f))

	grid_usd = _sum(
		[c.usd for c in cells], "usd", f"all {len(cells)} runs; their errors added as if they move together"
	)
	grid_hours = _sum(
		[c.gpu_hours for c in cells],
		"gpu_hours",
		f"all {len(cells)} runs; their errors added as if they move together",
	)

	# ---- budget: cut the dearest runs until the high end of what's left fits
	cut: list[str] = []
	if budget is not None and _ok(grid_usd) and grid_usd.hi > budget:
		left = grid_usd.hi
		for c in sorted(cells, key=lambda c: -(c.usd.hi if _ok(c.usd) else 0.0)):
			if left <= budget:
				break
			cut.append(c.id)
			left -= c.usd.hi if _ok(c.usd) else 0.0
		cells = [_replace(c, cut=c.id in cut) for c in cells]
		notes.append(
			f"the grid's high end (${grid_usd.hi:,.2f}) is over the ${budget:,.2f} budget: {len(cut)} dearest runs cut"
		)
	kept = [i for i, c in enumerate(cells) if not c.cut]
	kept_why = (
		f"{len(kept)} runs"
		+ (f" after cutting {len(cut)}" if cut else "")
		+ "; their errors added as if they move together"
	)
	usd_total = (
		_sum([cells[i].usd for i in kept], "usd", kept_why) if kept else _abstain("usd", "every run is cut")
	)
	hours_total = (
		_sum([cells[i].gpu_hours for i in kept], "gpu_hours", kept_why)
		if kept
		else _abstain("gpu_hours", "every run is cut")
	)

	# ---- lanes: the longest runs first, each to the slot that frees up soonest (LPT, on the point estimates)
	load = {name: [0.0] * n for name, n in lane_slots.items()}
	order = sorted(kept, key=lambda i: -(cells[i].time.point or 0.0))
	lane_of: dict[int, tuple[str, int]] = {}
	for i in order:
		name, slot = min(
			((n, k) for n, ss in load.items() for k in range(len(ss))), key=lambda nk: load[nk[0]][nk[1]]
		)
		lane_of[i] = (name, slot)
		load[name][slot] += cells[i].time.point or 0.0
	cells = [_replace(c, lane=lane_of[i][0] if i in lane_of else None) for i, c in enumerate(cells)]

	# ---- the end date: each slot runs its runs back to back; the campaign ends with its last slot
	completion: dict[str, datetime] = {}
	p_dead = None
	if not kept:
		finish_est = _abstain("s", "every run is cut, so nothing is scheduled")
	elif not all(runnable[i] for i in kept):
		bad = next(cells[i] for i in kept if not runnable[i])
		finish_est = _abstain(
			"s", f"{sum(1 for i in kept if not runnable[i])} runs can't be dated ({bad.id}: {bad.reason})"
		)
	else:
		slot_end = {nk: np.zeros(draws) for nk in ((n, k) for n, ss in load.items() for k in range(len(ss)))}
		for i in order:
			slot_end[lane_of[i]] = slot_end[lane_of[i]] + wall[i]
		finish = np.max(np.stack(list(slot_end.values())), axis=0)
		finish = np.where(np.isfinite(finish), finish, np.nan)
		q = np.nanquantile(finish, [0.05, 0.10, 0.50, 0.90, 0.95])
		finish_est = _est(
			float(q[2]),
			float(q[0]),
			float(q[4]),
			"s",
			"predicted",
			f"the last of {len(kept)} runs ends; Monte Carlo over the engine's intervals",
		)
		completion = {
			k: start + timedelta(seconds=float(v))
			for k, v in zip(("p10", "p50", "p90"), (q[1], q[2], q[3]), strict=True)
		}
		if cons.deadline:
			deadline = cons.deadline if cons.deadline.tzinfo else cons.deadline.replace(tzinfo=UTC)
			if cons.deadline.tzinfo is None:
				notes.append("the deadline has no time zone, so it is read as UTC")
			p_dead = float(np.nanmean(finish <= (deadline - start).total_seconds()))

	# ---- quota: the largest checkpoint in the grid, kept by every run going at once
	sizes = [_checkpoint_bytes(record, e) for e in {_cell_id(e): e for e in engine_sets}.values()]
	ck = max((c for c in sizes if _ok(c)), key=lambda c: c.hi, default=sizes[0])
	concurrent = min(len(kept), sum(lane_slots.values()))
	if _ok(ck) and concurrent:
		k = concurrent * max(1, cl.keep_checkpoints)
		needed = _est(
			ck.point * k,
			ck.lo * k,
			ck.hi * k,
			"bytes",
			ck.status,
			f"{concurrent} runs at once x {cl.keep_checkpoints} checkpoints each; {ck.reason}",
		)
		if quota:
			fits = needed.hi <= quota
			why = (
				"fits the disk quota"
				if fits
				else ("may not fit the disk quota" if needed.lo <= quota else "won't fit the disk quota")
			)
		else:
			fits, why = None, "no disk quota set"
		qc = QuotaCheck(needed, quota, fits, why)
	else:
		qc = QuotaCheck(ck, quota, None, ck.reason or "checkpoint size unknown")

	# ---- ASHA: an alternative to cutting, over the whole grid
	asha_plan = None
	if asha is not None:
		steps, why_steps = api.total_steps(record, engine_sets[0])
		run_steps = int(steps) if steps else None
		top = min(asha.max_steps or run_steps or 0, run_steps or asha.max_steps or 0)
		if not run_steps or not _ok(grid_hours):
			notes.append(f"no ASHA plan: {why_steps or 'the grid has no GPU-hours'}")
		elif asha.min_steps > top:
			notes.append(
				f"no ASHA plan: its first rung ({asha.min_steps} steps) is longer than the run ({top} steps)"
			)
		else:
			rungs = asha_rungs(len(cells), asha.min_steps, asha.eta, top)
			share = asha_steps(rungs) / (len(cells) * run_steps)
			hrs = _est(
				grid_hours.point * share,
				grid_hours.lo * share,
				grid_hours.hi * share,
				"gpu_hours",
				"predicted",
				f"{share:.0%} of the full grid's steps",
			)
			saved = _est(
				grid_hours.point - hrs.point,
				grid_hours.lo - hrs.lo,
				grid_hours.hi - hrs.hi,
				"gpu_hours",
				"predicted",
				f"against running all {len(cells)} cells in full",
			)
			asha_plan = AshaPlan(
				tuple(rungs),
				hrs,
				saved,
				f"all {len(cells)} cells, eta {asha.eta}, from {asha.min_steps} steps; an alternative to cutting; assumes every step costs the same in every run",
			)

	return CampaignPlan(
		cells=tuple(cells),
		gpu_hours=hours_total,
		usd=usd_total,
		finish_s=finish_est,
		completion=completion,
		p_by_deadline=p_dead,
		checkpoint_interval=interval,
		quota=qc,
		asha=asha_plan,
		lanes=tuple(lane_slots),
		cut_list=tuple(cut),
		notes=tuple(notes),
		grid_gpu_hours=grid_hours,
		grid_usd=grid_usd,
	)
