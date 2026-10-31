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

from preflight.engine import api
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
	gpu_hours: Estimate
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
	"""Expected share of wall time spent saving or redoing work (Young's first-order model)."""
	return save_s / interval_s + (interval_s + save_s) / (2 * mtti_s) + restart_s / mtti_s


def _save_seconds(record: FlightRecord) -> float | None:
	"""Seconds per checkpoint save, as a probe measured it."""
	for p in record.probes:
		if p.checkpoint and p.checkpoint.seconds_mean:
			return float(p.checkpoint.seconds_mean)
	return None


def _checkpoint_bytes(record: FlightRecord, knobs: dict) -> Estimate:
	"""Bytes per checkpoint: measured by a probe, else the memory model's weights, master copy and optimizer."""
	for p in record.probes:
		if p.checkpoint and p.checkpoint.bytes_mean:
			b = float(p.checkpoint.bytes_mean)
			return _est(b, b, b, "bytes", "measured", "measured by a probe's checkpoint save")
	pred = api.predict(record, knobs).root
	terms = [pred.get(f"memory.{t}") for t in ("weights", "master_weights", "optimizer")]
	terms = [t for t in terms if _ok(t)]
	if not terms:
		return _abstain(
			"bytes", "no probe saved a checkpoint, and the memory model has no weights for this job"
		)
	return _est(
		sum(t.point for t in terms),
		sum(t.lo for t in terms),
		sum(t.hi for t in terms),
		"bytes",
		"predicted",
		"no probe saved a checkpoint: weights, master copy and optimizer state from the memory model",
	)


# ---------------------------------------------------------------- Slurm segments


def segments_needed(
	work_s: float, cap_s: float, interval_s: float, save_s: float, restart_s: float
) -> int | None:
	"""Jobs a run of `work_s` seconds splits into under a `cap_s` time limit. Work after a segment's last
	checkpoint is redone, so each segment banks only whole checkpoint intervals. None: it can't finish at all."""
	usable = cap_s - restart_s
	per_segment = math.floor(usable / (interval_s + save_s)) * interval_s
	if work_s <= cap_s - restart_s:
		return 1
	if per_segment <= 0:
		return None
	return math.ceil(work_s / per_segment)


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
	knob_sets = cells_of(grid)
	if not knob_sets or len(knob_sets) > MAX_CELLS:
		raise ValueError(f"a campaign needs between 1 and {MAX_CELLS} cells, got {len(knob_sets)}")
	cap = cl.wall_time_cap_s or cons.wall_time_cap_s
	mtti = cl.mtti_s or (cons.preemption.mtti_s if cons.preemption.kind == "poisson" else None)
	budget = cl.budget_usd if cl.budget_usd is not None else cons.budget_usd
	quota = cl.disk_quota_bytes or (cons.quota.disk_bytes if cons.quota else None)
	notes: list[str] = []

	# ---- grid pricing: the engine's time and cost for each cell
	declared = {k.id for k in record.knobs}
	labels = [k for k in grid if k not in declared]
	if labels:
		# a seed or a learning rate the script doesn't expose as a knob: the engine sees no difference, so those
		# cells cost what the base run costs; the axis still multiplies the runs
		notes.append(
			f"{', '.join(labels)} {'isn' if len(labels) == 1 else 'aren'}'t knobs the engine models, so each value costs what "
			"the run costs at its other settings"
		)
	cells: list[CellPlan] = []
	for ks in knob_sets:
		engine_ks = {k: v for k, v in ks.items() if k in declared}
		pred = api.predict(record, engine_ks).root
		time = pred.get("time.total") or _abstain("s", "the engine has no run time for this job")
		_, jc = api.run_cost(record, engine_ks, provider=cl.provider, billing=cl.billing, region=cl.region)
		hours = jc.gpu_hours if jc.gpu_hours is not None else None
		usd = jc.total_usd if jc.total_usd is not None else jc.market_equivalent_usd
		reason = "" if _ok(time) else (time.reason or "no run time")
		cells.append(CellPlan(_cell_id(ks), ks, time, hours, usd, reason=reason))
	missing = [c.id for c in cells if not _ok(c.time)]
	if missing:
		first = next(c for c in cells if not _ok(c.time))
		notes.append(
			f"{len(missing)} of {len(cells)} runs have no run time from the engine ({first.id}: {first.reason})"
		)

	# ---- checkpoints: what the runs will do (your interval, else the script's), and Young/Daly's advice
	save_s = _save_seconds(record)
	base_ks = {k: v for k, v in knob_sets[0].items() if k in declared}
	tau, tau_why = cl.checkpoint_interval_s, "your interval"
	if tau is None:
		every, step = (
			api.checkpoint_every(record, base_ks),
			api.predict(record, base_ks).root.get("time.step"),
		)
		if every and _ok(step):
			tau, tau_why = float(every) * step.point, f"the script saves every {every:g} steps"
	if mtti and save_s:
		best = daly_interval(save_s, mtti)
		why = (
			f"Young/Daly: save {save_s:.0f} s, one interruption every {mtti / 3600:.1f} h per run, so checkpoint "
			f"every {best / 60:.0f} min ({waste_fraction(best, save_s, mtti, cl.restart_s):.1%} of wall time lost)"
		)
		if tau:
			why += f"; {tau_why} ({tau / 60:.0f} min) loses {waste_fraction(tau, save_s, mtti, cl.restart_s):.1%}"
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
	if tau and not save_s:
		notes.append("no probe timed a checkpoint save, so saves are counted as free")
	save = save_s or 0.0

	# ---- Slurm segments
	seg_cells = []
	for c in cells:
		segs = None
		if _ok(c.time) and cap:
			segs = segments_needed(c.time.hi, cap, tau or c.time.hi, save, cl.restart_s)
			if segs is None:
				c = CellPlan(
					**{
						**c.__dict__,
						"reason": (c.reason + "; " if c.reason else "")
						+ "can't finish under the job time limit without checkpoints",
					}
				)
		elif _ok(c.time):
			segs = 1
		seg_cells.append(CellPlan(**{**c.__dict__, "segments": segs}))
	cells = seg_cells
	if cap and any(c.segments is None and _ok(c.time) for c in cells):
		notes.append(f"some runs outlast the {cap / 3600:.0f} h job limit and save no checkpoint in time")

	# ---- totals, with every run's error shared (comonotone): the widest honest range
	def total(attr: str, unit: str) -> Estimate:
		es = [getattr(c, attr) for c in cells]
		if not es or not all(_ok(e) for e in es):
			return _abstain(unit, f"{sum(1 for e in es if not _ok(e))} of {len(es)} runs have no {unit}")
		return _est(
			sum(e.point for e in es),
			sum(e.lo for e in es),
			sum(e.hi for e in es),
			unit,
			"predicted",
			f"{len(es)} runs; their errors added as if they move together",
		)

	usd_total = total("usd", "usd")
	hours_total = total("gpu_hours", "gpu_hours")

	# ---- budget: cut the dearest runs until the high end fits
	cut: list[str] = []
	if budget is not None and _ok(usd_total) and usd_total.hi > budget:
		left = usd_total.hi
		for c in sorted(cells, key=lambda c: -(c.usd.hi if _ok(c.usd) else 0.0)):
			if left <= budget:
				break
			cut.append(c.id)
			left -= c.usd.hi
		cells = [CellPlan(**{**c.__dict__, "cut": c.id in cut}) for c in cells]
		notes.append(
			f"the grid's high end (${usd_total.hi:,.2f}) is over the ${budget:,.2f} budget: {len(cut)} dearest runs cut"
		)

	# ---- lanes: longest runs first, each to the lane that frees up soonest (LPT)
	lane_slots = dict(cl.lanes) if cl.lanes else {"cluster": max(1, cl.slots)}
	load = {name: [0.0] * max(1, n) for name, n in lane_slots.items()}
	# runs that are cut don't run: they get no lane and don't count towards the end date
	kept = [i for i, c in enumerate(cells) if not c.cut]
	order = sorted(kept, key=lambda i: -(cells[i].time.point or 0.0))
	lane_of: dict[int, tuple[str, int]] = {}
	for i in order:
		name, slot = min(
			((n, s) for n, ss in load.items() for s in range(len(ss))), key=lambda ns: load[ns[0]][ns[1]]
		)
		lane_of[i] = (name, slot)
		load[name][slot] += cells[i].time.point or 0.0
	cells = [
		CellPlan(**{**c.__dict__, "lane": lane_of[i][0] if i in lane_of else None})
		for i, c in enumerate(cells)
	]

	# ---- quota: checkpoints kept by every run going at once
	# the largest checkpoint in the grid: a bigger LoRA rank or another precision changes it
	sizes = [
		_checkpoint_bytes(record, ks)
		for ks in {
			_cell_id(e): e for e in ({k: v for k, v in ks.items() if k in declared} for ks in knob_sets)
		}.values()
	]
	ck = max((c for c in sizes if _ok(c)), key=lambda c: c.hi, default=sizes[0])
	concurrent = min(len(kept), sum(lane_slots.values()))
	if _ok(ck):
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

	# ---- Monte Carlo: model error shared by every run, interruptions and resubmit waits per run
	rng = np.random.default_rng(seed)
	z = rng.standard_normal(draws)
	if not kept:
		finish_est = _abstain("s", "every run is cut, so nothing is scheduled")
		completion, p_dead = {}, None
	elif all(_ok(cells[i].time) for i in kept):
		dur = np.zeros((len(cells), draws))
		for i in kept:
			dur[i] = _scale(z, cells[i].time)  # cells x draws; cut rows stay zero and are never scheduled
		if mtti:
			# Poisson interruptions; each redoes the work since the last checkpoint (on average half an
			# interval, Young/Daly) or, with no checkpoints, everything so far, plus a restart
			hits = rng.poisson(dur / mtti)
			redo = tau if tau else dur
			dur = dur + hits * (rng.uniform(0, 1, dur.shape) * redo + cl.restart_s)
		extra_segs = np.array([[max(0, (c.segments or 1) - 1)] for c in cells], dtype=float)
		if cap and extra_segs.any():
			# each resubmit restarts and waits in the queue; the waits of k resubmits add up to a gamma(k)
			waits = (
				rng.gamma(np.maximum(extra_segs, 1e-12) * np.ones(dur.shape), cl.queue_wait_s)
				if cl.queue_wait_s
				else 0.0
			)
			dur = dur + extra_segs * cl.restart_s + np.where(extra_segs > 0, waits, 0.0)
		# list-schedule each draw: runs in LPT order onto the slot the lanes gave them
		finish = np.zeros(draws)
		slot_end = {(n, s): np.zeros(draws) for n, ss in load.items() for s in range(len(ss))}
		for i in order:
			slot_end[lane_of[i]] = slot_end[lane_of[i]] + dur[i]
		for v in slot_end.values():
			finish = np.maximum(finish, v)
		q = np.quantile(finish, [0.05, 0.10, 0.50, 0.90, 0.95])
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
		p_dead = None
		if cons.deadline:
			left = (cons.deadline - start).total_seconds()
			p_dead = float(np.mean(finish <= left))
	else:
		finish_est = _abstain(
			"s",
			f"{sum(1 for i in kept if not _ok(cells[i].time))} runs have no run time, so the campaign has no end date",
		)
		completion, p_dead = {}, None

	# ---- ASHA
	asha_plan = None
	if asha is not None:
		steps, why_steps = api.total_steps(record, {k: v for k, v in knob_sets[0].items() if k in declared})
		max_steps = asha.max_steps or (int(steps) if steps else None)
		if not max_steps or not _ok(hours_total):
			asha_plan = None
			notes.append(f"no ASHA plan: {why_steps or 'the grid has no GPU-hours'}")
		else:
			rungs = asha_rungs(len(cells), asha.min_steps, asha.eta, max_steps)
			share = asha_steps(rungs) / (len(cells) * max_steps)
			hrs = _est(
				hours_total.point * share,
				hours_total.lo * share,
				hours_total.hi * share,
				"gpu_hours",
				"predicted",
				f"{share:.0%} of the full grid's steps",
			)
			saved = _est(
				hours_total.point - hrs.point,
				hours_total.lo - hrs.lo,
				hours_total.hi - hrs.hi,
				"gpu_hours",
				"predicted",
				"against running every cell in full",
			)
			asha_plan = AshaPlan(
				tuple(rungs),
				hrs,
				saved,
				f"eta {asha.eta}, from {asha.min_steps} steps; assumes every step costs the same in every run",
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
	)
