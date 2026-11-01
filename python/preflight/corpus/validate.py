"""The validation table: every target in the plan's "Validation targets", scored on full runs (Chiya, Oct 31).

One manifest lists the full runs (zoo and unseen scripts): each run's record (its probes, so the engine can predict)
and its ground truth (A's `ground_truth_<script>.json`, or the fields below). The engine predicts every run at the
full run's own settings, and nothing is fitted here: the scorer reads, predicts and counts. Unseen scripts are
never tuned on; the scorer refuses a manifest that marks an unseen run as used for tuning.

Every row reports counts, a Wilson 95% interval on the proportion and hit or miss. A row with nothing to score says
"not measured" and why, instead of a number.

Manifest (JSON):
	{"runs": [{
		"script": "finetune_lora_hf", "set": "zoo" | "unseen", "job_type": "finetune",
		"record": "path/to/record.json",            # the probe run's FlightRecord
		"ground_truth": "path/to/ground_truth.json", # A's full-run result (bare.ended, peak, step walls)
		"values": {...},                             # the full run's knob values, if not the record's own
		"planted": ["eval.no_grad", ...],            # zoo: finding codes the run should raise
		"harnessed_without_edits": true,             # unseen: the harness ran it unchanged
		"rollout_s": 22.4,                           # RL: a full run's mean rollout time per step
		"tokens_per_s": 1830.0,                      # serving: a full run's measured throughput
		"agent_total_usd": 6.80,                     # agent: the full benchmark run's cost
		"preflight_usd": 0.42, "preflight_s": 210,  # what this Preflight run itself cost and took
		"used_for_tuning": false,
		"calibrates": ["rollout"]                    # rows it calibrated, unscored; "intervals": out of coverage; "*": every row
	}],
	"knob_ms": [41, 55, ...],                        # browser timings of knob moves (C)
	"corpus": ["path/to/kernels/store", ...]         # for the cross-GPU row (leave-one-GPU-out)
	}
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from preflight.engine import conformal

# cited: Chiya, Nov 1: a "within X%" target is a hit when at least 90% of runs are within it, and the hit is
# "confident" only when the Wilson 95% lower bound is at least 90% ("hit, few runs" otherwise)
WITHIN_SHARE = 0.9
CONFIDENT = 0.9
# cited: recorder/bare.py, a bare run that ran to its end ("finished") or its step budget ("steps"); review, Nov 1
FINISHED = ("finished", "steps")
WARMUP_STEPS = 3  # guess: the bare run's first steps compile and warm caches; the steady step comes after
OOM_ALARM = 0.5  # decision: P(OOM) at or above this is an alarm; below it, a run that ran out is a miss
COVERAGE_TARGET = 0.88  # cited: plan, "90% intervals cover at least 88%"
KNOB_P95_MS = 400.0  # cited: plan
PREFLIGHT_USD, PREFLIGHT_S = 1.0, 300.0  # cited: plan, "under $1 and under 5 minutes"
HARNESS_TARGET = (8, 9)  # cited: plan, "8 of 9 without hand edits"

# Corpus rows measured before Nov 2 (each with its run), so the table carries them without rerunning the corpus.
# measured: scripts/fit_cost_gnn.py on b-corpus-export-3 and b-corpus-export-hf-2 (Oct 31), paired bootstrap over
# architectures. The GNN mix's gain over the GBDT alone: +4.8 points within 20% (95% interval -1.4 to +11.6).
GNN_VS_GBDT = {"gain": 0.048, "lo": -0.014, "hi": 0.116}


@dataclass
class Row:
	metric: str
	target: str
	measured_on: str
	hits: int | None = None
	n: int | None = None
	hit: bool | None = None  # None: not measured
	note: str = ""
	misses: list[str] = field(default_factory=list)

	def cells(self) -> list[str]:
		if self.n is None or self.hit is None:
			return [self.metric, self.target, self.measured_on, "not measured", "", "", self.note]
		if (
			self.hits is None
		):  # a verdict with its own interval in the note (a classifier's AUROC, a paired gain)
			runs = f"{self.n} runs" if self.n else ""
			return [
				self.metric,
				self.target,
				self.measured_on,
				runs,
				"",
				"hit" if self.hit else "miss",
				self.note,
			]
		lo, hi = conformal.wilson(self.hits, self.n)
		counts = f"{self.hits} of {self.n}" if self.n else "0"
		return [
			self.metric,
			self.target,
			self.measured_on,
			counts,
			f"{math.floor(lo * 100)}% to {math.ceil(hi * 100)}%" if self.n else "",  # never rounds up to 90%
			verdict(self.hit, lo),
			self.note,
		]


def verdict(hit: bool, wilson_lo: float) -> str:
	"""A miss, a hit whose Wilson lower end reaches CONFIDENT, or a hit on too few runs to say so."""
	if not hit:
		return "miss"
	return "hit, confident" if wilson_lo >= CONFIDENT else "hit, few runs"


# ---------------------------------------------------------------- reading


def _load(path: str | Path | None) -> dict | None:
	if not path:
		return None
	return json.loads(Path(path).read_text())


def _record(path: str | Path):
	from preflight.schema import load_record

	return load_record(Path(path).read_text())


def steady_step_s(gt: dict) -> float | None:
	"""Median wall time of the bare run's steps after warm-up."""
	walls = [w for w in ((gt.get("bare") or {}).get("walls_s") or [])[WARMUP_STEPS:] if w]
	return statistics.median(walls) if walls else None


def ran_out(gt: dict) -> bool | None:
	ended = (gt.get("bare") or {}).get("ended")
	return None if ended is None else ended == "oom"


def _same_gpu(a: str, b: str) -> bool:
	from preflight.engine import gpus

	try:
		return gpus.resolve(a) == gpus.resolve(b)
	except KeyError:
		return a == b


def _within(pred: float | None, actual: float | None, tol: float) -> bool | None:
	if not pred or not actual:
		return None
	return abs(pred / actual - 1) <= tol


def _inside(e, actual: float | None) -> bool | None:
	if actual is None or e is None or e.status == "abstain" or e.lo is None or e.hi is None:
		return None
	return e.lo <= actual <= e.hi


# ---------------------------------------------------------------- scoring


@dataclass
class Scored:
	"""What one full run contributes, row by row."""

	script: str
	set: str
	memory: bool | None = None
	memory_inside: bool | None = None
	oom_actual: bool | None = None
	oom_p: float | None = None
	step_same: bool | None = None
	step_other: bool | None = None
	step_inside: bool | None = None
	rollout: bool | None = None
	rollout_inside: bool | None = None
	serving: bool | None = None
	serving_inside: bool | None = None
	agent_inside: bool | None = None
	planted: tuple[int, int] | None = None
	harnessed: bool | None = None
	preflight_ok: bool | None = None
	notes: list[str] = field(default_factory=list)


def score_run(run: dict, base: Path) -> Scored:
	from preflight.engine import api

	if run.get("set") == "unseen" and run.get("used_for_tuning"):
		raise ValueError(f"{run.get('script')}: an unseen script's rows are never tuned on")
	rec = _record(base / run["record"])
	gt = _load(base / run["ground_truth"]) if run.get("ground_truth") else {}
	pred = api.predict(rec, run.get("values") or None).root
	s = Scored(run.get("script", "?"), run.get("set", "zoo"))

	bare = gt.get("bare") or {}
	peak = bare.get("peak_reserved_bytes")
	mem = pred.get("memory.peak")
	s.oom_actual = ran_out(gt)
	# only a run that finished showed its full need: one that ran out, errored or timed out stopped short of it
	if bare.get("ended") in FINISHED and peak:
		s.memory = _within(mem.point if mem and mem.status != "abstain" else None, peak, 0.10)
		s.memory_inside = _inside(mem, peak)
	oom = pred.get("risk.oom")
	if oom is not None:
		s.oom_p = oom.point if oom.status != "abstain" else None

	step = pred.get("time.step")
	actual_step = steady_step_s(gt)
	probe_gpu = api._env_gpu(rec)
	run_gpu = (run.get("values") or {}).get("gpu_type") or gt.get("gpu")
	same = None if probe_gpu is None or run_gpu is None else _same_gpu(probe_gpu, run_gpu)
	if same is None:
		s.notes.append(
			f"{s.script}: the probe's or the full run's GPU isn't known, so its step time isn't scored"
		)
	if same is not None and step is not None and step.status != "abstain" and actual_step:
		ok = _within(step.point, actual_step, 0.15 if same else 0.20)
		if same:
			s.step_same = ok
		else:
			s.step_other = ok
		s.step_inside = _inside(step, actual_step)

	if run.get("rollout_s") is not None and "time.rollout" in pred:
		e = pred["time.rollout"]
		s.rollout = _within(e.point if e.status != "abstain" else None, run["rollout_s"], 0.20)
		s.rollout_inside = _inside(e, run["rollout_s"])
	if run.get("tokens_per_s") is not None and "serving.throughput" in pred:
		e = pred["serving.throughput"]
		s.serving = _within(e.point if e.status != "abstain" else None, run["tokens_per_s"], 0.15)
		s.serving_inside = _inside(e, run["tokens_per_s"])
	if run.get("agent_total_usd") is not None and "cost.total" in pred:
		s.agent_inside = _inside(pred["cost.total"], run["agent_total_usd"])

	if run.get("planted"):
		found = {f.code for f in rec.findings}
		want = list(run["planted"])
		s.planted = (sum(c in found for c in want), len(want))
		missed = [c for c in want if c not in found]
		if missed:
			s.notes.append(f"{s.script} missed {', '.join(missed)}")
	if run.get("set") == "unseen" and "harnessed_without_edits" in run:
		s.harnessed = bool(run["harnessed_without_edits"])
	usd = run.get("preflight_usd", rec.costs.total_usd if rec.costs else None)
	secs = run.get("preflight_s")
	if usd is not None and secs is not None:
		s.preflight_ok = usd < PREFLIGHT_USD and secs < PREFLIGHT_S
	return s


def _share_row(metric, target, on, flags: list[tuple[str, bool | None]], share=WITHIN_SHARE, note="") -> Row:
	got = [(name, f) for name, f in flags if f is not None]
	if not got:
		return Row(metric, target, on, note=note or "no full run scored yet")
	hits = sum(f for _, f in got)
	misses = [name for name, f in got if not f]
	return Row(
		metric,
		target,
		on,
		hits,
		len(got),
		hits / len(got) >= share,
		note or (f"misses: {', '.join(misses)}" if misses else ""),
		misses,
	)


def cross_gpu_row(stores: list[Path]) -> Row:
	"""Step time on a GPU the corpus never ran: train on the other GPUs, test on the one left out (K1 within 20%)."""
	metric, target, on = "Step time, unmeasured GPU", "Within 20%", "Train on 3 GPU types, test on the 4th"
	if not stores:
		return Row(metric, target, on, note="no corpus store given")
	from preflight.corpus import calibrate as cal
	from preflight.corpus import crossgpu

	rows = [r for p in stores for r in cal.load_rows(p)]
	cs = crossgpu.cases(rows)
	if not cs:
		return Row(metric, target, on, note="no cross-GPU pairs in the store")
	hits = sum(abs(c["k1"] / c["actual"] - 1) <= 0.20 for c in cs)
	gpus = sorted({c["tgt"] for c in cs})
	amd = any(g.startswith("MI") for g in gpus)
	return Row(
		metric,
		target,
		on + ("" if amd else "; no AMD measurements"),
		hits,
		len(cs),
		hits / len(cs) >= WITHIN_SHARE,
		f"K1 roofline, {len(cs)} config pairs over {', '.join(gpus)}",
	)


def _with_full_runs(row: Row, flags: list[tuple[str, bool | None]]) -> Row:
	"""Full runs on a GPU the probes didn't run on, added to the corpus row's note (the corpus sets the count)."""
	got = [f for _, f in flags if f is not None]
	if got:
		extra = f"full runs on another GPU: {sum(got)} of {len(got)} within 20%"
		row.note = f"{row.note}; {extra}" if row.note else extra
	return row


# what a manifest's "calibrates" names, per Scored field: "step" covers same-GPU, other-GPU and the band
_ROWS = {"memory": "memory", "step_same": "step", "step_other": "step", "rollout": "rollout", "serving": "serving",
	"agent": "agent"}  # fmt: skip


def _row_of(attr: str) -> str:
	base = attr.removesuffix("_inside")
	return _ROWS.get(base, base)


def models_used(runs: list[dict], base: Path) -> list[str]:
	"""Every model id and version the engine scored these runs with, and the calibration table's version, so the
	table can show that nothing was retuned on the unseen scripts (Chiya, Nov 1: the engine is frozen at the
	validation-freeze tag)."""
	from preflight.engine import api, conformal

	ids: set[str] = set()
	for run in runs:
		rec = _record(base / run["record"])
		ids |= {e.model_id for e in api.predict(rec, run.get("values") or None).root.values() if e.model_id}
	return [*sorted(ids), f"calibration table {conformal.TABLE.version}"]


def calibration_note(manifest: dict) -> str:
	"""The runs that calibrated a model, by name: they are 'used for calibration', never held out."""
	used = [
		f"{r.get('script')} ({'every row' if '*' in r['calibrates'] else ', '.join(r['calibrates'])})"
		for r in manifest.get("runs", [])
		if r.get("calibrates")
	]
	return f"Used for calibration, so not held out: {'; '.join(used)}." if used else ""


def build(manifest: dict, base: Path, divergence: dict | None = None) -> tuple[list[Row], list[Scored]]:
	all_runs = [score_run(r, base) for r in manifest.get("runs", [])]
	# per run, not per script: two runs of one script can differ (review, Nov 1). "*" is a run used for calibration
	# altogether (Chiya, Nov 1: the Oct 31 LoRA run), which no held-out row scores
	all_cal = [set(r.get("calibrates") or ()) for r in manifest.get("runs", [])]
	keep = [i for i, cal in enumerate(all_cal) if "*" not in cal]
	runs = [all_runs[i] for i in keep]
	calibrates = [all_cal[i] for i in keep]

	def flags(attr, only=None):
		row = _row_of(attr)  # a run that calibrated a model is neither its point nor its band check
		return [
			(r.script, getattr(r, attr))
			for r, cal in zip(runs, calibrates, strict=True)
			if (only is None or r.set == only) and row not in cal
		]

	rows = [
		_share_row(
			"Peak memory error", "Within 10%", "Bug zoo and unseen scripts, full runs", flags("memory")
		),
	]
	# OOM: no missed OOMs, at most 1 false alarm per 10 scripts; the reliability bins are in the notes
	scored = [r for r in runs if r.oom_actual is not None]
	if scored:
		missed = [r.script for r in scored if r.oom_actual and (r.oom_p is None or r.oom_p < OOM_ALARM)]
		alarms = [
			r.script for r in scored if not r.oom_actual and r.oom_p is not None and r.oom_p >= OOM_ALARM
		]
		n = len(scored)
		ok = not missed and len(alarms) <= n / 10
		bins = reliability(scored)
		rows.append(
			Row(
				"OOM prediction",
				"No missed OOMs; at most 1 false alarm per 10 scripts",
				"Bug zoo and unseen scripts",
				n - len(missed) - len(alarms),
				n,
				ok,
				f"missed: {', '.join(missed) or 'none'}; false alarms: {', '.join(alarms) or 'none'}; "
				f"reliability: {bins}",
			)
		)
	else:
		rows.append(
			Row(
				"OOM prediction",
				"No missed OOMs; at most 1 false alarm per 10 scripts",
				"Bug zoo and unseen scripts",
			)
		)
	rows += [
		_share_row("Step time, same GPU", "Within 15%", "Full runs", flags("step_same")),
		_with_full_runs(cross_gpu_row([base / p for p in manifest.get("corpus", [])]), flags("step_other")),
		Row(
			"GNN + GBDT ensemble vs GBDT alone",
			"Lower error on held-out architectures",
			"Corpus holdout",
			None,
			0,
			GNN_VS_GBDT["lo"] > 0,
			f"paired gain {GNN_VS_GBDT['gain']:+.1%} (95% {GNN_VS_GBDT['lo']:+.1%} to {GNN_VS_GBDT['hi']:+.1%}); "
			+ (
				"the mix is used"
				if GNN_VS_GBDT["lo"] > 0
				else "the interval includes zero, so the GBDT is used"
			),
		),
		divergence_row(divergence),
		_share_row(
			"Rollout time (RL)",
			"Within 20%",
			"An unseen TRL script (the GRPO zoo script calibrated it, Nov 1)",
			flags("rollout"),
		),
		_share_row("Serving tokens/sec", "Within 15%", "vLLM full runs", flags("serving")),
		_share_row(
			"Agent run total cost",
			"Inside the 90% interval",
			"Full benchmark run vs the 30-task sample",
			flags("agent_inside"),
			share=1.0,
		),
	]
	inside = [
		(f"{r.script}:{k}", getattr(r, k))
		for r, cal in zip(runs, calibrates, strict=True)
		for k in ("memory_inside", "step_inside", "rollout_inside", "serving_inside", "agent_inside")
		if _row_of(k) not in cal and "intervals" not in cal
	]
	rows.append(
		_share_row(
			"Interval coverage",
			"90% intervals cover at least 88%",
			"All of the above",
			inside,
			COVERAGE_TARGET,
		)
	)
	rows.append(
		Row(
			"Abstention",
			"Every query outside the probed range is flagged",
			"Automated test",
			note="tests/test_engine_api.py, run with the suite",
		)
	)
	planted = [r.planted for r in runs if r.planted]
	if planted:
		h, n = sum(p[0] for p in planted), sum(p[1] for p in planted)
		missed = "; ".join(m for r in runs for m in r.notes if "missed" in m)
		rows.append(Row("Planted problems caught", "All of them", "Bug zoo", h, n, h == n, missed))
	else:
		rows.append(Row("Planted problems caught", "All of them", "Bug zoo"))
	harness = [(r.script, r.harnessed) for r in runs if r.set == "unseen"]
	got = [f for _, f in harness if f is not None]
	rows.append(
		Row(
			"Unseen scripts harnessed",
			"8 of 9 without hand edits",
			"One per job type",
			sum(got) if got else None,
			len(got) if got else None,
			(sum(got) >= HARNESS_TARGET[0] * len(got) / HARNESS_TARGET[1]) if got else None,
		)
	)
	ms = manifest.get("knob_ms") or []
	if ms:
		p95 = float(sorted(ms)[max(0, math.ceil(0.95 * len(ms)) - 1)])
		under = sum(m < KNOB_P95_MS for m in ms)
		rows.append(
			Row(
				"Knob response",
				"Under 400 ms at p95",
				"Browser timing",
				under,
				len(ms),
				p95 < KNOB_P95_MS,
				f"p95 {p95:.0f} ms",
			)
		)
	else:
		rows.append(
			Row("Knob response", "Under 400 ms at p95", "Browser timing", note="no browser timings given")
		)
	rows.append(
		_share_row(
			"Cost of one Preflight run",
			"Under $1 and under 5 minutes",
			"Modal billing and timer",
			flags("preflight_ok"),
			share=1.0,
		)
	)
	return rows, all_runs


def reliability(scored: list[Scored]) -> str:
	"""P(OOM) against what happened, in the bins the readout shows: 'predicted range: k of n ran out'."""
	edges = [(0.0, 0.05), (0.05, 0.5), (0.5, 0.95), (0.95, 1.01)]
	parts = []
	for lo, hi in edges:
		inb = [r for r in scored if r.oom_p is not None and lo <= r.oom_p < hi]
		if inb:
			parts.append(
				f"{lo:.0%}-{min(hi, 1):.0%}: {sum(bool(r.oom_actual) for r in inb)} of {len(inb)} ran out"
			)
	none = [r for r in scored if r.oom_p is None]
	if none:
		parts.append(
			f"uncertain or abstained: {sum(bool(r.oom_actual) for r in none)} of {len(none)} ran out"
		)
	return "; ".join(parts) or "no P(OOM) shown"


def divergence_row(d: dict | None) -> Row:
	"""From the divergence classifier's held-out report: {"auroc": {point, lo, hi}, "lead_steps": {...}, "n": ...}."""
	metric, target, on = "Divergence warning", "AUROC 0.85+, 100+ steps early", "Held-out short runs"
	if not d:
		return Row(metric, target, on, note="the classifier's held-out report isn't in yet")
	a, s = d["auroc"], d["lead_steps"]
	hit = a["point"] >= 0.85 and s["point"] >= 100
	return Row(
		metric,
		target,
		on,
		None,
		d.get("n", 0),
		hit,
		f"AUROC {a['point']:.2f} ({a['lo']:.2f} to {a['hi']:.2f}); warns {s['point']:.0f} steps early "
		f"({s['lo']:.0f} to {s['hi']:.0f}); bootstrap over settings families (the sweep is one architecture)",
	)


# ---------------------------------------------------------------- output

HEADER = ["Metric", "Target", "Measured on", "Result", "95% interval", "Hit", "Notes"]
START, END = "<!-- validation:start -->", "<!-- validation:end -->"


def markdown(rows: list[Row], stamp: str = "") -> str:
	def esc(c: str) -> str:
		return c.replace("|", "\\|")

	lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
	for r in rows:
		c = r.cells()
		if r.n == 0 and r.hit is not None:  # a corpus verdict with no run count
			c[3], c[4] = "", ""
		lines.append("| " + " | ".join(esc(x) for x in c) + " |")
	foot = (
		f'\nA "within" target is a hit when at least {WITHIN_SHARE:.0%} of runs are within it; intervals are Wilson '
		f'95% on the proportion, and a hit is "confident" only when its lower end is at least {CONFIDENT:.0%} '
		'("hit, few runs" otherwise). Unseen scripts are never tuned on.'
	)
	return "\n".join(lines) + foot + (f" {stamp}" if stamp else "") + "\n"


def into_readme(readme: Path, table: str) -> bool:
	"""Replace the block between the validation markers; False (and nothing written) when they're missing."""
	text = readme.read_text()
	if START not in text or END not in text:
		return False
	head, rest = text.split(START, 1)
	_, tail = rest.split(END, 1)
	readme.write_text(f"{head}{START}\n{table}{END}{tail}")
	return True
