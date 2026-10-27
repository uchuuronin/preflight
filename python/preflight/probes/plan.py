"""Which probes to run for a record, and how long each runs.

Defaults come from the Oct 27 Monte Carlo (docs/decisions/probe-plan.md):
  - warmup 2, timed 5, median step time: 99.3% of trials within 15% of the
    long-run mean, against 89% for 3 timed steps averaged;
  - three memory probes (current, half batch, half sequence) so the fit can
    separate the bs·s terms from the bs·s² attention term: 88% of queries
    within 10% and 1.0% missed OOMs, against 66% and 4.4% for two probes;
  - agent runs sample 30 tasks (pending Chiya's call between 30 and 50).

The current-settings probe also pulls the first eval and one checkpoint
inside the probe window, so an eval-time OOM or a slow save is measured, not
guessed. The other probes keep the script's own eval and checkpoint cadence.
"""

from dataclasses import dataclass, field

from preflight.schema import FlightRecord

WARMUP_STEPS = 2
TIMED_STEPS = 5
EVAL_BATCHES = 2
AGENT_TASKS = 30
MEMORY_LENSES = ("finetune", "pretrain", "rl", "vision", "gnn", "retrieval")


@dataclass
class PlannedProbe:
	probe_id: str
	role: str  # current, half_batch, half_seq, quarter_seq, quarter_batch, agent_sample
	knobs: dict = field(default_factory=dict)  # recorded on the Probe
	overrides: dict = field(default_factory=dict)  # applied to the code (knobs + cadence pulls)
	warmup: int = WARMUP_STEPS
	timed: int = TIMED_STEPS
	eval_batches: int = 0
	agent_tasks: int = 0
	profile: bool = False


def _int_knob(rec: FlightRecord, kid: str) -> int | None:
	try:
		k = rec.knob(kid)
	except KeyError:
		return None
	return (
		k.current
		if isinstance(k.current, int) and not isinstance(k.current, bool) and not k.needs_code
		else None
	)


def plan_probes(rec: FlightRecord) -> list[PlannedProbe]:
	lens = rec.job.lens
	if lens == "agent":
		return [PlannedProbe(probe_id="p1", role="agent_sample", agent_tasks=AGENT_TASKS, warmup=0, timed=0)]

	bs = _int_knob(rec, "batch_size")
	seq = _int_knob(rec, "seq_len")
	base = {k: v for k, v in (("batch_size", bs), ("seq_len", seq)) if v is not None}

	points: list[tuple[str, dict]] = [("current", dict(base))]
	if lens in MEMORY_LENSES and base:
		if bs is not None and seq is not None:
			if bs >= 2:
				points += [
					("half_batch", {**base, "batch_size": bs // 2}),
					("half_seq", {**base, "seq_len": max(1, seq // 2)}),
				]
			else:
				points += [
					("half_seq", {**base, "seq_len": max(1, seq // 2)}),
					("quarter_seq", {**base, "seq_len": max(1, seq // 4)}),
				]
		elif bs is not None and bs >= 4:
			points += [("half_batch", {"batch_size": bs // 2}), ("quarter_batch", {"batch_size": bs // 4})]
		elif seq is not None and seq >= 4:
			points += [("half_seq", {"seq_len": seq // 2}), ("quarter_seq", {"seq_len": seq // 4})]

	plan = []
	for i, (role, knobs) in enumerate(points, 1):
		p = PlannedProbe(probe_id=f"p{i}", role=role, knobs=knobs, overrides=dict(knobs))
		if role == "current":
			p.profile = True
			if _int_knob(rec, "eval_every") is not None:
				p.overrides["eval_every"] = WARMUP_STEPS + 1
				p.eval_batches = EVAL_BATCHES
			elif any(k.id in ("eval_batch_size", "eval_no_grad") for k in rec.knobs):
				p.eval_batches = EVAL_BATCHES
			if _int_knob(rec, "checkpoint_every") is not None:
				# two steps before the end, so it lands inside the probe whether the script counts from 0 or 1;
				# the recorder takes save time out of the step timings
				p.overrides["checkpoint_every"] = WARMUP_STEPS + TIMED_STEPS - 2
		# never record a no-op override
		p.overrides = {
			k: v for k, v in p.overrides.items() if k in knobs or k in ("eval_every", "checkpoint_every")
		}
		plan.append(p)
	return plan
