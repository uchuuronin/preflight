"""The job function: fetch, map, skeleton, plan, probes, merge. One call per leased job.

  fetching    the source (zoo sample, upload, notebook or GitHub repo)
  mapping     the static reader; a skeleton record is written right away, so the
              visitor sees the map within seconds without waiting for Muse Spark
  skeleton    record rev 1; Muse Spark's reading, when a client is configured,
              is reconciled after it and written as the next revision
  harnessing  the probe plan (probes/plan.py) and the first GPU (probes/gpu_choice.py)
  probing     each planned point: overrides applied to a copy, run, checked;
              a point whose override can't be applied is skipped, not fatal
  done        the merged record: probes, env, model shape, gpu_type, costs

Failure policy, per probe point:
  - override can't be applied        skip the point, record why, go on
  - training-step GPU OOM            next GPU up (once, or the whole ladder when
                                     the model's size was unknown)
  - eval OOM                         re-run on the same GPU with eval off; the
                                     eval memory per batch says where it ran out
  - job budget (time or dollars)     stop probing, keep what was measured
  - the probe's own failure          recorded on the probe (status, error code)
Nothing here retries a probe that failed for the visitor's reasons.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

from preflight.harness.mapper import map_script
from preflight.harness.skeleton import skeleton_record
from preflight.harness.static import map_static
from preflight.probes import untrusted
from preflight.probes.gpu_choice import (
	choose_probe_gpu,
	estimate_need_gb,
	max_steps_up,
	probe_on_ladder,
	usd_per_hour,
)
from preflight.probes.local import InvalidSubmission, ProbeResult, ProbeSpec, run_probe
from preflight.probes.plan import PlannedProbe, plan_probes
from preflight.recorder.overrides import OverrideError, apply_overrides, override_args, values_from_args
from preflight.schema import Constraints, Env, FlightRecord, Objective, Probe, ProbeCost
from preflight.schema.script_map import ScriptMap

from .records import RecordWriter
from .sources import Fetched, fetch
from .worker import JobContext

log = logging.getLogger("preflight.pipeline")

# per-job caps (Contract A "Done means": a record costs under $0.60 of GPU). guess: replaced by the
# measured distribution of probe costs once the zoo runs on Modal (Wed Oct 28)
JOB_BUDGET_USD = 0.60
JOB_PROBE_WALL_S = 15 * 60
PROBE_TIMEOUT_S = (
	6 * 60
)  # guess: one probe point with model download and the profiled window; zoo runs set it (Oct 28-29)


@dataclass
class Budget:
	usd: float = JOB_BUDGET_USD
	wall_s: float = JOB_PROBE_WALL_S
	spent_usd: float = 0.0
	started: float = field(default_factory=time.monotonic)

	def left_s(self) -> float:
		return self.wall_s - (time.monotonic() - self.started)

	def allows(self, est_usd: float, est_s: float) -> bool:
		return self.spent_usd + est_usd <= self.usd and est_s <= self.left_s()


class ProbeRunner(Protocol):
	def __call__(self, spec: ProbeSpec, gpu: str, *, should_stop: Callable[[], bool]) -> ProbeResult: ...


def local_runner(spec: ProbeSpec, gpu: str, *, should_stop: Callable[[], bool]) -> ProbeResult:
	"""Runs on this machine's device, whatever `gpu` says (development, tests, a visitor's own box): no GPU bill."""
	return run_probe(replace(spec, gpu_usd_per_hour=0.0), should_stop=should_stop)


class ProbeCache(Protocol):
	def get(self, key: str) -> dict | None: ...
	def put(self, key: str, probe: dict) -> None: ...


class MemoryCache:
	def __init__(self):
		self._d: dict[str, dict] = {}

	def get(self, key: str) -> dict | None:
		return self._d.get(key)

	def put(self, key: str, probe: dict) -> None:
		self._d[key] = probe


def cache_key(spec: ProbeSpec, gpu: str, image: str) -> str:
	"""Content hash of everything that decides a measurement: code after overrides, args, steps, GPU, image."""
	h = hashlib.sha256()
	body = {
		"files": {k: hashlib.sha256(v.encode("utf-8", "surrogatepass")).hexdigest() for k, v in sorted(spec.files.items())},
		"entrypoint": spec.entrypoint,
		"args": spec.args,
		"warmup": spec.warmup,
		"timed": spec.timed,
		"eval_batches": spec.eval_batches,
		"profile": spec.profile,
		"seed": spec.seed,
		"gpu": gpu,
		"image": image,
	}  # fmt: skip
	h.update(json.dumps(body, sort_keys=True).encode())
	return "sha256:" + h.hexdigest()


@dataclass
class Deps:
	records: RecordWriter
	runner: ProbeRunner = local_runner
	spark: object | None = None  # harness.spark.ModelClient
	cache: ProbeCache | None = None
	image: str = "local"  # probe image tag, part of the cache key
	budget: Callable[[], Budget] = Budget
	probe_timeout_s: float = PROBE_TIMEOUT_S
	blobs: object | None = None  # remote_store.Blobs: uploaded files the job store moved out of its state


@dataclass
class _Ctx:
	"""Stand-in for a worker's JobContext when the pipeline runs outside the queue (CLI, tests)."""

	events: list = field(default_factory=list)

	def check(self) -> None:
		pass

	def should_stop(self) -> bool:
		return False

	def emit(self, stage: str, **fields) -> None:
		self.events.append({"stage": stage, **fields})


def _emit(ctx, stage: str, rec: FlightRecord | None = None, rev: int | None = None, **fields):
	if rec is not None:
		fields.update(record_id=rec.record_id, share_token=rec.share_token, record_rev=rev)
	ctx.emit(stage, **{k: v for k, v in fields.items() if v is not None})


# ---------------------------------------------------------------- stages


def _skeleton(
	fetched: Fetched, sm: ScriptMap, request: dict, base: FlightRecord | None = None
) -> FlightRecord:
	constraints = Constraints.model_validate(request["constraints"]) if request.get("constraints") else None
	objective = Objective.model_validate(request["objective"]) if request.get("objective") else None
	extra = {"objective": objective} if objective is not None else {}
	rec = skeleton_record(sm, fetched.source, gpu=request.get("gpu"), constraints=constraints, **extra)
	if base is not None:  # same record, next revision
		rec = rec.model_copy(
			update={
				"record_id": base.record_id,
				"share_token": base.share_token,
				"created_at": base.created_at,
			}
		)
	return rec


def _skipped(p: PlannedProbe, why: str) -> Probe:
	return Probe(
		id=p.probe_id,
		knobs=p.knobs,
		status="error",
		steps=0,
		error=why[:500],
		error_code="override.unsupported",
	)


def _spec(
	fetched: Fetched, sm: ScriptMap, p: PlannedProbe, deps: Deps, *, eval_off: bool = False
) -> ProbeSpec:
	overrides = dict(p.overrides)
	eval_batches = p.eval_batches
	if eval_off:
		overrides.pop("eval_every", None)
		eval_batches = 0
	files = apply_overrides(fetched.files, sm, overrides)
	args = override_args(fetched.files, sm, fetched.args, overrides)
	return ProbeSpec(
		probe_id=p.probe_id,
		files=files,
		entrypoint=fetched.entrypoint,
		args=args,
		knobs=p.knobs,
		warmup=p.warmup,
		timed=p.timed,
		eval_batches=eval_batches if eval_batches else 0,
		profile=p.profile,
		timeout_s=deps.probe_timeout_s,
	)


def _run_point(spec: ProbeSpec, gpu: str, deps: Deps, ctx) -> tuple[ProbeResult | None, Probe | None]:
	"""Run one point, or answer it from the cache. Returns (fresh result, cached probe)."""
	key = cache_key(spec, gpu, deps.image)
	if deps.cache is not None:
		hit = deps.cache.get(key)
		if hit is not None:
			try:
				probe = Probe.model_validate(hit)
				# this point's own labels, not the run that filled the cache (and not when it ran)
				return None, probe.model_copy(
					update={"status": "cached", "cache_key": key, "id": spec.probe_id, "knobs": spec.knobs,
						"started_at": None, "cost": ProbeCost()}
				)  # fmt: skip
			except Exception:
				log.warning("ignoring an unreadable probe-cache entry %s", key)
	spec = replace(spec, gpu_usd_per_hour=_price(gpu))
	result = deps.runner(spec, gpu, should_stop=ctx.should_stop)
	result.probe = result.probe.model_copy(update={"cache_key": key})
	if deps.cache is not None and result.probe.status == "ok":
		deps.cache.put(key, json.loads(result.probe.model_dump_json(exclude_none=True)))
	return result, None


def run_pipeline(request: dict, ctx, deps: Deps, *, ids: tuple[str, str] | None = None) -> FlightRecord:
	"""request: a JobRequest body. ctx: a worker JobContext (or _Ctx). Returns the final record.

	ids: (record_id, share_token) an earlier attempt of this job started; the retry continues
	that record so a visitor already watching it keeps the same link."""
	_emit(ctx, "fetching")
	fetched = fetch(request.get("source") or {}, blobs=deps.blobs)
	ctx.check()

	_emit(ctx, "mapping")
	sm = values_from_args(
		fetched.files, map_static(fetched.files, entrypoint=fetched.entrypoint), fetched.args
	)
	rec = _skeleton(fetched, sm, request)
	if ids and all(ids):
		rec = rec.model_copy(update={"record_id": ids[0], "share_token": ids[1]})
	rev = deps.records.put(rec)
	_emit(ctx, "skeleton", rec, rev, message="skeleton from the static reader")

	if deps.spark is not None:
		ctx.check()
		spark_sm, calls = map_script(fetched.files, entrypoint=fetched.entrypoint, client=deps.spark)
		if calls:
			sm = values_from_args(fetched.files, spark_sm, fetched.args)
			rec = _skeleton(fetched, sm, request, base=rec)
		rec = rec.model_copy(update={"provenance": [*rec.provenance, *calls]})
		rev = deps.records.put(rec)
		_emit(ctx, "skeleton", rec, rev, message="map refined by Muse Spark")

	if rec.job.lens == "agent":
		# agent runs are sampled by the agent recorder (OTel proxy), not by GPU probes
		_emit(ctx, "harnessing", rec, rev, message="agent jobs run through the agent sampler")
		return rec

	ctx.check()
	_emit(ctx, "harnessing", rec, rev)
	plan = plan_probes(rec)
	allowed = _allowed_gpus(rec)
	need = estimate_need_gb(sm)
	gpu = _requested_gpu(request, allowed) or choose_probe_gpu(need, allowed)
	budget = deps.budget()

	probes: list[Probe] = []
	results: list[ProbeResult] = []
	discarded_usd = 0.0
	for i, p in enumerate(plan):
		ctx.check()
		_emit(ctx, "probing", rec, rev, probe_id=p.probe_id, progress=round(i / len(plan), 3))
		est_s = max((r.probe.duration_s or 0.0 for r in results), default=60.0)
		if probes and not budget.allows(_price(gpu) * est_s / 3600, est_s):
			why = (
				f"skipped: the job's probe budget (${budget.usd:.2f}, {budget.wall_s / 60:.0f} min) is spent"
			)
			probes.append(
				Probe(id=p.probe_id, knobs=p.knobs, status="error", steps=0, error=why, error_code="budget")
			)
			continue
		try:
			spec = _spec(fetched, sm, p, deps)
		except OverrideError as exc:
			probes.append(_skipped(p, f"this point can't be set in the code: {exc}"))
			continue
		attempts: list = []

		def run(g, spec=spec, attempts=attempts):
			result, cached = _run_point(spec, g, deps, ctx)
			out = result if result is not None else _CachedResult(cached)
			if result is not None:
				budget.spent_usd += result.probe.cost.usd  # every attempt counts, kept or not
			attempts.append(out)
			return out

		def may_step_up(g, est_s=est_s):  # another GPU only while the job is live and the budget holds it
			return not ctx.should_stop() and budget.allows(_price(g) * est_s / 3600, est_s)

		steps = max_steps_up(need) if i == 0 else 1
		ladder = probe_on_ladder(run, gpu, allowed, steps=steps, may_step_up=may_step_up)
		out = ladder.result
		fits = out.probe.status != "oom" or out.probe.error_code == "oom.eval"
		if i == 0 and len(ladder.tried) > 1 and fits:
			gpu = ladder.tried[-1]  # the GPU that held the current settings runs the smaller points too
		if out.probe.status == "oom" and out.probe.error_code == "oom.eval":
			try:
				# the eval batch ran out, not training: same GPU, eval off; the eval probe keeps
				# its memory per batch so the guide can say where it ran out
				again = run(ladder.tried[-1], spec=_spec(fetched, sm, p, deps, eval_off=True))
				probes.append(out.probe.model_copy(update={"id": f"{p.probe_id}-eval"}))
				if isinstance(out, ProbeResult):
					results.append(out)
				out = again
			except OverrideError:
				pass
		probe = out.probe
		if ladder.step_up_error:
			probe = probe.model_copy(
				update={"error": ((probe.error or "") + " " + ladder.step_up_error).strip()}
			)
		probes.append(probe)
		if isinstance(out, ProbeResult):
			results.append(out)
		kept_ids = {id(out), *(id(r) for r in results)}
		discarded_usd += sum(
			a.probe.cost.usd for a in attempts if isinstance(a, ProbeResult) and id(a) not in kept_ids
		)
	billed = deps.runner.billed_usd(_price) if hasattr(deps.runner, "billed_usd") else None
	return _merge(
		rec, probes, results, gpu, spent_usd=budget.spent_usd, discarded_usd=discarded_usd, billed_usd=billed
	)


def _price(gpu: str) -> float:
	p = usd_per_hour(gpu)
	return 0.0 if p == float("inf") else p


def _offered() -> set[str]:
	from preflight.probes.gpu_choice import offered_gpus

	return set(offered_gpus())


def _allowed_gpus(rec: FlightRecord) -> list[str] | None:
	"""The visitor's allowed GPUs, in canonical names; one we can't run is a visitor error, not ours."""
	from preflight.probes.gpu_choice import canonical

	types = rec.job.constraints.gpu_types
	if not types:
		return None
	names = [canonical(g) for g in types]
	unknown = [g for g, n in zip(types, names, strict=True) if n not in _offered()]
	if unknown:
		raise InvalidSubmission(
			f"Preflight can't run on {', '.join(unknown)}; choose from {', '.join(sorted(_offered()))}"
		)
	return list(dict.fromkeys(names))


def _requested_gpu(request: dict, allowed: list[str] | None) -> str | None:
	from preflight.probes.gpu_choice import canonical

	g = request.get("gpu")
	if not g:
		return None
	name = canonical(str(g))
	if name not in _offered() or (allowed and name not in allowed):
		raise InvalidSubmission(
			f"Preflight can't run on {g!r}; choose from {', '.join(sorted(allowed or _offered()))}"
		)
	return name


@dataclass
class _CachedResult:
	"""A probe answered from the cache: no env, no kernels, no shape beyond what the Probe carries."""

	probe: Probe
	env: Env | None = None
	arch: dict | None = None
	kernels: object = None


def _merge(
	rec: FlightRecord,
	probes: list[Probe],
	results: list[ProbeResult],
	gpu: str,
	*,
	spent_usd: float | None = None,
	discarded_usd: float = 0.0,
	billed_usd: float | None = None,
) -> FlightRecord:
	env = next((r.env for r in results if r.probe.status == "ok" and r.env is not None), None)
	env = env or next((r.env for r in results if r.env is not None), None)
	update: dict = {"probes": probes}
	if env is not None:
		update["env"] = env
	job = rec.job
	arch = next((r.arch for r in results if r.arch), None)
	if job.script_map is not None and arch:
		sm = job.script_map.model_copy(update={"model": untrusted.with_shape(job.script_map.model, arch)})
		job = job.model_copy(update={"script_map": sm})
		update["job"] = job
	knobs = []
	seen = env.gpu.name if env is not None and env.gpu is not None else gpu
	for k in rec.knobs:
		if k.id == "gpu_type" and seen:
			values = list(k.domain.values or [])
			if seen not in values:
				values.append(seen)
			k = k.model_copy(
				update={"current": seen, "domain": k.domain.model_copy(update={"values": values})}
			)
		knobs.append(k)
	update["knobs"] = knobs
	kept = round(sum(p.cost.usd for p in probes if p.status != "cached"), 6)
	# what the job really cost: every attempt (a probe that ran out and was re-run on a bigger GPU
	# still ran), and the sandbox's whole lifetime when the runner knows it (start-up, idle)
	gpu_usd = max(kept, spent_usd or 0.0, billed_usd or 0.0)
	breakdown = {**rec.costs.breakdown, "probes": kept}
	if discarded_usd:
		breakdown["discarded_attempts"] = round(discarded_usd, 6)
	if billed_usd and billed_usd > kept + discarded_usd:
		breakdown["sandbox_overhead"] = round(billed_usd - kept - discarded_usd, 6)
	costs = rec.costs.model_copy(update={"gpu_usd": round(gpu_usd, 6), "breakdown": breakdown})
	update["costs"] = costs
	return FlightRecord.model_validate({**rec.model_dump(), **{k: v for k, v in update.items()}})


def run_job(lease, ctx: JobContext, *, deps: Deps) -> str:
	"""worker.JobFn: run the pipeline for a leased job, store the final record, return its id."""
	rec = run_pipeline(lease.request, ctx, deps, ids=(lease.record_id, lease.share_token))
	rev = deps.records.put(rec)
	_emit(ctx, "predicting", rec, rev, message="measured; predictions are B's next")
	return rec.record_id
