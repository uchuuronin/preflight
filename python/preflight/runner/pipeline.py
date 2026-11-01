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

Job modes (JobRequest.mode): plan is all of the above. probe, verify and
sample_more start from an existing record (runner/record_jobs.py) and write a
new record derived from it: probe runs one profiled point at the guide's
setting; verify runs that setting and the current one for VERIFY_STEPS at one
fixed seed and records whether the loss holds; sample_more is for agent runs.
GPUs: never above PREFLIGHT_GPU_CEILING (the public demo: A100-80GB), and a
step up after an OOM only while the job's dollar budget holds it.
"""

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

from pydantic import ValidationError

from preflight.harness.mapper import map_script
from preflight.harness.skeleton import skeleton_record
from preflight.harness.static import map_static
from preflight.probes import untrusted
from preflight.probes.gpu_choice import (
	ceiling as gpu_ceiling,
)
from preflight.probes.gpu_choice import (
	choose_probe_gpu,
	estimate_need_gb,
	max_steps_up,
	probe_on_ladder,
	under_ceiling,
	usd_per_hour,
)
from preflight.probes.local import InvalidSubmission, ProbeResult, ProbeSpec, run_probe
from preflight.probes.plan import PlannedProbe, next_probe_id, plan_probes, point_probe
from preflight.recorder.overrides import OverrideError, apply_overrides, override_args, values_from_args
from preflight.schema import (
	Constraints,
	Env,
	FlightRecord,
	JobRequest,
	Objective,
	Predictions,
	Probe,
	ProbeCost,
	dump_record,
	load_record,
)
from preflight.schema.script_map import ScriptMap

from . import record_jobs
from .records import RecordReader, RecordWriter
from .sources import Fetched, fetch
from .worker import JobContext, PermanentError

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
	spent_usd: float = 0.0  # what the cap is checked against: GPU and the sandbox overhead estimate
	gpu_usd: float = 0.0  # GPU at measured probe time, every attempt
	overhead_usd: float = 0.0  # the sandbox CPU and memory estimate alone
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
		"seed_force": spec.seed_force,
		"install_requirements": spec.install_requirements,
		"gpu": gpu,
		"image": image if spec.image == "probe" else f"{image}+{spec.image}",
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
	reader: RecordReader | None = None  # stored records, for jobs that start from one (record_jobs.py)
	install_requirements: bool = False  # hosted probes pip-install requirements.txt; the CLI uses yours
	# what a probe's sandbox costs beside its GPU (Modal bills sandbox CPU and memory at 3x list),
	# counted against the job budget; 0 for local runs
	overhead_usd_per_h: float = 0.0
	# agent and API runs: runner/agent_jobs.py's sampler (the CLI's runs here with the user's key);
	# None where no sampler is wired (the hosted runner, for now)
	agent: Callable | None = None


@dataclass
class _Ctx:
	"""Stand-in for a worker's JobContext when the pipeline runs outside the queue (CLI, tests)."""

	events: list = field(default_factory=list)

	def check(self) -> None:
		pass

	def should_stop(self) -> bool:
		return False

	def confirm(self) -> None:
		pass

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


_VLLM_IMPORT = re.compile(r"^\s*(?:from\s+vllm\b|import\s+vllm\b)", re.M)


def _uses_vllm(sm: ScriptMap, files: dict[str, str]) -> bool:
	"""Serving, or any script that imports vLLM (an eval sweep on vLLM is lens eval): the probe
	image has no vLLM."""
	if sm.lens == "serving":
		return True
	# only an eval may need it: a GRPO script with an optional `import vllm` (use_vllm) needs trl
	return sm.lens == "eval" and any(_VLLM_IMPORT.search(t) for n, t in files.items() if n.endswith(".py"))


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
		seed=p.seed,
		seed_force=p.seed_force,
		install_requirements=deps.install_requirements,
		image="serving" if _uses_vllm(sm, fetched.files) else "probe",  # vLLM lives in its own image
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


def _validated(request: dict) -> JobRequest:
	if not isinstance(request, dict):
		raise InvalidSubmission("the job request is not an object", code="submission.invalid")
	try:
		return JobRequest.model_validate(request)
	except ValidationError as exc:
		first = exc.errors()[0] if exc.errors() else {}
		where = ".".join(str(x) for x in first.get("loc", ()))
		why = first.get("msg", "invalid")
		raise InvalidSubmission(
			f"the job request is invalid: {where + ': ' if where else ''}{why}"[:300],
			code="submission.invalid",
		) from exc


def run_pipeline(request: dict, ctx, deps: Deps, *, ids: tuple[str, str] | None = None) -> FlightRecord:
	"""request: a JobRequest body. ctx: a worker JobContext (or _Ctx). Returns the final record.

	ids: (record_id, share_token) an earlier attempt of this job started; the retry continues
	that record so a visitor already watching it keeps the same link.

	mode plan maps and probes a script; probe, verify and sample_more start from an existing
	record (runner/record_jobs.py) and write a new record derived from it."""
	req = _validated(request)
	source = request.get("source") or {}
	origin = None
	if source.get("kind") == "record":
		origin = record_jobs.resolve(source, reader=deps.reader, blobs=deps.blobs)
		source = origin.fetch_source
	_emit(ctx, "fetching")
	fetched = fetch(source, blobs=deps.blobs)
	if origin is not None and origin.parent is not None:
		was = origin.parent.job.source.content_hash
		if was and fetched.source.content_hash != was:
			raise InvalidSubmission("this record's code has changed since it was measured; check it in again",
				code="submission.code_changed")  # fmt: skip
	ctx.check()
	fixed = None
	if req.patch is not None:  # applied before anything is stored: a refused patch leaves no record
		from .patches import apply_patch

		fixed = replace(fetched, files=apply_patch(fetched.files, req.patch))

	_emit(ctx, "mapping")
	followup = req.mode != "plan"
	if followup and origin.parent is not None:
		# the parent's map and knobs carry over (no Muse Spark call): a new record, same script
		rec = record_jobs.derive(origin.parent, ids)
		sm = rec.job.script_map or values_from_args(
			fetched.files, map_static(fetched.files, entrypoint=fetched.entrypoint), fetched.args
		)
		rev = deps.records.put(rec)
		_emit(ctx, "skeleton", rec, rev, message=f"continuing from record {origin.parent.share_token}")
	else:
		sm = values_from_args(
			fetched.files, map_static(fetched.files, entrypoint=fetched.entrypoint), fetched.args
		)
		rec = _skeleton(fetched, sm, request)
		update = {"parent_id": origin.parent_id} if origin is not None else {}
		if ids and all(ids):
			update.update(record_id=ids[0], share_token=ids[1])
		if update:
			rec = rec.model_copy(update=update)
		rev = deps.records.put(rec)
		_emit(ctx, "skeleton", rec, rev, message="skeleton from the static reader")
		from .flags import flag

		# the job's code goes to Muse Spark only with harness.spark_script_map on: the data-use notice
		# discloses it under that flag, so the key being set is not enough (D's review, Chiya, Nov 1)
		if deps.spark is not None and not followup and flag("harness.spark_script_map"):
			ctx.check()
			spark_sm, calls = map_script(fetched.files, entrypoint=fetched.entrypoint, client=deps.spark)
			if calls:
				sm = values_from_args(fetched.files, spark_sm, fetched.args)
				rec = _skeleton(fetched, sm, request, base=rec)
				if origin is not None:
					rec = rec.model_copy(update={"parent_id": origin.parent_id})
			rec = rec.model_copy(update={"provenance": [*rec.provenance, *calls]})
			rev = deps.records.put(rec)
			_emit(ctx, "skeleton", rec, rev, message="map refined by Muse Spark")

	if req.mode == "sample_more":
		if rec.job.lens != "agent":
			raise InvalidSubmission(
				"sampling more tasks is for agent and API runs", code="submission.not_agent"
			)
		return _sample_agent(rec, rev, fetched, sm, deps, ctx, top_up=True, tasks=req.tasks)
	if rec.job.lens == "agent":
		if followup:
			raise InvalidSubmission(
				"agent runs are measured by sampling tasks, not GPU probes", code="submission.not_gpu_job"
			)
		if deps.agent is None:
			# agent runs are sampled through the agent recorder's proxy, not by GPU probes
			_emit(ctx, "harnessing", rec, rev, message="agent jobs run through the agent sampler")
			return rec
		return _sample_agent(rec, rev, fetched, sm, deps, ctx, top_up=False)

	ctx.check()
	_emit(ctx, "harnessing", rec, rev)
	allowed = _allowed_gpus(rec)
	need = estimate_need_gb(sm)
	pinned = False  # a GPU the guide asked for: an OOM there is the answer, not a reason to move
	plan_knobs: dict = {}
	base_knobs: dict = {}
	if req.mode == "plan":
		plan = plan_probes(rec)
	else:
		knobs, unknown = record_jobs.setting(rec, req.knobs)
		if unknown:
			_emit(ctx, "harnessing", rec, rev, level="warn",
				message=f"not knobs of this record, left as they are: {', '.join(sorted(unknown))}")  # fmt: skip
		plan_knobs = record_jobs.full_setting(rec, knobs)
		base_knobs = record_jobs.full_setting(rec, {})
		plan = _followup_plan(rec, req.mode, knobs, plan_knobs, base_knobs, patched=req.patch is not None)
		if "gpu_type" in knobs:
			request = {**request, "gpu": knobs["gpu_type"]}
			pinned = True
	gpu = _requested_gpu(request, allowed) or (
		_parent_gpu(rec) if followup and origin.parent is not None else None
	) or choose_probe_gpu(need, allowed)  # fmt: skip
	cap = gpu_ceiling()
	if not under_ceiling(gpu, cap):
		_emit(ctx, "harnessing", rec, rev, level="warn",
			message=f"probes here run on GPUs up to {cap}; {gpu} numbers come from the cross-GPU model")  # fmt: skip
		gpu = cap
	budget = deps.budget()
	patched_for: dict = {}
	if fixed is not None:
		# C's fix loop: probe runs the patched code; verify runs the setting on it and the base
		# and its second seed on the record's own code, so the finding reads before and after. The
		# knobs are found again in the patched code (a fix that adds a line moves them)
		from .patches import relocate_knobs

		fixed_sm = relocate_knobs(sm, fetched.files, fixed.files)  # the record's own knobs, Spark's included
		patched_ids = [p.probe_id for p in plan] if req.mode == "probe" else [plan[0].probe_id]
		patched_for = dict.fromkeys(patched_ids, (fixed, fixed_sm))
		_emit(ctx, "harnessing", rec, rev, message=f"patch applied to {', '.join(patched_ids)}")
	probes, results, gpu, discarded_usd = _probe_points(
		plan,
		rec,
		rev,
		fetched,
		sm,
		deps,
		ctx,
		budget,
		gpu=gpu,
		allowed=allowed,
		need=need,
		pinned=pinned,
		patched_for=patched_for,
	)
	billed = deps.runner.billed_usd(_price) if hasattr(deps.runner, "billed_usd") else None
	kept = rec.probes
	if followup and origin.parent is not None and gpu != _parent_gpu(origin.parent):
		# measured on another GPU: the record's env is this one, so the parent's probes don't
		# belong, nor what was fitted or found from them; the script's own findings stay
		kept = []
		dropped = {p.id for p in rec.probes}
		findings = [f for f in rec.findings if not any(e.probe_id in dropped for e in f.evidence)]
		rec = rec.model_copy(update={"predictions": Predictions(), "models": [], "findings": findings})
	if patched_for:
		# measured on patched code, not the record's: marked, so B's fits and the plan leave them out
		probes = [q.model_copy(update={"extras": {**q.extras, "patched": 1.0}}) if q.id in patched_for else q
			for q in probes]  # fmt: skip
	out = _merge(rec, [*kept, *probes], results, gpu, spent_usd=budget.gpu_usd, overhead_usd=budget.overhead_usd,
		discarded_usd=discarded_usd, billed_usd=billed, new_ids={p.id for p in probes})  # fmt: skip
	from .measured_rules import hygiene_findings

	# run hygiene, from the code the probes ran (a probe job's patch included, so a fix can clear
	# its own finding; a verify job's base is the record's own code) and the probes so far
	code = fixed.files if fixed is not None and req.mode == "probe" else fetched.files
	keep = [f for f in out.findings if not f.id.startswith(("hygiene.", "measured.rl."))]
	out = out.model_copy(update={"findings": [*keep, *hygiene_findings(out, code)]})
	if req.mode == "verify":
		by_id = {p.id: p for p in probes}
		vid = plan[0].probe_id
		finding = record_jobs.verify_finding(by_id[vid], by_id.get(f"{vid}-base"), plan_knobs, base_knobs,
			{n.id for n in out.graph.nodes}, other_seed=by_id.get(f"{vid}-seed"))  # fmt: skip
		if req.patch is not None:
			from preflight.schema import Evidence

			from .patches import patch_id

			finding = finding.model_copy(update={"evidence": [*finding.evidence,
				Evidence(metric="verify.patch", value=patch_id(req.patch), probe_id=vid)]})  # fmt: skip
		out = out.model_copy(update={"findings": [*out.findings, finding]})
	return out


def _sample_agent(rec: FlightRecord, rev, fetched: Fetched, sm, deps: Deps, ctx, *, top_up: bool,
	tasks: int | None = None) -> FlightRecord:  # fmt: skip
	"""30 tasks of the agent run through the proxy, or up to 20 more on top of the record's sample (D3)."""
	from preflight.engine.agent_cost import DEFAULT_SAMPLE, MAX_SAMPLE, TOP_UP_TASKS, combine_samples
	from preflight.recorder.agent import task_key
	from preflight.recorder.agent_sampler import draw_indices

	if deps.agent is None:
		raise PermanentError(
			"sampling agent tasks isn't available here yet", code="unsupported.agent_sampling"
		)
	first = rec.agent_sample
	if top_up and (first is None or not first.tasks):
		raise InvalidSubmission("this record has no task sample to add to; sample it first",
			code="submission.no_sample")  # fmt: skip
	ctx.check()
	population = getattr(getattr(sm, "agent", None), "n_tasks", None)
	if top_up:
		k = min(tasks or TOP_UP_TASKS, TOP_UP_TASKS, MAX_SAMPLE - len(first.tasks))
		if k <= 0:
			raise InvalidSubmission(f"this record's sample already has {len(first.tasks)} tasks",
				code="submission.sample_full")  # fmt: skip
		skip_keys = [key for t in first.tasks if (key := task_key(t.task_id))]
		if len(skip_keys) < len(first.tasks):
			# sampled before tasks were keyed by content: a top-up couldn't tell them apart and would
			# pay for them again
			raise PermanentError("this record's sample can't be topped up; sample the run again",
				code="unsupported.agent_sampling")  # fmt: skip
		# the first sample's indices are offered again first, and refused by content
		offered = tuple(draw_indices(population, DEFAULT_SAMPLE) or ())
	else:
		k, skip_keys, offered = DEFAULT_SAMPLE, set(), ()
	_emit(ctx, "probing", rec, rev, message=f"sampling {k} tasks through the agent proxy"
		+ (f", skipping the {len(first.tasks)} already sampled" if top_up else ""))  # fmt: skip
	sample = deps.agent(fetched, sm, k=k, skip_keys=skip_keys, offered_first=offered,
		run_id="s2" if top_up else "s1", should_stop=getattr(ctx, "should_stop", None))  # fmt: skip
	ctx.check()
	if not sample.tasks:
		why = ("the top-up ran no new tasks: the script may only have the tasks already sampled, or its tasks all open with the same first call"
			if top_up else "the script made no model calls through the proxy: is its client reading OPENAI_BASE_URL?")  # fmt: skip
		raise PermanentError(why, code="probe.no_agent_calls")
	if top_up:
		sample = combine_samples(first, sample)
	out = rec.model_copy(update={"agent_sample": sample})
	from .measured_rules import agent_findings

	fresh = agent_findings(out)  # measured: replaces the parent's, keeps what was read from the code
	keep = [f for f in out.findings if not f.id.startswith("measured.agent.")]
	out = out.model_copy(update={"findings": [*keep, *fresh]})
	_emit(ctx, "probing", out, deps.records.put(out), message=f"{len(sample.tasks)} tasks sampled")
	return out


def _followup_plan(rec: FlightRecord, mode: str, knobs: dict, plan_knobs: dict, base_knobs: dict, *,
	patched: bool = False):  # fmt: skip
	"""probe: one profiled point at the setting. verify: the setting, and the current settings
	beside it when they differ, both for VERIFY_STEPS at one fixed seed."""
	if mode == "probe":
		return [point_probe(rec, next_probe_id(rec), knobs, role="point")]
	vid = next_probe_id(rec, "v")
	steps = {"warmup": record_jobs.VERIFY_WARMUP, "timed": record_jobs.VERIFY_STEPS, "cadence": False}
	plan = [point_probe(rec, vid, knobs, role="verify", **steps)]
	differs = {k: v for k, v in plan_knobs.items() if k != "gpu_type" and base_knobs.get(k) != v}
	if differs or patched:  # a patch is a difference too: its verify needs the base beside it
		base = {k: base_knobs[k] for k in knobs if k in base_knobs}
		plan.append(point_probe(rec, f"{vid}-base", base, role="verify_base", **steps))
		plan.append(point_probe(rec, f"{vid}-seed", base, role="verify_seed", **steps))
	for p in plan:
		p.seed = record_jobs.VERIFY_SEED_2 if p.role == "verify_seed" else record_jobs.VERIFY_SEED
		p.seed_force = True
	return plan


def _parent_gpu(rec: FlightRecord) -> str | None:
	"""The GPU the parent measured on (or chose), so a follow-up probe compares like with like."""
	from preflight.probes.gpu_choice import canonical

	name = rec.env.gpu.name if rec.env is not None and rec.env.gpu is not None else None
	if not name:
		try:
			name = rec.knob("gpu_type").current
		except KeyError:
			return None
	name = canonical(str(name))
	return name if name in _offered() else None


def _probe_points(
	plan: list[PlannedProbe],
	rec: FlightRecord,
	rev: int,
	fetched: Fetched,
	sm: ScriptMap,
	deps: Deps,
	ctx,
	budget: Budget,
	*,
	gpu: str,
	allowed: list[str] | None,
	need: float,
	pinned: bool = False,
	patched_for: dict | None = None,
) -> tuple[list[Probe], list[ProbeResult], str, float]:
	"""Run the planned points in order; returns their probes, fresh results, the GPU they settled on
	and what discarded attempts cost."""
	probes: list[Probe] = []
	results: list[ProbeResult] = []
	discarded_usd = 0.0
	for i, p in enumerate(plan):
		ctx.check()
		_emit(ctx, "probing", rec, rev, probe_id=p.probe_id, progress=round(i / len(plan), 3))
		if any(q.error_code == "missing_package.install" for q in probes):
			# the same requirements fail the same way at every point: don't pay for it again
			probes.append(Probe(id=p.probe_id, knobs=p.knobs, status="error", steps=0,
				error="skipped: requirements.txt didn't install (see the first probe)",
				error_code="missing_package.install"))  # fmt: skip
			continue
		est_s = max((r.probe.duration_s or 0.0 for r in results), default=60.0)
		over = deps.overhead_usd_per_h
		if probes and not budget.allows((_price(gpu) + over) * est_s / 3600, est_s):
			why = (
				f"skipped: the job's probe budget (${budget.usd:.2f}, {budget.wall_s / 60:.0f} min) is spent"
			)
			probes.append(
				Probe(id=p.probe_id, knobs=p.knobs, status="error", steps=0, error=why, error_code="budget")
			)
			continue
		try:
			use_f, use_sm = (patched_for or {}).get(p.probe_id, (fetched, sm))
			spec = _spec(use_f, use_sm, p, deps)
		except OverrideError as exc:
			probes.append(_skipped(p, f"this point can't be set in the code: {exc}"))
			continue
		attempts: list = []

		def run(g, spec=spec, attempts=attempts, over=over):
			result, cached = _run_point(spec, g, deps, ctx)
			out = result if result is not None else _CachedResult(cached)
			if result is not None:
				# every attempt counts, kept or not, with the sandbox's CPU and memory beside the GPU
				budget.spent_usd += result.probe.cost.usd + (result.probe.duration_s or 0) * over / 3600
				budget.gpu_usd += result.probe.cost.usd
				budget.overhead_usd += (result.probe.duration_s or 0) * over / 3600
			attempts.append(out)
			return out

		def may_step_up(
			g, est_s=est_s, over=over
		):  # another GPU only while the job is live and the budget holds it
			return (
				not pinned
				and not ctx.should_stop()
				and budget.allows((_price(g) + over) * est_s / 3600, est_s)
			)

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
				out = _marked_eval_off(again, rec)
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
	return probes, results, gpu, discarded_usd


def _marked_eval_off(out, rec: FlightRecord):
	"""The eval-off re-run, saying so (C, Oct 28): extras.eval_off, so a held-out check knows its
	peak is training's alone. Its knobs stay the record's (B's engine abstains on an eval batch of
	0, and 0 is outside the knob's domain). Marked in place: the attempt's identity is how its cost
	is counted as kept, not discarded."""
	out.probe = out.probe.model_copy(update={"extras": {**out.probe.extras, "eval_off": 1.0}})
	return out


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
	overhead_usd: float = 0.0,
	discarded_usd: float = 0.0,
	billed_usd: float | None = None,
	new_ids: set[str] | None = None,
) -> FlightRecord:
	"""probes: every probe the record will carry; new_ids: the ones this job ran (default all).
	Costs add this job's spend to what the record had already cost (a derived record keeps its
	parent's), so a record always says what making it cost in total."""
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
	ran = [p for p in probes if new_ids is None or p.id in new_ids]
	kept = round(sum(p.cost.usd for p in ran if p.status != "cached"), 6)
	# what the job really cost: every attempt (a probe that ran out and was re-run on a bigger GPU
	# still ran), and the sandbox's whole lifetime when the runner knows it (start-up, idle)
	job_usd = max(kept, spent_usd or 0.0, billed_usd or 0.0)
	prior = rec.costs.breakdown
	breakdown = {**prior, "probes": round(prior.get("probes", 0.0) + kept, 6)}
	if discarded_usd:
		breakdown["discarded_attempts"] = round(prior.get("discarded_attempts", 0.0) + discarded_usd, 6)
	if billed_usd and billed_usd > kept + discarded_usd:
		over = billed_usd - kept - discarded_usd
		breakdown["sandbox_overhead"] = round(prior.get("sandbox_overhead", 0.0) + over, 6)
	gpu_usd = round(rec.costs.gpu_usd + job_usd, 6)
	if overhead_usd:
		# an estimate of the sandbox's CPU and memory (Deps.overhead_usd_per_h), kept apart from the
		# GPU figure, which is measured probe time at list price; Modal's bill replaces both
		breakdown["sandbox_cpu_memory_estimate"] = round(
			prior.get("sandbox_cpu_memory_estimate", 0.0) + overhead_usd, 6
		)
	# every model call the record carries (a derived record keeps its parent's), priced per call
	llm_usd = round(sum(c.cost_usd for c in rec.provenance), 6)
	if llm_usd:
		breakdown["model_calls"] = llm_usd
	costs = rec.costs.model_copy(update={"gpu_usd": gpu_usd, "llm_usd": llm_usd, "breakdown": breakdown})
	update["costs"] = costs
	return FlightRecord.model_validate({**rec.model_dump(), **{k: v for k, v in update.items()}})


# derived: the fill took 0.09 s on the first L4 record (Oct 28, 5 estimates and 80 grids); 300x
# that is a hang, and the measured record is stored without predictions instead of holding the
# lease until WORKER_TIMEOUT_S and being retried into the same hang
PREDICT_TIMEOUT_S = 30.0


def with_predictions(
	rec: FlightRecord, ctx, *, enabled: bool | None = None, timeout_s: float = PREDICT_TIMEOUT_S
) -> tuple[FlightRecord, bool]:
	"""The measured record with B's engine's predictions filled (engine.api.fill_predictions), when
	the engine.memory flag is on, and whether the engine failed. Any engine failure or a fill that
	runs past timeout_s keeps the measured record as it was and says so in a warn event: a
	prediction never costs the visitor their measurements."""
	from .flags import flag

	if not (flag("engine.memory") if enabled is None else enabled):
		return rec, False
	box: dict = {}

	def fill() -> None:
		try:
			from preflight.engine.api import fill_predictions

			# the engine returns a copy without re-validating: store-and-load it exactly as C's API
			# will (nulls dropped, then parsed), and fail here rather than in the reader;
			# dump_record also enforces the 1 MB gzipped budget
			box["rec"] = load_record(dump_record(fill_predictions(rec)))
		except BaseException as exc:  # SystemExit from a library too: reported, never a fake timeout
			box["exc"] = exc

	t0 = time.monotonic()
	# a daemon thread: a fill stuck in a loop can't be killed, but it can't hold the job either
	th = threading.Thread(target=fill, name=f"predict-{rec.record_id}", daemon=True)
	th.start()
	th.join(timeout_s)
	took = time.monotonic() - t0
	if "rec" in box:
		log.info("job %s: predictions filled in %.2f s", rec.record_id, took)
		return box["rec"], False
	exc = box.get("exc") or TimeoutError(f"no predictions after {timeout_s:g} s")
	log.error("job %s: the engine couldn't fill predictions after %.2f s; keeping the measured record",
		rec.record_id, took, exc_info=exc)  # fmt: skip
	_emit(ctx, "predicting", rec, None, level="warn",
		message=f"predictions unavailable ({type(exc).__name__}); the measurements stand")  # fmt: skip
	return rec, True


def _has_predictions(rec: FlightRecord) -> bool:
	return bool(rec.predictions.grids) or any(e.point is not None for e in rec.predictions.estimates.values())


def run_job(lease, ctx: JobContext, *, deps: Deps) -> str:
	"""worker.JobFn: run the pipeline for a leased job, fill B's predictions, store the final record,
	return its id."""
	rec = run_pipeline(lease.request, ctx, deps, ids=(lease.record_id, lease.share_token))
	_emit(ctx, "predicting", rec, None, message="measured; predicting")
	rec, failed = with_predictions(rec, ctx)
	ctx.confirm()  # asks the store: a job cancelled or reaped during the fill stores nothing more
	rev = deps.records.put(rec)
	if failed:
		done = "measured; predictions unavailable"
	elif _has_predictions(rec):
		done = "predicted"
	else:
		done = "measured; no predictions yet"
	_emit(ctx, "predicting", rec, rev, message=done)
	return rec.record_id
