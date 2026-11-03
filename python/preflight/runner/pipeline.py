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
GPUs: never above PREFLIGHT_GPU_CEILING (the public demo: L40S, Nov 1), and a
step up after an OOM only while the job's dollar budget holds it.
"""

import contextlib
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
from .sources import Fetched, extra_lines, fetch
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
	# time spent waiting for a GPU to schedule: not the job's probe time (Chiya, Nov 1: the wall
	# counts from when a probe starts on its GPU; the wait has its own cap, in the runner)
	waited: Callable[[], float] = field(default=lambda: 0.0)

	def left_s(self) -> float:
		return self.wall_s - (time.monotonic() - self.started - self.waited())

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
		**({"install_repo": True} if spec.install_repo else {}),  # absent: keys stay as they were
		**({"extra_packages": list(spec.extra_packages)} if spec.extra_packages else {}),
		**({"install_pyg": True} if spec.install_pyg else {}),  # another index can install another build
		**({"install_uv": True} if spec.install_uv else {}),  # another resolver can pick other versions
		**({"install_exec": True} if spec.install_exec else {}),  # a list read from the build installs more
		**({"torch_stop": True} if spec.torch_stop else {}),  # a stop instead of a run
		**({"dataset_cache": True} if spec.dataset_cache else {}),  # no download in the measured time
		**({"step_clock": True} if spec.step_clock else {}),  # a probe that reports more
		**({"install_sdists": list(spec.install_sdists)} if spec.install_sdists else {}),
		**({"eval_units": spec.eval_units} if spec.eval_units else {}),
		"gpu": gpu,
		"image": image if spec.image == "probe" else f"{image}+{spec.image}" + _image_pins(spec.image),
	}  # fmt: skip
	if (
		spec.generate_sample
	):  # a serving probe's sample size decides what it measured; absent, keys stay as they were
		body["generate_sample"] = spec.generate_sample
	if spec.generate_calls:  # an eval probe's measured calls, likewise
		body["generate_calls"] = spec.generate_calls
	if spec.lr_range:  # the range test's ramp (D, Nov 2), likewise
		body["lr_range"] = spec.lr_range
	h.update(json.dumps(body, sort_keys=True).encode())
	return "sha256:" + h.hexdigest()


@dataclass
class Deps:
	records: RecordWriter
	runner: ProbeRunner = local_runner
	spark: object | None = None  # harness.spark.ModelClient
	triage: object | None = (
		None  # Spark Open on Baseten: triage (harness/triage.py) and the diagnosis check (diagnosis_check.py)
	)
	glimmer: object | None = None  # Muse Glimmer on B's host: a private job's triage (0.6.2)
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


# guess: a generation-bound probe (an eval's or a GRPO step's long completions, a serving batch)
# passes few of its own boundaries per minute; the Nov 1 eval sweep probe ran out at 360 s
GENERATION_LENSES = ("eval", "rl", "serving")
# a serving probe's one measured generate call: more prompts than a 24 GB card's KV cache holds at
# once (about 16 to 21 at 4,096 tokens for a 7B model), so preemption shows, and few enough to finish
# well inside the probe's time limit on an L4 (32 x 4,096 tokens) [guess: the next probe replaces it]
SERVING_PROBE_PROMPTS = 32
# an eval probe's measured generate calls: two batches give a per-batch time, the truncation share
# and the bytes a batch writes; the Nov 1 eval smoke ran batch after batch to its 600 s timeout and
# kept nothing [guess: two]
EVAL_PROBE_CALLS = 2
# probes.eval_units: an eval probe's scoring batches (a forward under no_grad, no generate): a few give a per-batch
# time and the peak; lm_eval's loglikelihood tasks score thousands of them [guess: eight]
EVAL_FORWARD_UNITS = 8
GENERATION_PROBE_TIMEOUT_S = 10 * 60


def _probe_timeout(sm: ScriptMap, base: float) -> float:
	"""The probe's time limit: longer for generation-bound lenses, within the job's probe wall."""
	if sm.lens in GENERATION_LENSES:
		return max(base, min(GENERATION_PROBE_TIMEOUT_S, JOB_PROBE_WALL_S - 60))
	return base


_VLLM_IMPORT = re.compile(r"^\s*(?:from\s+vllm\b|import\s+vllm\b)", re.M)


def _uses_vllm(sm: ScriptMap, files: dict[str, str]) -> bool:
	"""Serving, or any script that imports vLLM (an eval sweep on vLLM is lens eval): the probe
	image has no vLLM."""
	if sm.lens == "serving":
		return True
	# only an eval may need it: a GRPO script with an optional `import vllm` (use_vllm) needs trl
	return sm.lens == "eval" and any(_VLLM_IMPORT.search(t) for n, t in files.items() if n.endswith(".py"))


_TORCHVISION = re.compile(r"^\s*(?:from|import)\s+torchvision\b", re.M)


def _uses_torchvision(files: dict[str, str]) -> bool:
	return any(_TORCHVISION.search(t) for n, t in files.items() if n.endswith(".py"))


def _image_pins(image: str) -> str:
	"""A sandbox image's own pins in its probes' cache key, for the images image_tag doesn't hash: torch27's."""
	if image != "torch27":
		return ""
	from .images import PROBE_NEW_PACKAGES

	return "@" + hashlib.sha256("\n".join(PROBE_NEW_PACKAGES).encode()).hexdigest()[:12]


def _image(sm: ScriptMap, files: dict[str, str]) -> str:
	"""The sandbox image: "serving" for vLLM (its own image); with probes.torch_new, "torch27" when the code's
	own torch requirement rules out the probe image's 2.5.1 and admits 2.7.1 (LitGPT's torch>=2.7, Nov 2);
	else "probe"."""
	if _uses_vllm(sm, files):
		return "serving"
	if _flag("probes.torch_new"):
		from preflight.recorder.requirements import IMAGE_TORCH, NEW_IMAGE_TORCH, torch_asks, torch_fits

		asks = torch_asks(files)
		if asks and not torch_fits(asks, IMAGE_TORCH) and torch_fits(asks, NEW_IMAGE_TORCH):
			return "torch27"
	return "probe"


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
		timeout_s=_probe_timeout(sm, deps.probe_timeout_s),
		seed=p.seed,
		seed_force=p.seed_force,
		install_requirements=deps.install_requirements,
		install_repo=deps.install_requirements and _repo_install_on(),
		extra_packages=extra_lines(fetched.source) if deps.install_requirements else (),
		install_uv=deps.install_requirements and _flag("probes.uv_install"),
		install_pyg=deps.install_requirements and _flag("probes.extra_hosts"),
		install_exec=deps.install_requirements and _repo_install_on() and _flag("probes.repo_install_exec"),
		torch_stop=deps.install_requirements and _flag("probes.torch_new"),
		# only code that imports torchvision: the flag re-keys no other job's probes (review, Nov 3)
		dataset_cache=_flag("probes.dataset_cache") and _uses_torchvision(fetched.files),
		step_clock=_flag("probes.step_clock"),
		install_sdists=_command_sdists(fetched) if deps.install_requirements else (),
		eval_units=EVAL_FORWARD_UNITS if sm.lens == "eval" and _flag("probes.eval_units") else 0,
		image=_image(sm, fetched.files),
		generate_sample=SERVING_PROBE_PROMPTS if sm.lens == "serving" else 0,
		generate_calls=EVAL_PROBE_CALLS if sm.lens == "eval" else 0,
	)


def _import_knobs(knobs: dict) -> tuple:
	"""The knobs that can change what a script imports (quantization, attention, optimizer, precision:
	any value that isn't a number); points that differ only in sizes import the same packages."""
	return tuple(
		sorted(
			(k, str(v))
			for k, v in (knobs or {}).items()
			if isinstance(v, bool) or not isinstance(v, int | float)
		)
	)


def _command_sdists(fetched: Fetched) -> tuple[str, ...]:
	"""A check-in by command's own source-only dependencies (sources.COMMANDS), else none."""
	from .sources import COMMANDS

	src = fetched.source
	if (
		_flag("probes.repo_install_exec")
		and "lm_eval/__init__.py" in fetched.files
		and fetched.entrypoint.startswith("lm_eval/")
	):
		# lm-evaluation-harness checked in as its repo (unseen 5): the same source-only dependencies its command
		# path builds (sqlitedict has no wheel), behind the flag that already runs a repo's build code
		return tuple(COMMANDS["lm_eval"].get("sdists", ()))
	if src is None or src.kind != "command" or not src.args:
		return ()
	return tuple((COMMANDS.get(src.args[0]) or {}).get("sdists", ()))


def _command_lens(sm: ScriptMap, fetched: Fetched) -> ScriptMap:
	"""A check-in by command is its tool's lens (sources.COMMANDS): the map reads only the wrapper."""
	return sm.model_copy(update={"lens": fetched.lens}) if fetched.lens else sm


def _flag(name: str) -> bool:
	from .flags import flag

	return flag(name)


def _repo_install_on() -> bool:
	"""probes.repo_install (off): a repo with no requirements.txt gets what it declares and the known
	packages it imports (Nov 2: the unseen scripts stopped at their imports)."""
	from .flags import flag

	return flag("probes.repo_install")


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
	ran = getattr(deps.runner, "last_ran", None) or gpu
	if ran != gpu:  # measured on a fallback GPU: not this GPU's answer, so not cached under it
		result.probe = result.probe.model_copy(update={"cache_key": None})
		return result, None
	result.probe = result.probe.model_copy(update={"cache_key": key})
	if deps.cache is not None and result.probe.status == "ok":
		deps.cache.put(key, json.loads(result.probe.model_dump_json(exclude_none=True)))
	return result, None


# Chiya, Nov 1: an L4 that won't schedule in 2 minutes gives way to an A10, then an L40S
GPU_FALLBACK = {"L4": ("A10", "L40S")}
# during judging a job may start on the A10 (judging.py); only its first point may fall back to the
# L4, before any point has run: after that the job's points stay on the GPU they started on


def _fallbacks(gpu: str, allowed: list[str] | None, pinned: bool) -> tuple[str, ...]:
	"""What the runner may use instead of gpu: none for a pinned GPU (the guide asked for that one),
	only GPUs the visitor allows, and none past the probe ceiling."""
	if pinned:
		return ()
	cap = gpu_ceiling()
	return tuple(
		g for g in GPU_FALLBACK.get(gpu, ()) if (allowed is None or g in allowed) and under_ceiling(g, cap)
	)


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
	if req.extra_packages:  # 0.6.3: checked against the list by fetch, and kept on the record's source
		# added to the record's own (a probe from a record that already installs some), in order, once each
		had = source.get("extra_packages") or []
		source = {**source, "extra_packages": list(dict.fromkeys([*had, *req.extra_packages]))}
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
		if fetched.source.extra_packages != rec.job.source.extra_packages:  # this job added packages
			src = rec.job.source.model_copy(update={"extra_packages": fetched.source.extra_packages})
			rec = rec.model_copy(update={"job": rec.job.model_copy(update={"source": src})})
		if rec.job.script_map:
			sm = rec.job.script_map
		else:
			mapped, fetched = _map_or_skip(fetched)
			sm = values_from_args(fetched.files, _command_lens(mapped, fetched), fetched.args)
		rev = deps.records.put(rec)
		_emit(ctx, "skeleton", rec, rev, message=f"continuing from record {origin.parent.share_token}")
	else:
		mapped, fetched = _map_or_skip(fetched)
		sm = values_from_args(
			fetched.files,
			_command_lens(mapped, fetched),
			fetched.args,
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
		mappable = not followup and not fetched.map_skipped  # a skipped map isn't refined (probe_without_map)
		if deps.spark is not None and mappable and flag("harness.spark_script_map"):
			ctx.check()
			spark_sm, calls = map_script(fetched.files, entrypoint=fetched.entrypoint, client=deps.spark)
			if calls:
				sm = values_from_args(fetched.files, _command_lens(spark_sm, fetched), fetched.args)
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
	from .judging import judging_gpu

	gpu = _requested_gpu(request, allowed) or (
		_parent_gpu(rec) if followup and origin.parent is not None else None
	)  # fmt: skip
	judged = False
	if gpu is None:
		chosen = choose_probe_gpu(need, allowed)
		gpu = judging_gpu(chosen, allowed, time.time())
		judged = gpu != chosen
	cap = gpu_ceiling()
	if cap and allowed and not any(under_ceiling(g, cap) for g in allowed) and gpu == cap:
		# every GPU the visitor allows is above the cap: probed on the cap, said as for a request (review)
		_emit(ctx, "harnessing", rec, rev, level="warn",
			message=f"probes here run on GPUs up to {cap}; {', '.join(allowed)} numbers come from the cross-GPU model")  # fmt: skip
	if not under_ceiling(gpu, cap):
		_emit(ctx, "harnessing", rec, rev, level="warn",
			message=f"probes here run on GPUs up to {cap}; {gpu} numbers come from the cross-GPU model")  # fmt: skip
		gpu = cap
	budget = deps.budget()
	with contextlib.suppress(AttributeError, TypeError):
		ctx.budget = budget  # the worker reports what the job spent to the day's GPU cap (queue.py)
	budget.waited = lambda: float(getattr(deps.runner, "queued_s", 0.0) or 0.0)
	from .queue import GPU_WAIT_PREFIX

	if hasattr(deps.runner, "notify"):  # a probe waiting for a GPU says so, apart from running (D, Nov 1)
		deps.runner.notify = lambda kind, g, waited: _emit(ctx, "probing", level="info", message=(
			f"{GPU_WAIT_PREFIX}{g} GPU ({waited / 60:.0f} min so far)" if kind == "waiting" and waited >= 60
			else f"{GPU_WAIT_PREFIX}{g} GPU" if kind == "waiting"
			else f"{g} GPU ready after {waited / 60:.0f} min; probe running" if waited >= 60
			else f"{g} GPU ready; probe running"))  # fmt: skip
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
		judged=judged,
		patched_for=patched_for,
	)
	if req.mode == "plan" and not patched_for:
		lr_probe = _lr_range_point(plan, probes, fetched, sm, deps, ctx, budget, gpu=gpu)
		if lr_probe is not None:
			probes.append(lr_probe)  # in the record only: not a result, so never triaged or searched
	billed = deps.runner.billed_usd(_price) if hasattr(deps.runner, "billed_usd") else None
	# review, Nov 2: a probe-more from the record page sends no `private`, so a record a private job touched keeps
	# its new logs off Baseten and Meta too
	held = req.private or _private_history(rec)
	rec, triaged = _triage(rec, results, deps, private=held)
	rec, grounded = _search(rec, results, deps, private=held, verify=req.mode == "verify", ctx=ctx)
	triaged = [*triaged, *grounded]
	kept = rec.probes
	# the GPU the points ran on (a fallback's, if the runner used one), else the one asked for
	ran = next((g for r in results for g in [_measured_gpu(r)] if g), gpu)
	if followup and origin.parent is not None and ran != _parent_gpu(origin.parent):
		# measured on another GPU: the record's env is this one, so the parent's probes don't
		# belong, nor what was fitted or found from them; the script's own findings stay
		kept = []
		dropped = {p.id for p in rec.probes}
		findings = [f for f in rec.findings if not any(e.probe_id in dropped for e in f.evidence)]
		rec = rec.model_copy(update={"predictions": Predictions(), "models": [], "findings": findings})
	# a probe that didn't finish names its step and the way past it by hand (item 43)
	if (
		_flag("runner.free_resend")
		and probes
		and all(q.steps == 0 and (q.error_code or "").startswith("missing_package") for q in probes)
	):
		# every probe stopped at an import: a resend with the packages is free today (runner.free_resend)
		_emit(
			ctx,
			"probing",
			rec,
			rev,
			import_stop=True,
			message="every probe stopped at an import before its first step",
		)
	probes = [_with_step(q, fetched.entrypoint) for q in probes]
	if patched_for:
		# measured on patched code, not the record's: marked, so B's fits and the plan leave them out
		probes = [q.model_copy(update={"extras": {**q.extras, "patched": 1.0}}) if q.id in patched_for else q
			for q in probes]  # fmt: skip
	if held:
		# B, Nov 2 (A's review): a private job's probes and findings carry the mark on the record, so no later job
		# on it (a probe-more, a verify) sends their numbers or text to Meta or Baseten
		from preflight.schema import Evidence

		probes = [q.model_copy(update={"extras": {**q.extras, "private_job": 1.0}}) for q in probes]
		triaged = [f.model_copy(update={"evidence": [*f.evidence, Evidence(metric="private_job", value=1.0)]})
			for f in triaged]  # fmt: skip
	out = _merge(rec, [*kept, *probes], results, gpu, spent_usd=budget.gpu_usd, overhead_usd=budget.overhead_usd,
		discarded_usd=discarded_usd, billed_usd=billed, new_ids={p.id for p in probes})  # fmt: skip
	from .measured_rules import hygiene_findings

	# run hygiene, from the code the probes ran (a probe job's patch included, so a fix can clear
	# its own finding; a verify job's base is the record's own code) and the probes so far
	code = fixed.files if fixed is not None and req.mode == "probe" else fetched.files
	out = _with_dataset_size(out, fetched.files)  # the record's own code: a patch may change the size
	keep = [f for f in out.findings if not f.id.startswith(("hygiene.", "measured.rl."))]
	keep = [f for f in keep if f.id not in {g.id for g in triaged}]
	keep = [
		f for f in keep if not f.id.startswith(("source.partial", "source.map_skipped"))
	]  # said afresh below
	out = out.model_copy(
		update={"findings": [*keep, *hygiene_findings(out, code), *triaged, *_reading_findings(fetched)]}
	)
	# a verify job's record answers one question; a private job's numbers and plots never go to Meta
	# or Baseten for a diagnosis (private mode, contract 0.6.2)
	touched = req.private or _private_history(out)
	if req.mode != "verify" and not touched:
		out = _diagnose(out, deps)
	out = _claim_check(out, deps, private=touched)
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
	from preflight.probes.plan import pull_verify_eval

	# A patch (a fix may touch the eval) or an eval knob changed: every side ends with one eval, so
	# an eval fix reads before and after. A training knob's verify stays as it was: an eval running
	# out at both settings would be the record's own problem, not the change's (review, Nov 2).
	eval_change = patched or any(k in EVAL_KNOBS for k in differs)
	for p in plan:
		p.seed = record_jobs.VERIFY_SEED_2 if p.role == "verify_seed" else record_jobs.VERIFY_SEED
		p.seed_force = True
		if eval_change:
			pull_verify_eval(rec, p)
	return plan


EVAL_KNOBS = ("eval_batch_size", "eval_no_grad", "eval_every")


# N_TRAIN_NODES, NUM_TRAIN_SAMPLES, N_TRAIN: counts of training rows only, never NUM_TRAIN_EPOCHS or
# N_TRAIN_STEPS (a run length, not a dataset size)
_TRAIN_COUNT = re.compile(r"^N(?:UM)?_TRAIN(?:_(?:NODES|SAMPLES|EXAMPLES|ROWS|PAIRS|IMAGES|EDGES|DOCS))?$")


def _train_count_setting(files: dict[str, str] | None) -> int | None:
	"""A training-set size the script states as a plain module-level setting (zoo/gnn_pyg's
	N_TRAIN_NODES, D, Nov 1): a script that samples its graph by hand has no loader to measure."""
	import ast

	for name, text in sorted((files or {}).items()):
		if not name.endswith(".py"):
			continue
		try:
			tree = ast.parse(text)
		except (SyntaxError, ValueError):
			continue
		for node in tree.body:
			if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
				and _TRAIN_COUNT.match(node.targets[0].id) and isinstance(node.value, ast.Constant)
				and type(node.value.value) is int and node.value.value > 0):  # fmt: skip
				return node.value.value
	return None


def _with_dataset_size(rec: FlightRecord, files: dict[str, str] | None = None) -> FlightRecord:
	"""The dataset size the static reader couldn't see, from the unpatched probes: the rows the
	training loader's sampler draws per epoch (extras.train_dataset_len), else the token-row scan's total. B's engine divides
	it by the batch for the run's length, so time.total and cost.total don't abstain (D, Nov 1)."""
	sm = rec.job.script_map
	if sm is None or sm.data is None or sm.data.n_examples:
		return rec
	# the script's own setting first: it says what one epoch trains on, where a loader's dataset
	# may be the whole graph or table
	# the loader's size doesn't depend on how a probe ended: a cache hit (a second visitor on the
	# same script, C's live LoRA record, Nov 1) or one that ran out at eval saw the same dataset. An
	# oom probe counts only past its first step: before the first backward the latest loader may be
	# an eval one (a baseline evaluate, eval_on_start; review, Nov 1). ok first, then cached, then oom
	rank = {"ok": 0, "cached": 1, "oom": 2}
	ok = sorted((p for p in rec.probes if p.status in rank and not p.extras.get("patched")  # a patch may subsample
		and (p.status != "oom" or (p.steps or 0) >= 1)), key=lambda p: rank[p.status])  # fmt: skip
	n = _train_count_setting(files) or next((p.extras["train_dataset_len"] for p in ok if p.extras.get("train_dataset_len")), None) or next(
		(p.extras["dataset_rows_total"] for p in ok if p.extras.get("dataset_rows_total")), None)  # fmt: skip
	if not n:
		return rec
	data = sm.data.model_copy(update={"n_examples": int(n)})
	job = rec.job.model_copy(update={"script_map": sm.model_copy(update={"data": data})})
	return rec.model_copy(update={"job": job})


def _mb(n: int) -> str:
	return f"{n / 2**20:.2f} MB" if n < 10 * 2**20 else f"{n / 2**20:.0f} MB"


def _n_files(n: int, word: str) -> str:
	if n == 0:
		return f"none of the {word}s"
	return f"the 1 {word}" if n == 1 else f"the {n:,} {word}s"


def _map_or_skip(fetched: Fetched) -> tuple[ScriptMap, Fetched]:
	"""The static map. With runner.probe_without_map (off; the PM, Nov 3), when the whole source doesn't map
	(over 2 MB of Python, a file the reader can't parse, the reader's own failure), the entry point alone is
	mapped and the job probes on as usual, the plain recorder on every file, saying the map was skipped and
	why (_reading_findings), not "syntax error". When the entry point itself doesn't map, that error stands."""
	try:
		return map_static(fetched.files, entrypoint=fetched.entrypoint), fetched
	except Exception as exc:
		if isinstance(exc, InvalidSubmission) or not _flag("runner.probe_without_map"):
			raise
		entry = fetched.files.get(fetched.entrypoint)
		if entry is None:
			raise
		try:
			sm = map_static({fetched.entrypoint: entry}, entrypoint=fetched.entrypoint)
		except Exception:
			raise exc from None
		code = getattr(exc, "code", None)
		why = (
			str(exc) if code == "map.too_large"
			else f"the static reader couldn't read it ({str(exc)[:240] or type(exc).__name__})"
		)  # fmt: skip
		return sm, replace(fetched, map_skipped=why)


def _reading_findings(fetched: Fetched) -> list:
	"""Say so when Preflight read part of the source (D, Nov 2): a repository too big to read whole
	is read as its entry point and the files it imports (runner/sources.py), and over 2 MB of Python
	is mapped the same way (harness/static.py mapped_files). Its knobs and checks cover what was
	read; a file reached only by a string import isn't."""
	from preflight.harness.static import mapped_files
	from preflight.schema import Evidence, Finding

	out = []
	if fetched.map_skipped:
		out.append(Finding(id="source.map_skipped", severity="info", code="source.partial", source="rule",
			title="Probed without a map of the whole source",
			detail=(f"Preflight didn't map the whole source: {fetched.map_skipped}. So the map shows only what "
				f"{fetched.entrypoint} itself sets, and the job's kind, its model and the GPU it needs were read from "
				"that file alone too, so they may be off. The probes ran the code as it is with the plain recorder: "
				"what they measured stands, but knobs set in other files aren't on the map."),
			evidence=[Evidence(metric="source.files", value=float(len(fetched.files)))]))  # fmt: skip
	p = fetched.partial
	if p:
		why = (
			"when the import lookups ran out (200,000)" if p.get("cut") == "lookups" else "for the size limit"
		)
		left = (
			f"; at least {p['skipped']:,} more imported files were left out {why}" if p.get("skipped") else ""
		)
		out.append(Finding(id="source.partial.read", severity="info", code="source.partial", source="rule",
			title="Read the entry point and what it imports, as asked" if p.get("asked")
				else "Read the entry point and what it imports, not the whole repository",
			detail=(f"{_why_partial(p)}, so it read {fetched.entrypoint} and {_n_files(p['imported'], 'Python file')} it imports "
				f"({_mb(p['bytes'])} in all){left}. The map and the findings cover what was read, and the probes run "
				"on it: a module left out, or loaded by name at run time, can stop a probe with an import error."),
			evidence=[Evidence(metric="source.files_read", value=float(p["files"])),
				Evidence(metric="source.repo_files", value=float(p["repo_files"]))]))  # fmt: skip
	try:  # with the map skipped, its finding above says it: no partial-map finding beside it
		_, m = (None, None) if fetched.map_skipped else mapped_files(fetched.files, fetched.entrypoint)
	except ValueError:
		m = None
	if m:
		out.append(Finding(id="source.partial.map", severity="info", code="source.partial", source="rule",
			title="Mapped the entry point and what it imports, not every file",
			detail=(f"The source has {_mb(m['python_bytes'])} of Python in {m['python_files']:,} files, more than "
				f"Preflight maps (2 MB), so the map reads {fetched.entrypoint} and {_n_files(m['imported'], 'file')} "
				f"it imports ({_mb(m['bytes'])}). Settings in other files aren't on the map."),
			evidence=[Evidence(metric="map.files_read", value=float(m["files"]))]))  # fmt: skip
	return out


def _why_partial(p: dict) -> str:
	size = f"{p['repo_files']:,} text files ({_mb(p['repo_bytes'])})"
	if p.get("asked"):  # Source.scope "entrypoint"
		return f"You asked Preflight to start from the entry point (the source has {size})"
	return f"The repository has {size}, more than Preflight reads whole"


def _triage_client(deps: Deps, private: bool):
	"""Who triages: Glimmer on Preflight's own host for a private job with harness.glimmer_private on
	(contract 0.6.2), else Spark Open on Baseten. A private job with no Glimmer client is never sent
	to Baseten instead: it isn't triaged."""
	from .flags import flag

	if private:  # never Baseten: Glimmer with its flag on, else no triage (review, Nov 2)
		return deps.glimmer if flag("harness.glimmer_private") else None
	return deps.triage


def _triage(
	rec: FlightRecord, results: list, deps: Deps, *, private: bool = False
) -> tuple[FlightRecord, list]:
	"""Spark Open reads each fresh probe's log (plan item 29): one call per probe, every call in the
	record's provenance (so its cost is in the record's), a finding for a probe that didn't finish.
	Off unless harness.spark_open_triage is on: the log may hold the visitor's script output."""
	from .flags import flag

	client = _triage_client(deps, private)
	if client is None or not flag("harness.spark_open_triage"):
		return rec, []
	from preflight.harness.triage import triage_probe

	findings, calls = [], []
	nodes = {n.id for n in rec.graph.nodes} if rec.graph else set()
	for r in results:
		if not isinstance(r, ProbeResult):
			continue
		f, cs = triage_probe(client, r.probe, r.log_tail, node_id=_tested_node(r.probe, nodes))
		calls += cs
		if f is not None:
			findings.append(f)
	return rec.model_copy(update={"provenance": [*rec.provenance, *calls]}), findings


LR_RANGE_LENSES = ("finetune", "pretrain")  # Chiya's list (Nov 2): LoRA and nanoGPT are where it's checked


def _lr_range_point(plan, probes, fetched, sm, deps, ctx, budget, *, gpu: str):
	"""The learning-rate range test (D, Nov 2; probes/lr_range.py): one more point after the plan,
	at the current settings with the eval off, about 100 steps on an exponential ramp. Only with
	probes.lr_range on, for a training lens, after the current-settings point ran, and inside the
	job's budget. Its probe `<p1>-lr` keeps the curve and the suggestion in extras and nothing a fit
	reads (probes/local.py with_lr_range)."""
	from preflight.probes.lr_range import LR_END, LR_START, STEPS

	from .flags import flag

	if not flag("probes.lr_range") or sm.lens not in LR_RANGE_LENSES or not plan or ctx.should_stop():
		return None
	first = next((q for q in probes if q.id == plan[0].probe_id), None)
	if first is None or first.status not in ("ok", "cached") or not first.steps:
		return None  # a p1 from the cache ran too (A's review, Nov 2)
	# the point's time: its first probe's, scaled to the ramp's steps [guess: setup plus steps]
	per_step = (first.step_time.mean_s if first.step_time else None) or 1.0
	est_s = (first.duration_s or 60.0) + STEPS * per_step
	usd_per_s = (_price(gpu) + deps.overhead_usd_per_h) / 3600
	if not budget.allows(usd_per_s * est_s, est_s):
		return None
	# it may run past its estimate, never past what the job has left (A's review, Nov 2)
	left_s = budget.left_s()
	if usd_per_s > 0:
		left_s = min(left_s, (budget.usd - budget.spent_usd) / usd_per_s)
	p = replace(plan[0], probe_id=f"{plan[0].probe_id}-lr", warmup=STEPS, timed=0, profile=False)
	try:
		spec = _spec(fetched, sm, p, deps, eval_off=True)
	except OverrideError:
		return None
	# a floor of 1 s [guess] so the timeout is never 0
	spec = replace(spec, lr_range={"start": LR_START, "end": LR_END, "steps": STEPS},
		timeout_s=max(1.0, min(spec.timeout_s, left_s)))  # fmt: skip
	if spec.nproc > 1:
		return None  # a guard (no plan runs ranks today): each rank would stop on its own loss
	result, cached = _run_point(spec, gpu, deps, ctx)
	if result is None:
		return cached
	over = (result.probe.duration_s or 0) * deps.overhead_usd_per_h / 3600
	budget.spent_usd += result.probe.cost.usd + over
	budget.gpu_usd += result.probe.cost.usd
	budget.overhead_usd += over
	return result.probe


SEARCH_MAX_QUESTIONS = 2  # guess: searches a job asks, in series, before the visitor gets the record
SEARCH_JOB_S = 120.0  # guess: no new search starts after this long searching in one job
SEARCH_LIBRARIES = ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes",
	"vllm", "sentence-transformers", "torch_geometric", "torchvision", "timm", "numpy", "deepspeed")  # fmt: skip


def _search(
	rec: FlightRecord, results: list, deps: Deps, *, private: bool = False, verify: bool = False, ctx=None
) -> tuple[FlightRecord, list]:
	"""Muse Spark searches the web for a known library issue matching each failed probe's error
	(D, Nov 2): at most 3 sources the search returned, or "no known issue found". Off unless
	harness.spark_search is on and the Meta client is configured; never for a private job (its
	error text would go to Meta and the web), nor a verify job (one question, as with the
	diagnosis). One search per distinct error code in a job: several points that ran out of memory
	the same way are one question; at most SEARCH_MAX_QUESTIONS, none started past SEARCH_JOB_S
	(A's review, Nov 2). Every call is in the record's provenance."""
	from .flags import flag

	if private or verify or deps.spark is None or not flag("harness.spark_search"):
		return rec, []
	from preflight.harness.search_grounding import ground_failure, skipped

	findings, calls, asked = [], [], set()
	started = time.monotonic()
	for r in results:
		if not isinstance(r, ProbeResult) or skipped(r.probe):
			continue
		if ctx is not None and ctx.should_stop():
			break
		key = r.probe.error_code or r.probe.error.splitlines()[0][:120]
		if key in asked:
			continue
		if len(asked) >= SEARCH_MAX_QUESTIONS or (asked and time.monotonic() - started > SEARCH_JOB_S):
			break
		asked.add(key)
		env = r.env or rec.env
		libs = {k: v for k, v in (env.libraries if env is not None else {}).items() if k in SEARCH_LIBRARIES}
		before = len(deps.spark.calls)
		try:
			f, cs = ground_failure(deps.spark, r.probe, libs, rec.job.lens)
		except Exception:  # a search never costs the visitor their record; its calls stay in provenance
			log.warning("job %s: search grounding failed", rec.record_id, exc_info=True)
			f, cs = None, deps.spark.calls[before:]
		calls += cs
		if f is not None:
			findings.append(f)
	return rec.model_copy(update={"provenance": [*rec.provenance, *calls]}), findings


def _with_step(probe, entrypoint: str | None):
	if probe.status in ("ok", "cached") or not probe.error:
		return probe
	from .failures import with_step

	return probe.model_copy(update={"error": with_step(probe.error, probe.error_code, entrypoint=entrypoint)})


def _tested_node(probe, nodes: set[str]) -> str | None:
	"""The one job-map node a probe clearly tested, else None (C lists those findings apart): the
	eval probe, or an eval-pass OOM, tested the eval node."""
	is_eval = probe.id.endswith("-eval") or (probe.error_code or "").startswith("oom.eval")
	return "eval" if is_eval and "eval" in nodes else None


def _private_history(rec: FlightRecord) -> bool:
	"""Whether any probe or finding on the record came from a private job (marked by the pipeline): then nothing on
	it goes to a hosted model, in this job or any later one (B, Nov 2, from A's review of the claim check)."""
	return any(p.extras.get("private_job") for p in rec.probes) or any(
		e.metric == "private_job" for f in rec.findings for e in f.evidence
	)


def _claim_check(rec: FlightRecord, deps: Deps, *, private: bool) -> FlightRecord:
	"""Every number a model-written finding states, checked against the record's raw probe numbers (harness/claim_check.py);
	one with no supporting number reads "unsupported: run a probe". Off unless harness.spark_claim_check is on. Spark
	Open only picks among fields the code found; a private job's text never goes to Baseten, so there it is code only."""
	from .flags import flag

	if not flag("harness.spark_claim_check"):
		return rec
	from preflight.harness.claim_check import annotate_findings

	try:
		findings, calls = annotate_findings(rec, client=None if private else deps.triage)
	except Exception:  # the check never costs the visitor their record
		log.warning("job %s: claim check failed", rec.record_id, exc_info=True)
		return rec
	added = round(sum(c.cost_usd for c in calls), 6)
	costs = rec.costs
	if added:
		breakdown = dict(costs.breakdown or {})
		breakdown["model_calls"] = round(breakdown.get("model_calls", 0) + added, 6)
		costs = costs.model_copy(
			update={"llm_usd": round((costs.llm_usd or 0) + added, 6), "breakdown": breakdown}
		)
	return rec.model_copy(
		update={"findings": findings, "provenance": [*rec.provenance, *calls], "costs": costs}
	)


def _diagnose(rec: FlightRecord, deps: Deps) -> FlightRecord:
	"""Muse Spark checks the finished record's plots against its numbers (plan item 31): one call,
	in the record's provenance; its findings replace any earlier diagnosis. Off unless
	harness.spark_diagnosis is on and the Meta client is configured."""
	from .flags import flag

	if deps.spark is None or not flag("harness.spark_diagnosis"):
		return rec
	from preflight.harness.diagnosis import diagnose_record

	try:
		found, calls, stats = diagnose_record(deps.spark, rec)
	except Exception:  # a diagnosis never costs the visitor their record
		log.warning("job %s: diagnosis failed", rec.record_id, exc_info=True)
		return rec
	log.info("job %s: diagnosis %s", rec.record_id, stats)
	if found and deps.triage is not None:
		# Spark Open checks each claim against the probe numbers through a tool call (item 30);
		# without it the claims stand as Muse Spark made them, citations checked by code
		from preflight.harness.diagnosis_check import check_diagnosis

		try:
			found, checks = check_diagnosis(deps.triage, rec, found)
			calls = [*calls, *checks]
		except Exception:  # the check never costs the visitor their record; its calls are lost with it
			log.warning("job %s: diagnosis check failed", rec.record_id, exc_info=True)
	keep = [f for f in rec.findings if f.source != "spark_diagnosis"]
	# after _merge priced the record: these calls go into its model-call cost too
	added = round(sum(c.cost_usd for c in calls), 6)
	breakdown = dict(rec.costs.breakdown or {})
	if added:
		breakdown["model_calls"] = round(breakdown.get("model_calls", 0) + added, 6)
	costs = rec.costs.model_copy(
		update={"llm_usd": round((rec.costs.llm_usd or 0) + added, 6), "breakdown": breakdown}
	)
	return rec.model_copy(
		update={"findings": [*keep, *found], "provenance": [*rec.provenance, *calls], "costs": costs}
	)


def _measured_gpu(result) -> str | None:
	from preflight.probes.gpu_choice import canonical

	env = getattr(result, "env", None)
	return canonical(env.gpu.name) if env is not None and env.gpu is not None and env.gpu.name else None


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
	judged: bool = False,  # judging started the job on the A10 (judging.py)
) -> tuple[list[Probe], list[ProbeResult], str, float]:
	"""Run the planned points in order; returns their probes, fresh results, the GPU they settled on
	and what discarded attempts cost."""
	from .flags import flag

	probes: list[Probe] = []
	results: list[ProbeResult] = []
	discarded_usd = 0.0
	stop_on_import = flag("runner.stop_on_import_error")  # off until its check passes
	for i, p in enumerate(plan):
		ctx.check()
		_emit(ctx, "probing", rec, rev, probe_id=p.probe_id, progress=round(i / len(plan), 3))
		if any(q.error_code == "gpu.unavailable" for q in probes):
			# no GPU could be scheduled within the job's wait cap: the next point would wait for nothing
			probes.append(Probe(id=p.probe_id, knobs=p.knobs, status="error", steps=0,
				error="skipped: no GPU could be scheduled (see the first probe)", error_code="gpu.unavailable"))  # fmt: skip
			continue
		gone = next((q for q in probes if q.error_code == "missing_package" and not q.steps), None)
		if (
			gone is not None
			and stop_on_import
			# the same code: a patched point (C's fix loop) may import something else
			and (p.probe_id in (patched_for or {})) == (gone.id in (patched_for or {}))
			and _import_knobs(p.knobs) == _import_knobs(gone.knobs)
		):
			# an import that failed before the first step fails the same way at every point: skip the
			# rest and their triage (unseen runs, Nov 2: three probes and three paid triage calls each)
			probes.append(Probe(id=p.probe_id, knobs=p.knobs, status="error", steps=0,
				error=f"skipped: the script couldn't import a package (see {gone.id})", error_code="missing_package",
				missing_module=gone.missing_module, missing_package=gone.missing_package))  # fmt: skip
			continue
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

		def run(g, spec=spec, attempts=attempts, over=over, first=not probes):
			fb = _fallbacks(g, allowed, pinned)
			if judged and first and g == "A10" and (allowed is None or "L4" in allowed):
				fb = ("L4",)  # judging's A10 start, before any point ran (GPU_FALLBACK's note)
			result, cached = _run_point(replace(spec, gpu_fallback=fb), g, deps, ctx)
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
		# A verify side's eval running out is that side's result ("before failed"), never re-run with
		# the eval off (A's design, Nov 1): a re-run would compare training alone with a patched eval.
		if out.probe.status == "oom" and out.probe.error_code == "oom.eval" and p.role not in VERIFY_ROLES:
			try:
				# the eval batch ran out, not training: same GPU, eval off; the eval probe keeps
				# its memory per batch so the guide can say where it ran out
				# the point's own code, patched when the job carries a patch (A's review, Nov 1)
				again = run(ladder.tried[-1], spec=_spec(use_f, use_sm, p, deps, eval_off=True))
				probes.append(out.probe.model_copy(update={"id": f"{p.probe_id}-eval"}))
				if isinstance(out, ProbeResult):
					# its own id, as in the record: the eval-off re-run below is the point's p-id
					out.probe = out.probe.model_copy(update={"id": f"{p.probe_id}-eval"})
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
			ran = getattr(deps.runner, "last_ran", None)
			if ran and ran != gpu:
				# this point ran on a fallback GPU: the rest run there too, and points answered from
				# the cache for the GPU asked for leave the record, so it holds one GPU's numbers
				_emit(ctx, "probing", rec, rev, level="warn",
					message=f"no {gpu} could be scheduled; this job's points run on {ran}")  # fmt: skip
				probes = [q if q.status != "cached" else q.model_copy(update={"status": "error",
					"error_code": "gpu.fallback", "cache_key": None,
					"error": f"left out: answered from the cache for {gpu}, and this job's points ran on {ran}"})
					for q in probes]  # fmt: skip
				gpu = ran
		kept_ids = {id(out), *(id(r) for r in results)}
		discarded_usd += sum(
			a.probe.cost.usd for a in attempts if isinstance(a, ProbeResult) and id(a) not in kept_ids
		)
	return probes, results, gpu, discarded_usd


VERIFY_ROLES = ("verify", "verify_base", "verify_seed")


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
			from preflight.engine import api
			from preflight.engine.api import fill_predictions

			# the engine reads no files: its switches come from the engine.* flags (from-b.md, B -> A)
			# every engine switch the engine names (api.FLAG_OF), so a new one reaches production when its flag goes on
			# (B, Nov 2: engine.carry_free_knobs and engine.context_same_stack were missing from a fixed list)
			api.FEATURES.update({name: flag(key) for name, key in api.FLAG_OF.items()})
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
