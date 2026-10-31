"""Guide jobs on an existing record (run_probe, run_verify, sample_more), C's knob fields, the job
error code and the demo's GPU ceiling."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from preflight.harness.knob_meta import changes_results
from preflight.probes.gpu_choice import choose_probe_gpu, next_gpu
from preflight.probes.local import InvalidSubmission, ProbeResult
from preflight.runner.pipeline import Deps, _Ctx, run_pipeline
from preflight.runner.queue import InMemoryJobStore
from preflight.runner.records import LocalRecordWriter
from preflight.runner.remote_store import DirBlobs, StoreMethods, externalise_files
from preflight.runner.worker import PermanentError, run_one
from preflight.schema import Env, FlightRecord, JobRequest, Probe, Stability, load_record

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import torch
batch_size = 8
seq_len = 64
grad_accum = 1
lr = 0.1
model = torch.nn.Linear(4, 4)
opt = torch.optim.SGD(model.parameters(), lr)
"""


class LossRunner:
	"""A fake GPU: each probe 'trains' with a loss that depends on its knobs, and is recorded."""

	def __init__(self, loss=None, outcomes=None):
		self.calls = []
		self.loss = loss or (lambda knobs, step: 2.0 / (1 + step))
		self.outcomes = outcomes or {}

	def __call__(self, spec, gpu, *, should_stop):
		self.calls.append({"id": spec.probe_id, "gpu": gpu, "knobs": dict(spec.knobs), "seed": spec.seed,
			"force": spec.seed_force,
			"timed": spec.timed, "files": dict(spec.files)})  # fmt: skip
		status, code = self.outcomes.get(spec.probe_id, ("ok", None))
		loss = self.loss
		try:
			series = {"loss": [loss(spec.knobs, s, spec.seed) for s in range(spec.timed)]}
		except TypeError:
			series = {"loss": [loss(spec.knobs, s) for s in range(spec.timed)]}
		series = series if status == "ok" else {}
		probe = Probe(id=spec.probe_id, knobs=spec.knobs, status=status, error_code=code,
			steps=spec.warmup + spec.timed, duration_s=30.0, stability={"series": series},
			cost={"gpu_seconds": 30.0, "usd": 30.0 * spec.gpu_usd_per_hour / 3600})  # fmt: skip
		return ProbeResult(probe=probe, env=Env(backend="cuda", provider="modal", gpu={"name": gpu, "memory_bytes": 24e9}),
			summary={}, kernels=None, arch=None, trace_file=None, log_tail="", workdir=Path("."))  # fmt: skip


def _deps(tmp_path, runner, **kw):
	store = LocalRecordWriter(tmp_path / "records")
	kw.setdefault("blobs", DirBlobs(tmp_path / "uploads"))
	return Deps(records=store, runner=runner, reader=store, **kw)


def _parent(tmp_path, runner=None):
	"""A planned record from an upload, as the job store hands it over (files moved to the blobs)."""
	runner = runner or LossRunner()
	d = _deps(tmp_path, runner)
	request = externalise_files({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, d.blobs)
	rec = run_pipeline(request, _Ctx(), d)
	d.records.put(rec)
	return rec, d


def _on(rec, mode, **kw):
	return {"source": {"kind": "record", "record_ref": rec.share_token}, "mode": mode, **kw}


# ---------------------------------------------------------------- the request contract


@pytest.mark.parametrize(
	"body",
	[
		{"source": {"kind": "upload"}, "mode": "probe", "knobs": {"batch_size": 4}},  # no record
		{"source": {"kind": "record", "record_ref": "sample:lora-qwen"}, "mode": "probe"},  # no knobs
		{"source": {"kind": "upload"}, "knobs": {"batch_size": 4}},  # knobs on a plan
		{"source": {"kind": "record", "record_ref": "sample:agent"}, "mode": "sample_more"},  # no tasks
		{"source": {"kind": "upload"}, "tasks": 20},  # tasks without sample_more
		{"source": {"kind": "record", "record_ref": "x" * 16}, "mode": "sample_more", "tasks": 99},
		{"source": {"kind": "record"}, "mode": "verify"},  # no ref
		{"source": {"kind": "upload"}, "purpose": "Guide Run!"},
	],
)
def test_job_requests_that_dont_add_up_are_refused(body):
	with pytest.raises(ValidationError):
		JobRequest.model_validate(body)


def test_the_requests_c_sends_validate():
	ref = {"kind": "record", "record_ref": "golden-finetune-share-tok"}
	for body in (
		{"source": ref, "purpose": "guide.run_probe", "mode": "probe", "knobs": {"batch_size": 2}},
		{"source": ref, "purpose": "guide.run_verify", "mode": "verify", "knobs": {"batch_size": 4}},
		{"source": {"kind": "record", "record_ref": "sample:agent"}, "purpose": "guide.sample_more",
			"mode": "sample_more", "tasks": 20},
	):  # fmt: skip
		JobRequest.model_validate(body)


# ---------------------------------------------------------------- probe


def test_run_probe_writes_a_new_record_from_the_parent_and_leaves_the_parent_alone(tmp_path):
	parent, _ = _parent(tmp_path)
	before = (tmp_path / "records" / f"{parent.share_token}.json").read_text()
	runner = LossRunner()
	ctx = _Ctx()
	rec = run_pipeline(
		_on(parent, "probe", knobs={"batch_size": 4, "seq_len": 64}), ctx, _deps(tmp_path, runner)
	)
	assert rec.parent_id == parent.record_id and rec.share_token != parent.share_token
	assert [p.id for p in rec.probes] == [*(p.id for p in parent.probes), "p4"]
	new = rec.probes[-1]
	assert new.knobs == {"batch_size": 4, "seq_len": 64}
	# only the changed knob is written into the code; no Muse Spark, same knobs as the parent
	(call,) = runner.calls
	assert "batch_size = 4" in call["files"]["train.py"] and "seq_len = 64" in call["files"]["train.py"]
	assert [k.id for k in rec.knobs] == [k.id for k in parent.knobs]
	assert rec.costs.gpu_usd == pytest.approx(parent.costs.gpu_usd + 30 * 0.80 / 3600, rel=1e-3)
	assert (tmp_path / "records" / f"{parent.share_token}.json").read_text() == before
	sk = next(e for e in ctx.events if e["stage"] == "skeleton")
	assert sk["share_token"] == rec.share_token and parent.share_token in sk["message"]


def test_run_probe_on_a_pinned_gpu_does_not_step_up_after_an_oom(tmp_path):
	parent, _ = _parent(tmp_path)
	runner = LossRunner(outcomes={"p1": ("oom", "oom")})
	rec = run_pipeline(_on(parent, "probe", knobs={"batch_size": 16, "gpu_type": "L40S"}), _Ctx(),
		_deps(tmp_path, runner))  # fmt: skip
	assert [c["gpu"] for c in runner.calls] == ["L40S"]
	# measured on another GPU than the parent: its probes don't belong in this record
	assert [p.id for p in rec.probes] == ["p4"] and rec.env.gpu.name == "L40S"


def test_unknown_knobs_are_left_alone_and_out_of_domain_values_refused(tmp_path):
	parent, _ = _parent(tmp_path)
	ctx = _Ctx()
	run_pipeline(_on(parent, "probe", knobs={"batch_size": 4, "nope": 3}), ctx, _deps(tmp_path, LossRunner()))
	assert any("nope" in (e.get("message") or "") for e in ctx.events)
	with pytest.raises(InvalidSubmission) as exc:
		run_pipeline(_on(parent, "probe", knobs={"batch_size": -1}), _Ctx(), _deps(tmp_path, LossRunner()))
	assert exc.value.code == "submission.knob_value"


def test_record_refs_that_dont_resolve_are_visitor_errors(tmp_path):
	d = _deps(tmp_path, LossRunner())
	for ref, code in (
		("missing-token-0000000", "submission.record_unknown"),
		("sample:agent", "submission.sample_unknown"),
	):
		with pytest.raises(InvalidSubmission) as exc:
			run_pipeline(
				{
					"source": {"kind": "record", "record_ref": ref},
					"mode": "probe",
					"knobs": {"batch_size": 1},
				},
				_Ctx(),
				d,
			)
		assert exc.value.code == code  # fmt: skip


def test_a_sample_ref_measures_its_zoo_script(tmp_path):
	runner = LossRunner()
	rec = run_pipeline({"source": {"kind": "record", "record_ref": "sample:pretrain"}, "mode": "probe",
		"knobs": {"batch_size": 4}}, _Ctx(), _deps(tmp_path, runner))  # fmt: skip
	assert rec.parent_id == "sample:pretrain" and rec.job.source.uri == "sample://zoo/pretrain_nanogpt"
	assert [p.id for p in rec.probes] == ["p1"] and rec.probes[0].knobs == {"batch_size": 4}


def test_an_upload_is_measured_again_from_the_kept_files(tmp_path):
	blobs = DirBlobs(tmp_path / "uploads")
	svc = StoreMethods(InMemoryJobStore(), blobs)
	sub = svc.submit({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, "v", "k1")
	lease = svc.store.lease("w")
	d = _deps(tmp_path, LossRunner(), blobs=blobs)
	parent = run_pipeline(lease.request, _Ctx(), d)
	assert sub["job_id"] and parent.job.source.uri.startswith("upload://sha256:")
	d.records.put(parent)
	rec = run_pipeline(_on(parent, "probe", knobs={"batch_size": 2}), _Ctx(), d)
	assert rec.probes[-1].knobs == {"batch_size": 2} and rec.job.source == parent.job.source


# ---------------------------------------------------------------- verify


def test_verify_at_the_current_settings_is_one_fixed_seed_run(tmp_path):
	parent, _ = _parent(tmp_path)
	runner = LossRunner()
	rec = run_pipeline(_on(parent, "verify"), _Ctx(), _deps(tmp_path, runner))
	(call,) = runner.calls
	assert call["id"] == "v1" and call["seed"] is not None and call["timed"] == 20
	f = rec.findings[-1]
	assert f.code == "verify.passed" and f.node_id == "loss"


def test_verify_keeps_the_loss_when_the_effective_batch_holds(tmp_path):
	parent, _ = _parent(tmp_path)
	runner = LossRunner()  # loss doesn't depend on the split
	rec = run_pipeline(
		_on(parent, "verify", knobs={"batch_size": 4, "grad_accum": 2}), _Ctx(), _deps(tmp_path, runner)
	)
	assert [c["id"] for c in runner.calls] == ["v1", "v1-base", "v1-seed"]
	seeds = {c["id"]: c["seed"] for c in runner.calls}
	assert seeds["v1"] == seeds["v1-base"] != seeds["v1-seed"] and all(c["force"] for c in runner.calls)
	assert rec.findings[-1].code == "verify.passed"


def test_verify_passes_a_change_inside_the_measured_seed_spread_and_fails_one_outside_it(tmp_path):
	parent, _ = _parent(tmp_path)

	def loss(knobs, step, seed, bump):
		wiggle = 0.02 if seed % 2 else 0.0  # the second seed lands 2% off
		change = bump if knobs.get("batch_size") == 4 else 0.0
		return 2.0 / (1 + step) * (1 + wiggle + change)

	inside = run_pipeline(_on(parent, "verify", knobs={"batch_size": 4}), _Ctx(),
		_deps(tmp_path, LossRunner(loss=lambda k, s, seed: loss(k, s, seed, 0.01))))  # fmt: skip
	f = inside.findings[-1]
	assert f.code == "verify.passed"
	ev = {e.metric: e for e in f.evidence}
	assert ev["verify.loss_delta"].value == pytest.approx(0.01) and ev[
		"verify.seed_spread"
	].value == pytest.approx(0.02)
	assert ev["verify.loss_delta"].threshold == ev["verify.seed_spread"].value  # measured, not a fixed %
	outside = run_pipeline(_on(parent, "verify", knobs={"batch_size": 4}), _Ctx(),
		_deps(tmp_path, LossRunner(loss=lambda k, s, seed: loss(k, s, seed, 0.05))))  # fmt: skip
	assert outside.findings[-1].code == "verify.loss_changed"


def test_verify_flags_a_change_that_alters_the_loss_and_says_why(tmp_path):
	parent, _ = _parent(tmp_path)
	runner = LossRunner(loss=lambda knobs, s: (3.0 if knobs.get("batch_size") == 4 else 2.0) / (1 + s))
	rec = run_pipeline(_on(parent, "verify", knobs={"batch_size": 4}), _Ctx(), _deps(tmp_path, runner))
	f = rec.findings[-1]
	assert f.code == "verify.loss_changed" and f.severity == "warn"
	assert "effective batch changed from 8 to 4" in f.detail
	assert f.evidence[0].value == pytest.approx(0.5)


def test_verify_says_so_when_the_plan_does_not_run(tmp_path):
	parent, _ = _parent(tmp_path)
	runner = LossRunner(outcomes={"v1": ("oom", "oom")})
	rec = run_pipeline(_on(parent, "verify", knobs={"batch_size": 64}), _Ctx(), _deps(tmp_path, runner))
	assert rec.findings[-1].code == "verify.failed" and rec.findings[-1].severity == "error"


# ---------------------------------------------------------------- sample_more and job error codes


def test_sample_more_is_refused_on_training_runs_and_not_yet_built_for_agents(tmp_path):
	parent, _ = _parent(tmp_path)
	with pytest.raises(InvalidSubmission) as exc:
		run_pipeline(_on(parent, "sample_more", tasks=20), _Ctx(), _deps(tmp_path, LossRunner()))
	assert exc.value.code == "submission.not_agent"
	agent = load_record((ROOT / "schema" / "examples" / "agent.json").read_text())
	store = LocalRecordWriter(tmp_path / "records")
	src = agent.job.source.model_copy(
		update={"uri": "sample://zoo/pretrain_nanogpt", "entrypoint": None, "content_hash": None}
	)
	store.put(agent.model_copy(update={"job": agent.job.model_copy(update={"source": src})}))
	with pytest.raises(PermanentError) as exc:
		run_pipeline(_on(agent, "sample_more", tasks=20), _Ctx(), _deps(tmp_path, LossRunner()))
	assert exc.value.code == "unsupported.agent_sampling"


def test_a_failed_job_carries_a_stable_error_code(tmp_path):
	store = InMemoryJobStore()
	sub = store.submit({"source": {"kind": "upload"}}, visitor="v", idempotency_key="k")

	def job(lease, ctx):
		raise InvalidSubmission("the job has no files", code="submission.files_gone")

	run_one(store, "w", job)
	ev = store.events(sub.job_id)[-1]
	assert ev.stage == "failed" and ev.error_code == "submission.files_gone"
	assert store.info(sub.job_id)["error_code"] == "submission.files_gone"
	sub2 = store.submit({"source": {"kind": "upload"}}, visitor="v", idempotency_key="k2")

	def broken(lease, ctx):
		raise RuntimeError("bug")

	while store.status(sub2.job_id) != "failed":
		run_one(store, "w", broken)
	assert store.events(sub2.job_id)[-1].error_code == "internal"


# ---------------------------------------------------------------- knob fields and fixtures


def test_skeleton_knobs_say_what_they_act_on_and_whether_they_change_results(tmp_path):
	rec, _ = _parent(tmp_path)
	nodes = {n.id for n in rec.graph.nodes}
	k = {x.id: x for x in rec.knobs}
	assert set(k["batch_size"].affects) <= nodes and "data" in k["batch_size"].affects
	assert k["seq_len"].changes_results is True and k["batch_size"].changes_results is None
	assert k["gpu_type"].changes_results is False
	assert changes_results("lora_r") and not changes_results("num_workers")
	bad = json.loads(rec.model_dump_json())
	bad["knobs"][0]["affects"] = ["not_a_node"]
	with pytest.raises(ValidationError, match="unknown nodes"):
		FlightRecord.model_validate(bad)


def test_golden_records_mark_their_fixture_models():
	for name in ("finetune", "pretrain", "agent"):
		rec = load_record((ROOT / "schema" / "examples" / f"{name}.json").read_text())
		assert rec.models and all(m.fixture for m in rec.models)


# ---------------------------------------------------------------- the demo's GPU ceiling


def test_the_public_demo_never_probes_above_its_ceiling(tmp_path, monkeypatch):
	monkeypatch.setenv("PREFLIGHT_GPU_CEILING", "A100-SXM-80GB")
	assert choose_probe_gpu(200.0) == "A100-SXM-80GB"
	assert next_gpu("A100-SXM-80GB") is None
	assert choose_probe_gpu(10.0, ["H200-SXM-141GB"]) == "A100-SXM-80GB"
	runner = LossRunner()
	ctx = _Ctx()
	run_pipeline({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}, "gpu": "H200-SXM-141GB"}, ctx,
		_deps(tmp_path, runner))  # fmt: skip
	assert {c["gpu"] for c in runner.calls} == {"A100-SXM-80GB"}
	assert any("cross-GPU" in (e.get("message") or "") for e in ctx.events)


def test_a_mistyped_ceiling_still_caps(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_GPU_CEILING", "NOT-A-GPU")
	assert choose_probe_gpu(200.0) == "A100-SXM-80GB"


def test_verify_fails_a_run_whose_loss_went_nan(tmp_path):
	parent, _ = _parent(tmp_path)

	class NaNRunner(LossRunner):
		def __call__(self, spec, gpu, *, should_stop):
			r = super().__call__(spec, gpu, should_stop=should_stop)
			if spec.probe_id == "v1":
				series = r.probe.stability.series["loss"]
				bad = {
					"series": {"loss": [*series[:2], *([None] * (len(series) - 2))]},
					"nan_count": len(series) - 2,
				}
				r.probe = r.probe.model_copy(update={"stability": Stability.model_validate(bad)})
			return r

	rec = run_pipeline(
		_on(parent, "verify", knobs={"batch_size": 4, "grad_accum": 2}), _Ctx(), _deps(tmp_path, NaNRunner())
	)
	assert rec.findings[-1].code == "verify.diverged" and rec.findings[-1].severity == "error"


def test_a_record_whose_kept_files_changed_is_not_measured_as_if_they_hadnt(tmp_path):
	parent, d = _parent(tmp_path)
	key = parent.job.source.uri.removeprefix("upload://")
	d.blobs._path(key).write_text(json.dumps({"train.py": SCRIPT + "\nx = 2\n"}))
	with pytest.raises(InvalidSubmission) as exc:
		run_pipeline(_on(parent, "probe", knobs={"batch_size": 2}), _Ctx(), d)
	assert exc.value.code == "submission.code_changed"


def test_a_visitor_cannot_point_an_upload_at_someone_elses_files(tmp_path):
	d = _deps(tmp_path, LossRunner())
	rec = run_pipeline({"source": {"kind": "upload", "uri": "upload://sha256:" + "0" * 64,
		"files": {"train.py": SCRIPT}}}, _Ctx(), d)  # fmt: skip
	assert rec.job.source.uri is None


def test_moving_to_another_gpu_drops_what_was_fitted_on_the_parents_probes(tmp_path):
	parent, d = _parent(tmp_path)
	from preflight.schema import Evidence, Finding, ModelCard

	f = Finding(id="m1", severity="warn", code="data.padding", title="t", source="rule",
		evidence=[Evidence(metric="probe.data.padding_fraction", value=0.4, probe_id="p1")])  # fmt: skip
	p2 = parent.model_copy(
		update={"findings": [*parent.findings, f], "models": [ModelCard(id="m@1", kind="rule", version="1")]}
	)
	d.records.put(p2)
	rec = run_pipeline(_on(p2, "probe", knobs={"batch_size": 4, "gpu_type": "L40S"}), _Ctx(), d)
	assert all(x.id != "m1" for x in rec.findings) and rec.models == []


def test_requirement_installs_are_shared_by_a_jobs_probes_and_pinned_to_public_versions(
	tmp_path, monkeypatch
):
	from preflight.recorder import requirements as R

	calls = []

	def fake_run(cmd, **kw):
		calls.append(cmd)
		prefix = Path(cmd[cmd.index("--prefix") + 1])
		(prefix / "lib" / "python3.11" / "site-packages").mkdir(parents=True)
		return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

	monkeypatch.setattr(R.subprocess, "run", fake_run)
	monkeypatch.setattr(R, "installed", lambda: {"torch": "2.5.1+cu121"})
	monkeypatch.setenv("PREFLIGHT_DEPS_DIR", str(tmp_path / "deps"))
	(tmp_path / "deps").mkdir()
	work = tmp_path / "w"
	work.mkdir()
	(work / "requirements.txt").write_text("einops\n")
	for _ in range(2):
		err, dirs = R.install(work, tmp_path / "own", log=lambda m: None)
		assert err is None and dirs and dirs[0].name == "site-packages"
	assert len(calls) == 1 and "--target" not in calls[0]
	cons = Path(calls[0][calls[0].index("-c") + 1]).read_text()
	assert cons == "torch==2.5.1\n"


@pytest.mark.skipif(
	not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() != 0, reason="root only"
)
def test_the_cli_as_root_gives_each_run_fresh_caches_it_never_walks(monkeypatch):
	import os
	import shutil

	from preflight.cli.main import _run_caches_when_root

	for v in ("HF_HOME", "PREFLIGHT_PIP_CACHE", "PREFLIGHT_DEPS_DIR", "PREFLIGHT_PROBE_UID"):
		monkeypatch.delenv(v, raising=False)
	run = _run_caches_when_root()
	try:
		uid = int(os.environ["PREFLIGHT_PROBE_UID"])
		assert run.stat().st_uid == 0 and run.stat().st_mode & 0o777 == 0o711
		for v in ("HF_HOME", "PREFLIGHT_PIP_CACHE", "PREFLIGHT_DEPS_DIR"):
			d = Path(os.environ[v])
			assert d.parent == run and d.stat().st_uid == uid and not any(d.iterdir())
		again = _run_caches_when_root()
		assert again != run  # nothing carries over from a run whose probes could have planted links
		shutil.rmtree(again)
	finally:
		shutil.rmtree(run)


AGENT_SCRIPT = """from openai import OpenAI

client = OpenAI()
for task in range(500):
	client.chat.completions.create(model="muse-spark-1.3", messages=[{"role": "user", "content": str(task)}])
"""


class FakeSampler:
	"""Stands in for runner/agent_jobs.local_sampler: k tasks with fresh ids, and what it was asked."""

	def __init__(self):
		self.calls = []

	def __call__(self, fetched, sm, *, k, skip=0, exclude=(), run_id="s1"):
		from preflight.schema import AgentSample

		self.calls.append(
			{"k": k, "skip": skip, "exclude": exclude, "run_id": run_id, "entry": fetched.entrypoint}
		)
		n = len(self.calls)
		tasks = [{"task_id": f"{run_id}-{n}-{i}", "turns": 3, "input_tokens": 900, "output_tokens": 120,
			"latency_s": 4.0, "cost_usd": 0.01 + 0.001 * i} for i in range(k)]  # fmt: skip
		return AgentSample.model_validate(
			{"tasks": tasks, "population": 500, "selection": f"first {k} tasks"}
		)


def test_an_agent_run_is_sampled_and_a_top_up_adds_twenty_more(tmp_path):
	sampler = FakeSampler()
	d = _deps(tmp_path, LossRunner(), agent=sampler)
	request = externalise_files({"source": {"kind": "upload", "files": {"run.py": AGENT_SCRIPT}}}, d.blobs)
	rec = run_pipeline(request, _Ctx(), d)
	assert rec.job.lens == "agent" and not rec.probes
	assert len(rec.agent_sample.tasks) == 30 and sampler.calls[0]["k"] == 30 and sampler.calls[0]["skip"] == 0
	d.records.put(rec)
	more = run_pipeline(_on(rec, "sample_more", tasks=20), _Ctx(), d)
	c = sampler.calls[1]
	assert c["k"] == 20 and c["skip"] == 30 and c["run_id"] == "s2"
	assert more.parent_id == rec.record_id and len(more.agent_sample.tasks) == 50
	assert "top-up of 20" in more.agent_sample.selection


def test_an_agent_script_that_never_calls_the_proxy_fails_with_a_reason(tmp_path):
	from preflight.schema import AgentSample

	d = _deps(tmp_path, LossRunner(), agent=lambda *a, **kw: AgentSample(tasks=[], population=None))
	request = externalise_files({"source": {"kind": "upload", "files": {"run.py": AGENT_SCRIPT}}}, d.blobs)
	with pytest.raises(PermanentError) as exc:
		run_pipeline(request, _Ctx(), d)
	assert exc.value.code == "probe.no_agent_calls"
