"""The job pipeline end to end on CPU: fetch, map, skeleton, plan, probes, merge, through the queue and worker."""

import json
from pathlib import Path

import pytest

from preflight.probes.local import ProbeResult
from preflight.runner.pipeline import Budget, Deps, MemoryCache, _Ctx, run_job, run_pipeline
from preflight.runner.queue import InMemoryJobStore
from preflight.runner.records import LocalRecordWriter
from preflight.runner.worker import run_one
from preflight.schema import Env, Probe, load_record

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def upload(name, **extra):
	return {
		"source": {"kind": "upload", "files": {name: (FIX / name).read_text()}, "entrypoint": name},
		**extra,
	}


# ---------------------------------------------------------------- fakes (fast, no torch)


class FakeRunner:
	"""Answers probes from a table keyed by (probe id, gpu); records what it was asked."""

	def __init__(self, outcomes=None, seconds=30.0):
		self.calls = []
		self.outcomes = outcomes or {}
		self.seconds = seconds

	def __call__(self, spec, gpu, *, should_stop):
		self.calls.append((spec.probe_id, gpu, spec.eval_batches, dict(spec.knobs), list(spec.args)))
		status, code = self.outcomes.get((spec.probe_id, gpu), self.outcomes.get(spec.probe_id, ("ok", None)))
		probe = Probe(id=spec.probe_id, knobs=spec.knobs, status=status, error_code=code, steps=7 if status == "ok" else 1,
			duration_s=self.seconds, cost={"gpu_seconds": self.seconds, "usd": self.seconds * spec.gpu_usd_per_hour / 3600})  # fmt: skip
		return ProbeResult(probe=probe, env=Env(backend="cuda", provider="modal", gpu={"name": gpu, "memory_bytes": 24e9}),
			summary={}, kernels=None, arch={"n_layers": 2, "source": "modules"}, trace_file=None, log_tail="", workdir=Path("."))  # fmt: skip


SCRIPT = """
import torch
batch_size = 8
seq_len = 64
eval_every = 100
model = torch.nn.Linear(4, 4)
opt = torch.optim.SGD(model.parameters(), 0.1)
"""


def deps(tmp_path, runner, **kw):
	return Deps(records=LocalRecordWriter(tmp_path / "records"), runner=runner, **kw)


def test_skeleton_is_written_before_any_probe_and_the_final_record_merges_them(tmp_path):
	runner = FakeRunner()
	ctx = _Ctx()
	rec = run_pipeline(
		{"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, ctx, deps(tmp_path, runner)
	)
	stages = [e["stage"] for e in ctx.events]
	assert stages[:3] == ["fetching", "mapping", "skeleton"] and "probing" in stages
	sk = next(e for e in ctx.events if e["stage"] == "skeleton")
	assert sk["record_rev"] == 1 and sk["share_token"] == rec.share_token
	assert [p.id for p in rec.probes] == ["p1", "p2", "p3"] and all(p.status == "ok" for p in rec.probes)
	assert rec.env.gpu.name == "L4" and rec.knob("gpu_type").current == "L4"
	assert rec.job.script_map.model is None or rec.job.script_map.model.shape_source == "modules"
	assert rec.costs.gpu_usd == pytest.approx(3 * 30 * 0.80 / 3600, rel=1e-3)
	# C's LocalRecordStore reads <share_token>.json
	stored = load_record((tmp_path / "records" / f"{rec.share_token}.json").read_text())
	assert stored.record_id == rec.record_id


def test_a_point_whose_override_cannot_be_applied_is_skipped_not_fatal(tmp_path, monkeypatch):
	import preflight.runner.pipeline as pl
	from preflight.recorder.overrides import OverrideError

	real = pl.apply_overrides

	def picky(files, sm, values):
		if values.get("seq_len") == 32:
			raise OverrideError("knob seq_len: no literal found at train.py:4")
		return real(files, sm, values)

	monkeypatch.setattr(pl, "apply_overrides", picky)
	runner = FakeRunner()
	rec = run_pipeline(
		{"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(), deps(tmp_path, runner)
	)
	by_id = {p.id: p for p in rec.probes}
	assert by_id["p1"].status == "ok" and by_id["p2"].status == "ok"
	assert by_id["p3"].status == "error" and by_id["p3"].error_code == "override.unsupported"
	assert [c[0] for c in runner.calls] == ["p1", "p2"]  # nothing ran for the skipped point


def test_training_oom_steps_up_and_the_smaller_points_follow_it(tmp_path):
	runner = FakeRunner({("p1", "L4"): ("oom", "oom")})
	rec = run_pipeline(
		{"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(), deps(tmp_path, runner)
	)
	assert [(c[0], c[1]) for c in runner.calls] == [
		("p1", "L4"),
		("p1", "L40S"),
		("p2", "L40S"),
		("p3", "L40S"),
	]
	assert rec.env.gpu.name == "L40S"


def test_eval_oom_reruns_with_eval_off_on_the_same_gpu(tmp_path):
	runner = FakeRunner({("p1", "L4"): ("oom", "oom.eval")})
	runner.outcomes[("p1", "L4")] = ("oom", "oom.eval")

	def flip(spec, gpu, *, should_stop):  # the re-run (eval off) fits
		if spec.probe_id == "p1" and spec.eval_batches == 0:
			runner.outcomes[("p1", "L4")] = ("ok", None)
		return FakeRunner.__call__(runner, spec, gpu, should_stop=should_stop)

	rec = run_pipeline(
		{"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(), deps(tmp_path, flip)
	)
	first = [c for c in runner.calls if c[0] == "p1"]
	assert [(c[1], c[2] > 0) for c in first] == [("L4", True), ("L4", False)]  # no bigger GPU
	ids = [p.id for p in rec.probes]
	assert (
		ids[:2] == ["p1-eval", "p1"]
		and rec.probes[0].error_code == "oom.eval"
		and rec.probes[1].status == "ok"
	)


def test_budget_stops_probing_and_keeps_what_was_measured(tmp_path):
	runner = FakeRunner(seconds=600.0)  # ten minutes a point on an L4: $0.13 each
	d = deps(tmp_path, runner, budget=lambda: Budget(usd=0.20))
	rec = run_pipeline({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(), d)
	assert [p.status for p in rec.probes] == ["ok", "error", "error"]
	assert all(p.error_code == "budget" for p in rec.probes[1:])


def test_probe_cache_answers_an_identical_point(tmp_path):
	runner = FakeRunner()
	cache = MemoryCache()
	req = {"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}
	run_pipeline(req, _Ctx(), deps(tmp_path, runner, cache=cache))
	n = len(runner.calls)
	rec = run_pipeline(req, _Ctx(), deps(tmp_path, runner, cache=cache))
	assert len(runner.calls) == n  # nothing ran again
	assert all(p.status == "cached" and p.cache_key for p in rec.probes)
	assert rec.costs.gpu_usd == 0


def test_flags_on_the_command_line_follow_the_probe_point(tmp_path):
	script = 'import argparse\np = argparse.ArgumentParser()\np.add_argument("--batch_size", type=int, default=8)\np.add_argument("--seq_len", type=int, default=64)\na = p.parse_args()\n'
	runner = FakeRunner()
	req = {"source": {"kind": "upload", "files": {"train.py": script}, "args": ["--batch_size", "16"]}}
	run_pipeline(req, _Ctx(), deps(tmp_path, runner))
	args = {c[0]: c[4] for c in runner.calls}
	assert args["p1"] == ["--batch_size", "16"] and args["p2"] == [
		"--batch_size",
		"8",
	]  # half of 16, not of 8


def test_unknown_sample_and_bad_upload_are_visitor_errors(tmp_path):
	from preflight.probes.local import InvalidSubmission

	for src in (
		{"kind": "sample", "uri": "sample://../../etc"},
		{"kind": "upload", "files": "nope"},
		{"kind": "zip"},
	):
		with pytest.raises(InvalidSubmission):
			run_pipeline({"source": src}, _Ctx(), deps(tmp_path, FakeRunner()))


def test_zoo_sample_maps_to_a_skeleton(tmp_path):
	runner = FakeRunner()
	rec = run_pipeline(
		{"source": {"kind": "sample", "uri": "sample://zoo/finetune_lora_hf"}}, _Ctx(), deps(tmp_path, runner)
	)
	assert rec.job.lens == "finetune" and rec.job.source.content_hash.startswith("sha256:")
	assert rec.probes and rec.probes[0].status == "ok"


# ---------------------------------------------------------------- real probes, through the queue


@pytest.mark.torch
def test_tiny_loop_runs_end_to_end_through_the_queue_on_cpu(tmp_path):
	pytest.importorskip("torch")
	store = InMemoryJobStore()
	sub = store.submit(upload("tiny_loop.py"), visitor="v1", idempotency_key="k1")
	d = deps(tmp_path, runner=None)
	from preflight.runner.pipeline import local_runner

	d.runner = local_runner
	assert run_one(store, "w1", lambda lease, ctx: run_job(lease, ctx, deps=d))
	assert store.status(sub.job_id) == "done"
	events = store.events(sub.job_id)
	stages = [e.stage for e in events]
	assert "skeleton" in stages and stages[-1] == "done"
	token = next(e.share_token for e in events if e.stage == "skeleton")
	rec = load_record(json.loads((tmp_path / "records" / f"{token}.json").read_text()))
	assert [p.status for p in rec.probes] == ["ok", "ok", "ok"]
	p1 = rec.probes[0]
	assert p1.step_time.p50_s > 0 and p1.step_log and p1.kernels is not None
	assert rec.env.backend == "cpu"


def test_cli_map_prints_the_script_map(capsys):
	from preflight.cli.main import main

	assert main(["map", str(FIX / "tiny_loop.py")]) == 0
	sm = json.loads(capsys.readouterr().out)
	assert sm["entrypoint"] == "tiny_loop.py" and any(k["id"] == "batch_size" for k in sm["knobs"])
	assert main(["map", str(FIX / "nope.py")]) == 2


@pytest.mark.torch
def test_cli_run_writes_a_record(tmp_path, capsys):
	pytest.importorskip("torch")
	from preflight.cli.main import main

	assert main(["run", str(FIX / "tiny_loop.py"), "--out", str(tmp_path), "--", "--unused", "1"]) == 0
	out = json.loads(capsys.readouterr().out)
	assert Path(out["record"]).parent == tmp_path and Path(out["record"]).exists()
	assert out["probes"][0]["status"] == "ok"


@pytest.mark.torch
def test_cli_run_can_insist_on_a_gpu(tmp_path, capsys):
	pytest.importorskip("torch")
	import torch

	from preflight.cli.main import main

	if torch.cuda.is_available():
		pytest.skip("this box has a GPU")
	assert main(["run", str(FIX / "tiny_loop.py"), "--out", str(tmp_path), "--require-gpu"]) == 3
	assert json.loads(capsys.readouterr().out)["backend"] == "cpu"


def test_eval_oom_after_a_step_up_reruns_on_the_bigger_gpu(tmp_path):
	state = {"L4": ("oom", "oom"), "L40S": ("oom", "oom.eval")}

	def run(spec, gpu, *, should_stop):
		r = FakeRunner(
			{spec.probe_id: state[gpu] if spec.probe_id == "p1" and spec.eval_batches else ("ok", None)}
		)
		out = r(spec, gpu, should_stop=should_stop)
		calls.append((spec.probe_id, gpu, spec.eval_batches > 0))
		return out

	calls = []
	rec = run_pipeline(
		{"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(), deps(tmp_path, run)
	)
	assert calls == [
		("p1", "L4", True),
		("p1", "L40S", True),
		("p1", "L40S", False),
		("p2", "L40S", False),
		("p3", "L40S", False),
	]
	assert [p.id for p in rec.probes][:2] == ["p1-eval", "p1"]


def test_the_ladder_respects_the_budget_and_every_attempt_is_paid_for(tmp_path):
	script = SCRIPT.replace("model = torch.nn.Linear(4, 4)", "class M(torch.nn.Module): pass\nmodel = M()")
	runner = FakeRunner(
		{
			("p1", "L4"): ("oom", "oom"),
			("p1", "L40S"): ("oom", "oom"),
			("p1", "A100-SXM-80GB"): ("oom", "oom"),
		},
		seconds=360.0,
	)
	d = deps(tmp_path, runner, budget=lambda: Budget(usd=0.30, wall_s=3600))
	rec = run_pipeline({"source": {"kind": "upload", "files": {"train.py": script}}}, _Ctx(), d)
	gpus = [c[1] for c in runner.calls if c[0] == "p1"]
	assert gpus == ["L4", "L40S"]  # A100 at $0.25 for six minutes would pass $0.30 spent so far
	paid = 360 * (0.80 + 1.95) / 3600
	assert rec.costs.gpu_usd == pytest.approx(paid, rel=1e-3)
	assert rec.costs.breakdown["discarded_attempts"] == pytest.approx(360 * 0.80 / 3600, rel=1e-3)


@pytest.mark.parametrize(("extra", "ok"), [({"gpu": "totally-fake"}, False), ({"gpu": "L40S"}, True),
	({"gpu": "L4", "constraints": {"gpu_types": ["L40S"]}}, False), ({"constraints": {"gpu_types": ["nope-9000"]}}, False)])  # fmt: skip
def test_visitor_gpu_choices_are_checked(tmp_path, extra, ok):
	from preflight.probes.local import InvalidSubmission

	req = {"source": {"kind": "upload", "files": {"train.py": SCRIPT}}, **extra}
	if ok:
		rec = run_pipeline(req, _Ctx(), deps(tmp_path, FakeRunner()))
		assert rec.env.gpu.name == "L40S"
	else:
		with pytest.raises(InvalidSubmission):
			run_pipeline(req, _Ctx(), deps(tmp_path, FakeRunner()))


def test_cache_hits_carry_this_points_labels_not_the_cached_runs(tmp_path):
	runner = FakeRunner()
	cache = MemoryCache()
	req = {"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}
	run_pipeline(req, _Ctx(), deps(tmp_path, runner, cache=cache))
	rec = run_pipeline(req, _Ctx(), deps(tmp_path, runner, cache=cache))
	assert all(p.started_at is None and p.cost.usd == 0 for p in rec.probes)
	assert rec.probes[1].knobs == {"batch_size": 4, "seq_len": 64}


def test_the_records_model_calls_count_in_its_cost(tmp_path):
	import preflight.runner.pipeline as pl
	from preflight.schema import ModelCall

	call = ModelCall(provider="meta", model="muse-spark-1.3", purpose="script_map", input_tokens=4937,
		output_tokens=3265, cost_usd=0.020047, at="2026-10-28T05:12:00Z")  # fmt: skip

	def fake_map(files, *, entrypoint, client):
		return pl.map_static(files, entrypoint=entrypoint), [call]

	pl_map = pl.map_script
	pl.map_script = fake_map
	try:
		rec = run_pipeline({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(),
			deps(tmp_path, FakeRunner(), spark=object()))  # fmt: skip
	finally:
		pl.map_script = pl_map
	assert rec.costs.llm_usd == pytest.approx(0.020047)
	assert rec.costs.breakdown["model_calls"] == pytest.approx(0.020047)
	assert rec.costs.total_usd == pytest.approx(rec.costs.gpu_usd + 0.020047)


def test_gpu_cost_and_the_sandbox_overhead_estimate_are_kept_apart(tmp_path):
	rec = run_pipeline({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(),
		deps(tmp_path, FakeRunner(seconds=36.0), overhead_usd_per_h=1.0))  # fmt: skip
	gpu = 3 * 36.0 * 0.80 / 3600  # three probes on the L4 at list price
	assert rec.costs.gpu_usd == pytest.approx(gpu, rel=1e-3)
	assert rec.costs.breakdown["sandbox_cpu_memory_estimate"] == pytest.approx(3 * 36.0 / 3600, rel=1e-3)
