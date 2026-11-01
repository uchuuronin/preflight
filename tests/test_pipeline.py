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
	# the re-run says it ran no eval pass (C's checks page scores memory on that)
	assert rec.probes[1].extras["eval_off"] == 1.0 and "eval_off" not in rec.probes[0].extras
	assert "eval_batch_size" not in rec.probes[1].knobs  # this script has no eval batch knob
	paid = sum(p.cost.usd for p in rec.probes)
	assert rec.costs.gpu_usd == pytest.approx(paid, abs=2e-6)  # the re-run is a kept probe, not a discard


def test_the_eval_off_rerun_keeps_the_records_knobs_so_the_engine_accepts_them(tmp_path):
	runner = FakeRunner({("p1", "L4"): ("oom", "oom.eval")})

	def flip(spec, gpu, *, should_stop):
		if spec.probe_id == "p1" and spec.eval_batches == 0:
			runner.outcomes[("p1", "L4")] = ("ok", None)
		return FakeRunner.__call__(runner, spec, gpu, should_stop=should_stop)

	script = SCRIPT.replace("eval_every = 100", "eval_every = 100\neval_batch_size = 32")
	rec = run_pipeline(
		{"source": {"kind": "upload", "files": {"train.py": script}}}, _Ctx(), deps(tmp_path, flip)
	)
	rerun = next(p for p in rec.probes if p.id == "p1")
	assert rerun.extras["eval_off"] == 1.0 and "eval_batch_size" not in rerun.knobs
	from preflight.engine.api import predict
	from preflight.schema import dump_record

	back = load_record(dump_record(rec))  # the record as stored still validates
	preds = predict(back, next(p for p in back.probes if p.id == "p1").knobs)
	# its knobs are valid ones: the engine doesn't refuse them (an eval batch of 0 made it abstain on
	# everything). In the record as a whole memory there abstains on p1-eval's OOM; C's checks page
	# holds p1 and p1-eval out together, so that match isn't there when it predicts p1
	assert not any("must be at least" in (e.reason or "") for e in preds.root.values())


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


@pytest.fixture
def spark_map_flag(monkeypatch):
	"""harness.spark_script_map on (off by default), restored for the tests after."""
	from preflight.runner import flags

	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.spark_script_map": True}))
	flags.reload()
	yield
	monkeypatch.delenv("PREFLIGHT_FLAGS")
	flags.reload()


@pytest.mark.parametrize("on", [False, True])
def test_job_code_goes_to_muse_spark_only_with_the_script_map_flag(tmp_path, monkeypatch, on):
	import preflight.runner.pipeline as pl
	from preflight.runner import flags

	sent = []

	def fake_map(files, *, entrypoint, client):
		sent.append(sorted(files))
		return pl.map_static(files, entrypoint=entrypoint), []

	monkeypatch.setattr(pl, "map_script", fake_map)
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.spark_script_map": on}))
	flags.reload()
	try:
		run_pipeline({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(),
			deps(tmp_path, FakeRunner(), spark=object()))  # a client is there, as when the Meta key is set  # fmt: skip
	finally:
		flags.reload()
	assert sent == ([["train.py"]] if on else [])


def test_the_records_model_calls_count_in_its_cost(tmp_path, spark_map_flag):
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


# ---------------------------------------------------------------- B's predictions in run_job


def _job(tmp_path, monkeypatch, *, flag_on, fill=None, timeout_s=None, ctx=None):
	import preflight.engine.api as engine
	from preflight.runner import flags
	from preflight.runner import pipeline as pl
	from preflight.runner.pipeline import run_job
	from preflight.runner.queue import Lease

	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"engine.memory": flag_on}))
	if fill is not None:
		monkeypatch.setattr(engine, "fill_predictions", fill)
	if timeout_s is not None:
		monkeypatch.setattr(pl, "PREDICT_TIMEOUT_S", timeout_s)
		real = pl.with_predictions
		monkeypatch.setattr(pl, "with_predictions", lambda rec, ctx: real(rec, ctx, timeout_s=timeout_s))
	d = deps(tmp_path, FakeRunner())
	lease = Lease(job_id="j1", token=1, worker="w", request={"source": {"kind": "upload", "files": {"train.py": SCRIPT}}},
		attempt=1, record_id=None, share_token=None, expires_at=0.0)  # fmt: skip
	ctx = ctx or _Ctx()
	flags.reload()
	try:
		rid = run_job(lease, ctx, deps=d)
	finally:
		flags.reload()  # never leave the flag cached for later tests
	stored = next((tmp_path / "records").glob("*.json"))
	return rid, load_record(stored.read_text()), ctx.events


def _predicting(events):
	return [e for e in events if e["stage"] == "predicting"]


def test_run_job_fills_bs_predictions_when_the_flag_is_on(tmp_path, monkeypatch):
	from preflight.engine.api import fill_predictions as real

	calls = []

	def fill(rec):
		calls.append(rec.record_id)
		return real(rec)

	_, rec, events = _job(tmp_path, monkeypatch, flag_on=True, fill=fill)
	assert len(calls) == 1 and rec.predictions.estimates  # filled once, and stored
	first, last = _predicting(events)
	assert first["message"] == "measured; predicting" and "record_rev" not in first
	assert last == events[-1] and last["message"] == "predicted" and last["record_rev"] >= 1


def test_run_job_leaves_predictions_alone_when_the_flag_is_off(tmp_path, monkeypatch):
	calls = []
	_, rec, events = _job(tmp_path, monkeypatch, flag_on=False, fill=lambda r: calls.append(1) or r)
	assert not calls and not rec.predictions.estimates
	assert events[-1]["message"] == "measured; no predictions yet"


def test_an_engine_failure_keeps_the_measured_record_and_says_so(tmp_path, monkeypatch):
	def boom(rec):
		raise ZeroDivisionError("bad fit")

	_, rec, events = _job(tmp_path, monkeypatch, flag_on=True, fill=boom)
	assert [p.id for p in rec.probes] == ["p1", "p2", "p3"] and not rec.predictions.estimates
	warn = [e for e in _predicting(events) if e.get("level") == "warn"]
	assert len(warn) == 1 and "predictions unavailable (ZeroDivisionError)" in warn[0]["message"]
	assert "bad fit" not in warn[0]["message"]  # internals stay in the log
	assert events[-1]["message"] == "measured; predictions unavailable"  # the last word doesn't hide it


def test_an_engine_result_that_doesnt_validate_is_not_stored(tmp_path, monkeypatch):
	def broken(rec):
		return rec.model_copy(update={"knobs": [*rec.knobs, rec.knobs[0]]})  # duplicate knob ids

	_, rec, events = _job(tmp_path, monkeypatch, flag_on=True, fill=broken)
	assert len({k.id for k in rec.knobs}) == len(rec.knobs)
	warn = [e for e in _predicting(events) if e.get("level") == "warn"]
	assert len(warn) == 1 and "predictions unavailable (ValidationError)" in warn[0]["message"]


def test_a_fill_that_hangs_is_given_up_on_and_the_measurements_are_stored(tmp_path, monkeypatch):
	import threading

	release = threading.Event()

	def stuck(rec):
		release.wait(5)
		return rec

	try:
		_, rec, events = _job(tmp_path, monkeypatch, flag_on=True, fill=stuck, timeout_s=0.2)
	finally:
		release.set()
	assert [p.id for p in rec.probes] == ["p1", "p2", "p3"]
	warn = [e for e in _predicting(events) if e.get("level") == "warn"]
	assert len(warn) == 1 and "predictions unavailable (TimeoutError)" in warn[0]["message"]
	assert events[-1]["message"] == "measured; predictions unavailable"


def test_a_record_whose_estimates_all_abstain_is_not_called_predicted(tmp_path, monkeypatch):
	from preflight.schema import Estimate

	def abstain(rec):
		e = Estimate(point=None, lo=None, hi=None, unit="s", status="abstain", model_id="m@1", source="exact",
			reason="the run's length isn't known")  # fmt: skip
		preds = rec.predictions.model_copy(update={"estimates": {"time.total": e}, "grids": []})
		return rec.model_copy(update={"predictions": preds})

	_, rec, events = _job(tmp_path, monkeypatch, flag_on=True, fill=abstain)
	assert rec.predictions.estimates["time.total"].point is None
	assert events[-1]["message"] == "measured; no predictions yet"


def test_a_cancel_during_the_fill_stores_nothing_more(tmp_path, monkeypatch):
	"""The store is asked before the filled record is stored, not the heartbeat's last word."""
	import preflight.engine.api as engine
	from preflight.engine.api import fill_predictions as real
	from preflight.runner import flags

	store = InMemoryJobStore()
	sub = store.submit({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, visitor="v1",
		idempotency_key="k1")  # fmt: skip

	def fill(rec):
		store.cancel(sub.job_id)  # the visitor cancels while the engine works
		return real(rec)

	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"engine.memory": True}))
	monkeypatch.setattr(engine, "fill_predictions", fill)
	d = deps(tmp_path, FakeRunner())
	flags.reload()
	try:
		run_one(store, "w1", lambda lease, ctx: run_job(lease, ctx, deps=d), heartbeat_s=3600)
	finally:
		flags.reload()
	assert store.status(sub.job_id) == "cancelled"
	saved = list((tmp_path / "records").glob("*.json"))
	assert saved  # the pipeline's own live saves
	for f in saved:
		assert not load_record(f.read_text()).predictions.estimates


def test_a_store_hiccup_while_confirming_keeps_the_finished_job(tmp_path):
	store = InMemoryJobStore()
	sub = store.submit({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, visitor="v1",
		idempotency_key="k1")  # fmt: skip
	real = store.heartbeat
	calls = []

	def flaky(lease):
		calls.append(1)
		if len(calls) == 1:
			raise ConnectionError("control plane hiccup")
		return real(lease)

	store.heartbeat = flaky
	d = deps(tmp_path, FakeRunner())
	assert run_one(store, "w1", lambda lease, ctx: run_job(lease, ctx, deps=d), heartbeat_s=3600)
	assert calls and store.status(sub.job_id) == "done"


def test_runner_flags_find_the_repos_flags_json_without_help(monkeypatch):
	"""No PREFLIGHT_ROOT: the search from runner/ up finds <repo>/flags.json (in the worker image,
	/opt/preflight/flags.json the same way)."""
	from preflight.runner import flags

	monkeypatch.delenv("PREFLIGHT_ROOT", raising=False)
	assert flags._root() == ROOT
	assert (ROOT / "flags.json").is_file()


def test_a_record_with_abstentions_survives_storage():
	"""Records are stored without nulls; an abstention (point None) must load back."""
	from preflight.schema import Estimate, FlightRecord, dump_record

	rec = load_record((ROOT / "schema" / "examples" / "finetune.json").read_text())
	e = Estimate(point=None, lo=None, hi=None, unit="s", status="abstain", model_id="m@1", source="exact",
		reason="the run's length isn't known")  # fmt: skip
	rec = rec.model_copy(
		update={"predictions": rec.predictions.model_copy(update={"estimates": {"time.total": e}})}
	)
	back = load_record(dump_record(FlightRecord.model_validate(rec.model_dump())))
	assert back.predictions.estimates["time.total"].point is None


def test_an_estimate_keeps_its_nulls_only_where_storage_would_drop_them():
	from preflight.schema import Estimate

	e = Estimate(point=None, lo=None, hi=None, unit="s", status="abstain", model_id="m@1", source="exact",
		reason="r")  # fmt: skip
	stored = e.model_dump(exclude_none=True)
	assert list(stored) == ["point", "lo", "hi", "unit", "status", "model_id", "source", "reason", "level"]
	assert e.model_dump(include={"unit"}) == {"unit": "s"}  # an explicit include is honoured
	assert "point" not in e.model_dump(exclude={"point"}, exclude_none=True)
	f = Estimate(point=2.0, lo=None, hi=3.0, unit="s", status="predicted", model_id="m@1", source="grid")
	assert f.model_dump(exclude_none=True)["lo"] is None and "reason" not in f.model_dump(exclude_none=True)


def test_the_worker_reads_the_same_flag_files_as_the_api():
	"""Flags are baked into both images: every flag file in the repo must ship in the runner's image
	and the API's (Chiya, Oct 30: flags.production.json beside flags.json), with the same env."""
	runner = (ROOT / "python" / "preflight" / "runner" / "modal_app.py").read_text()
	api = (ROOT / "api" / "app.py").read_text()
	files = sorted(p.name for p in ROOT.glob("flags*.json"))
	assert "flags.json" in files and set(files) <= {"flags.json", "flags.production.json"}, files
	for name in files:
		# the runner ships flags.json and its override by name; the API must name each one too
		assert name == "flags.json" or 'FLAG_OVERRIDE = FLAGS_SRC.parent / f"flags.{FLAG_ENV}.json"' in runner
		assert f'"{name}"' in api or f"/ {name!r}" in api.replace('"', "'"), f"api/app.py doesn't ship {name}"
	assert 'FLAG_ENV = "production"' in runner and '"PREFLIGHT_ENV": FLAG_ENV' in runner
	assert '"PREFLIGHT_ENV": "production"' in api


def test_generation_bound_lenses_get_a_longer_probe_time_limit():
	from preflight.harness.static import map_static
	from preflight.runner.pipeline import GENERATION_PROBE_TIMEOUT_S, _probe_timeout

	sm = map_static({"train.py": SCRIPT}, entrypoint="train.py")
	assert _probe_timeout(sm, 360) == 360
	assert _probe_timeout(sm.model_copy(update={"lens": "eval"}), 360) == GENERATION_PROBE_TIMEOUT_S
	assert _probe_timeout(sm.model_copy(update={"lens": "eval"}), 900) == 900  # never shorter than asked


def test_a_probe_run_on_a_fallback_gpu_is_not_cached_as_the_gpu_asked_for(tmp_path):
	class FellBack(FakeRunner):  # every point ran on an A10 when an L4 was asked for
		last_ran = "A10"

		def __call__(self, spec, gpu, *, should_stop):
			# the pipeline allowed these for the L4; after the fallback the job asks for the A10 itself
			assert spec.gpu_fallback == (("A10", "L40S") if gpu == "L4" else ())
			r = super().__call__(spec, "A10", should_stop=should_stop)
			return r

	cache = MemoryCache()
	req = {"source": {"kind": "upload", "files": {"train.py": SCRIPT}}, "gpu": "L4"}
	rec = run_pipeline(req, _Ctx(), deps(tmp_path, FellBack(), cache=cache))
	# the point that fell back isn't stored under the L4 key; the rest asked for the A10 and are its
	assert rec.probes[0].cache_key is None and len(cache._d) == len(rec.probes) - 1
	assert rec.env.gpu.name == "A10"


def test_after_a_fallback_the_job_stays_on_that_gpu_and_drops_cache_hits_from_the_other(tmp_path):
	cache = MemoryCache()
	req = {"source": {"kind": "upload", "files": {"train.py": SCRIPT}}, "gpu": "L4"}
	first = run_pipeline(req, _Ctx(), deps(tmp_path / "a", FakeRunner(outcomes={"p2": ("error", "x"), "p3": ("error", "x")}),
		cache=cache))  # fmt: skip
	assert [p.status for p in first.probes][:1] == ["ok"] and len(cache._d) == 1  # only p1 cached, on an L4

	class FellBack(FakeRunner):
		last_ran = None

		def __call__(self, spec, gpu, *, should_stop):
			self.last_ran = "A10"
			return super().__call__(spec, "A10", should_stop=should_stop)

	runner = FellBack()
	rec = run_pipeline(req, _Ctx(), deps(tmp_path / "b", runner, cache=cache))
	assert rec.probes[0].status == "error" and rec.probes[0].error_code == "gpu.fallback"
	assert [g for _, g, *_ in runner.calls] == ["A10"] * len(runner.calls)
	asked = [c for c in runner.calls]
	assert len(asked) >= 2 and rec.env.gpu.name == "A10"


class _FakeTriageClient:
	"""Stands in for Spark Open: answers each probe's log with a category and a quoted line."""

	def __init__(self):
		from datetime import UTC, datetime

		from preflight.schema import ModelCall

		self.calls, self._now, self._call = [], datetime.now(UTC), ModelCall
		self.seen = []

	def structured(self, msgs, schema, *, purpose, reasoning_effort=None):
		self.seen.append((purpose, reasoning_effort, msgs[-1]["content"]))
		call = self._call(provider="baseten", model="meta-models/muse-spark-open-1026", purpose=purpose,
			input_tokens=900, output_tokens=120, cost_usd=0.001635, at=self._now)  # fmt: skip
		self.calls.append(call)
		return schema(category="oom", summary="The forward pass ran out of GPU memory.",
			evidence_line="CUDA out of memory"), call  # fmt: skip


@pytest.mark.parametrize("on", [False, True])
def test_spark_open_triages_each_probe_log_only_with_its_flag(tmp_path, monkeypatch, on):
	from preflight.runner import flags

	class LoggedRunner(FakeRunner):
		def __call__(self, spec, gpu, *, should_stop):
			r = super().__call__(spec, gpu, should_stop=should_stop)
			r.log_tail = "step 1\ntorch.OutOfMemoryError: CUDA out of memory\n"
			return r

	client = _FakeTriageClient()
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.spark_open_triage": on}))
	flags.reload()
	try:
		d = deps(tmp_path, LoggedRunner(outcomes={"p2": ("oom", "oom.train")}))
		d.triage = client
		rec = run_pipeline({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, _Ctx(), d)
	finally:
		flags.reload()
	triage_calls = [c for c in rec.provenance if c.purpose == "triage"]
	if not on:
		assert not client.seen and not triage_calls
		return
	assert len(triage_calls) == len(client.seen) == len(rec.probes)  # one call per probe, all in provenance
	assert all(effort == "low" for _, effort, _ in client.seen)
	f = [f for f in rec.findings if f.source == "spark_open_triage"]
	assert [x.id for x in f] == ["triage.p2"] and 'log: "CUDA out of memory"' in f[
		0
	].detail  # only the probe that didn't finish
	assert rec.costs.llm_usd == pytest.approx(0.001635 * len(triage_calls))
