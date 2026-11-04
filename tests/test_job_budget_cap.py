"""The PM's audit (Nov 3): a job held $0.60 of the day's GPU cap but could spend about $1.45. Every attempt is now bounded
by the dollars the job has left, the worker reports the larger of its estimate and the runner's bill, and lease() checks
the day's cap before any GPU time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from preflight.probes.local import ProbeResult
from preflight.runner.queue import InMemoryJobStore
from preflight.schema import Env, Probe


class FullTimeout:
	"""Every attempt runs to its full time limit (the audit's worst case); p1 runs out of memory on the L4, then its
	eval runs out on the next GPU, so the eval-off re-run follows."""

	def __init__(self):
		self.calls = []

	def __call__(self, spec, gpu, *, should_stop):
		self.calls.append((spec.probe_id, gpu, spec.timeout_s, spec.eval_batches))
		status, code = "ok", None
		if spec.probe_id == "p1" and gpu == "L4":
			status, code = "oom", "oom"
		elif spec.probe_id == "p1" and spec.eval_batches:
			status, code = "oom", "oom.eval"
		s = spec.timeout_s
		probe = Probe(id=spec.probe_id, knobs=spec.knobs, status=status, error_code=code, steps=7 if status == "ok" else 1,
			duration_s=s, cost={"gpu_seconds": s, "usd": s * spec.gpu_usd_per_hour / 3600})  # fmt: skip
		return ProbeResult(probe=probe, env=Env(backend="cuda", provider="modal", gpu={"name": gpu, "memory_bytes": 24e9}),
			summary={}, kernels=None, arch={"n_layers": 2, "source": "modules"}, trace_file=None, log_tail="",
			workdir=Path("."))  # fmt: skip


def test_no_job_spends_past_its_budget_even_when_every_attempt_runs_to_its_limit(tmp_path):
	from preflight.runner.pipeline import JOB_BUDGET_USD, _Ctx, run_pipeline
	from preflight.runner.worker import _spent
	from test_pipeline import SCRIPT, deps

	runner = FullTimeout()
	ctx = _Ctx()
	# production's sandbox overhead estimate (modal_app.SANDBOX_OVERHEAD_USD_PER_H): without the fix this job spends
	# about $0.76 of a $0.60 budget (L4 to its OOM, the L40S, then the unchecked eval-off re-run)
	d = deps(tmp_path, runner, overhead_usd_per_h=0.95)
	rec = run_pipeline({"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}, ctx, d)
	gpus = [g for pid, g, _, _ in runner.calls if pid == "p1"]
	assert gpus[0] == "L4" and len(gpus) >= 2  # it stepped up after the OOM, and the eval-off re-run followed
	assert _spent(ctx) <= JOB_BUDGET_USD + 1e-6, (_spent(ctx), runner.calls)
	assert any(p.error_code == "budget" for p in rec.probes)  # the rest is skipped, saying why


def test_the_worker_reports_the_runners_bill_when_it_is_larger():
	from types import SimpleNamespace

	from preflight.runner.worker import _spent

	b = SimpleNamespace(gpu_usd=0.20, overhead_usd=0.05, billed_usd=0.41)
	assert (
		_spent(SimpleNamespace(budget=b)) == 0.46
	)  # the bill (GPU only) with the overhead estimate beside it
	b.billed_usd = 0.10
	assert _spent(SimpleNamespace(budget=b)) == 0.25
	b.billed_usd = None
	assert _spent(SimpleNamespace(budget=b)) == 0.25


REQ = {"source": {"kind": "sample", "uri": "sample://zoo/x"}}


def test_lease_refuses_when_the_days_spend_passed_the_cap_after_the_job_was_queued():
	t = [1_762_200_000.0]
	cap = [2.0]
	store = InMemoryJobStore(clock=lambda: t[0], daily_cap_usd=lambda day: cap[0])
	store.submit(REQ, visitor="a", idempotency_key="1")
	queued = store.submit({**REQ, "n": 2}, visitor="b", idempotency_key="2")
	lease = store.lease("w")
	store.complete(lease, record_id="r", spent_usd=1.7)  # the day's real spend passed what the holds promised
	assert store.lease("w") is None  # no GPU time for the queued job
	assert store.status(queued.job_id) == "failed"
	ev = store.events(queued.job_id)[-1]
	assert ev.stage == "failed" and ev.error_code == "limit.daily_spend" and ev.retryable is False


def test_lease_waits_while_running_jobs_hold_the_room_and_leases_once_it_comes_back():
	t = [1_762_200_000.0]
	store = InMemoryJobStore(clock=lambda: t[0], daily_cap_usd=lambda day: 1.5, max_running=4)
	store.submit(REQ, visitor="a", idempotency_key="1")
	store.submit({**REQ, "n": 2}, visitor="b", idempotency_key="2")
	first = store.lease("w")
	store.complete(first, record_id="r", spent_usd=0.5)
	assert store.lease("w") is not None  # 0.5 spent + 0.6 for it fits under 1.5


def test_lease_leaves_a_job_queued_while_running_jobs_hold_the_room():
	t = [1_762_200_000.0]
	store = InMemoryJobStore(clock=lambda: t[0], daily_cap_usd=lambda day: 2.0, max_running=4)
	for i, v in enumerate("abc"):
		store.submit({**REQ, "n": i}, visitor=v, idempotency_key=str(i))
	assert store.lease("w") is not None and store.lease("w") is not None
	store._daily[next(iter(store._daily))]["spent"] = 0.7  # spend reported meanwhile: 0.7 + 3 x 0.6 > 2.0
	assert store.lease("w") is None and store.status(store._order[0]) == "queued"  # waits, not refused


def test_a_retry_that_no_longer_fits_fails_at_lease():
	t = [1_762_200_000.0]
	cap = [1.3]
	store = InMemoryJobStore(clock=lambda: t[0], daily_cap_usd=lambda day: cap[0])
	s = store.submit(REQ, visitor="a", idempotency_key="1")
	lease = store.lease("w")
	store.fail(
		lease, error="worker lost", retryable=True, error_code="internal", spent_usd=0.75
	)  # requeued: attempt 2
	assert store.status(s.job_id) == "queued"
	assert store.lease("w") is None  # 0.75 + 0.60 > 1.3: the retry fails before any GPU time
	assert store.status(s.job_id) == "failed" and store.events(s.job_id)[-1].error_code == "limit.daily_spend"


def test_the_worker_image_carries_the_two_settings():
	import ast

	src = (Path(__file__).resolve().parents[1] / "python/preflight/runner/modal_app.py").read_text()
	node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
		and any(getattr(t, "id", None) == "WORKER_SETTINGS" for t in n.targets))  # fmt: skip
	assert ast.literal_eval(node.value) == {
		"PREFLIGHT_DAILY_GPU_CAP_USD": "15",
		"PREFLIGHT_JUDGING_TO": "2026-11-10",
	}
	assert "**WORKER_SETTINGS" in src


def test_a_failed_job_reports_its_bill_with_the_overhead_beside_it():
	from types import SimpleNamespace

	from preflight.runner.worker import _spent

	b = SimpleNamespace(gpu_usd=0.10, overhead_usd=0.05, billed_usd=None, bill=lambda: 0.30)
	assert _spent(SimpleNamespace(budget=b)) == 0.35  # read live, though the job never reached billed_usd


def test_one_refusal_fails_every_queued_job_that_cannot_fit_today():
	t = [1_762_200_000.0]
	cap = [3.0]
	store = InMemoryJobStore(clock=lambda: t[0], daily_cap_usd=lambda day: cap[0], max_running=1)
	ids = [
		store.submit({**REQ, "n": i}, visitor=v, idempotency_key=str(i)).job_id for i, v in enumerate("abc")
	]
	lease = store.lease("w")
	store.complete(lease, record_id="r", spent_usd=2.6)
	assert store.lease("w") is None
	assert [store.status(j) for j in ids[1:]] == ["failed", "failed"]
