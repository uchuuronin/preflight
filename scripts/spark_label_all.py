"""Spark Open labels the whole published corpus (Chiya, Nov 1, 6 PM ET: Baseten credit, up to $10).

The first run (scripts/spark_label_corpus.py, 59 of 65) labelled only the failures: the rows that ran out of memory
and the runs that diverged. This one labels every row of preflight-corpus-0.3.0, the file on Hugging Face, with the
same model and effort (`meta-models/muse-spark-open-1026`, effort low, structured output), asking one question per
kind of row:

- a memory row (kernels and hf, fitted or out of memory): which memory use is the largest part of its peak (for one
  that ran out of memory, which filled the GPU). B's rule label: the engine's largest memory term at the row's
  settings (engine/memory.breakdown), as in the first run; families with no analytical model (cnn, mlp) are
  labelled but have no rule label, so the agreement leaves them out.
- a divergence run: whether it stayed stable, and if not, how it failed. The existing label is the corpus's
  hindsight label (rule D9): stable, or the failure mode its reason names.

Rows go to the model in batches of the same kind (run ids r1..rN, never the corpus key), to keep the cost under
the budget; the rule labels and the hindsight labels are never in the prompt. Every call, failed or not, is logged
with its cost. The budget guard counts each call in flight at its worst case (the full output cap), so the total
can't pass the budget whatever comes back.

    python scripts/spark_label_all.py prepare --corpus <preflight-corpus-0.3.0.jsonl.gz>
        writes python/preflight/corpus/data/corpus_label_inputs_nov1.jsonl.gz (inputs and existing labels)
    python scripts/spark_label_all.py label --out out [--limit N]      # in remote-run.yml (BASETEN_API_KEY)
        writes out/spark_all_labels.jsonl (one line per row) and out/spark_all_calls.jsonl (one per call)
    python scripts/spark_label_all.py score --labels out/spark_all_labels.jsonl --calls out/spark_all_calls.jsonl
        writes python/preflight/corpus/reports/spark_labels_all_nov1.json and the provenance file
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark_label_corpus as first  # the first run's rules, evidence and model

ROOT = first.ROOT
INPUTS = ROOT / "python/preflight/corpus/data/corpus_label_inputs_nov1.jsonl.gz"
REPORT = ROOT / "python/preflight/corpus/reports/spark_labels_all_nov1.json"
PROVENANCE = ROOT / "python/preflight/corpus/provenance/spark_open_labels_all_nov1.jsonl"
FIRST_ROWS = ROOT / "python/preflight/corpus/reports/spark_labels_nov1_rows.jsonl"
MODEL = first.MODEL
BUDGET_USD = 9.50  # Chiya, Nov 1, 6 PM ET: up to $10 of Baseten credit; the guard stops short of it
BATCH = 8  # guess: rows per call, so one round of reasoning covers several rows
WORKERS = 6  # guess: calls in flight; each has its own client
MAX_OUTPUT = 12_000  # the first run's 8,192 cut 8 single-row answers off; a batch gets more room
TIMEOUT_S = 120.0

MEM_LABELS = first.OOM_LABELS
DIV_LABELS = ("stable", *first.DIV_LABELS)
MEM_DEFS = first.OOM_DEFS
DIV_DEFS = "stable: the loss improves or levels off without any of the failures below; " + first.DIV_DEFS


# ---------------------------------------------------------------- prepare (local, no calls)


def _mem_evidence(r: dict) -> dict:
	m = r.get("memory") or {}
	return {
		"family": r["config"]["family"],
		"settings": r["config"]["params"],
		"model_shape": r.get("model_shape"),
		"gpu": r.get("gpu"),
		"ran_out_of_memory": r.get("status") == "oom",
		"gpu_total_bytes": m.get("total"),
		"peak_allocated_bytes": m.get("peak_allocated"),
		"n_params": r.get("n_params"),
		"n_trainable": r.get("n_trainable"),
	}


def prepare(corpus: str) -> None:
	items = []
	with gzip.open(corpus, "rt") as fh:
		rows = [json.loads(line) for line in fh]
	for r in rows:
		rid = f"{r['kind']}/{r.get('gpu')}/{r['key']}"
		if r["kind"] in ("kernels", "hf"):
			items.append(
				{
					"row_id": rid,
					"task": "memory",
					"status": r.get("status"),
					"evidence": _mem_evidence(r),
					"rule": first._oom_rule(r),
				}
			)
		elif r["kind"] == "divergence":
			lab = r.get("label") or {}
			items.append(
				{
					"row_id": rid,
					"task": "divergence",
					"status": "diverged" if lab.get("diverged") else "stable",
					"evidence": first._div_evidence(r),
					"rule": first._div_rule(r) if lab.get("diverged") else "stable",
				}
			)
	INPUTS.parent.mkdir(parents=True, exist_ok=True)
	with open(INPUTS, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
		gz.write("".join(json.dumps(i, sort_keys=True) + "\n" for i in items).encode())
	counts = {t: sum(1 for i in items if i["task"] == t) for t in ("memory", "divergence")}
	no_rule = sum(1 for i in items if i["rule"] is None)
	print(f"wrote {INPUTS.relative_to(ROOT)}: {counts}; rows without a rule label: {no_rule}")


# ---------------------------------------------------------------- label (Baseten; remote-run.yml)


def _inputs() -> list[dict]:
	with gzip.open(INPUTS, "rt") as fh:
		return [json.loads(x) for x in fh]


def _schema(labels: tuple[str, ...]):
	# built with create_model: under `from __future__ import annotations` a nested class's Literal[labels] is a string
	# pydantic can't resolve (the fake client's test caught it)
	from pydantic import create_model

	one = create_model(
		"One",
		run=(str, ...),
		label=(Literal[labels], ...),  # type: ignore[valid-type]
		confidence=(Literal["low", "medium", "high"], ...),
		why=(str, ...),
	)
	return create_model("Batch", labels=(list[one], ...))  # type: ignore[valid-type]


def _messages(task: str, chunk: list[dict]) -> list[dict]:
	if task == "memory":
		ask = (
			"Each run below trained a model on one GPU. From its settings and numbers, which memory use is the largest "
			"part of its peak GPU memory? For a run that ran out of memory, which use filled the GPU. "
			f"Labels: {MEM_DEFS}"
		)
	else:
		ask = (
			"Each run below trained a small GPT. From its settings and loss numbers, did it stay stable, and if not, "
			f"how did it fail? Labels: {DIV_DEFS}"
		)
	runs = "\n".join(f"r{n + 1}: {json.dumps(i['evidence'], sort_keys=True)}" for n, i in enumerate(chunk))
	return [
		{
			"role": "system",
			"content": "You label ML training runs for a dataset. Judge each run on its own, only from its numbers. "
			"Return JSON matching the schema: for every run id, one label, your confidence, and one sentence of why.",
		},
		{"role": "user", "content": f"{ask}\n\nRuns:\n{runs}"},
	]


def _chunks(items: list[dict], size: int) -> list[tuple[str, list[dict]]]:
	"""Batches of one kind, the kinds interleaved in proportion, so a run the budget stops short still covers both
	(review, Nov 1)."""
	per = {}
	for task in ("memory", "divergence"):
		rows = [i for i in items if i["task"] == task]
		per[task] = [(task, rows[k : k + size]) for k in range(0, len(rows), size)]
	total = sum(len(v) for v in per.values())
	keyed = [((j + 0.5) / len(v), t, j) for t, v in per.items() for j in range(len(v))]
	return [per[t][j] for _, t, j in sorted(keyed)] if total else []


def _pilot(items: list[dict], limit: int) -> list[dict]:
	"""A pilot: the same share of each kind, spread over the file, so its cost per row says what the whole run
	will cost. At least one row of each kind (review, Nov 1: --limit 1 divided by zero, --limit 0 ran everything)."""
	if limit < 2:
		raise SystemExit("--limit needs at least 2 rows, one of each kind")
	mem = [i for i in items if i["task"] == "memory"]
	div = [i for i in items if i["task"] == "divergence"]
	k = min(limit - 1, max(1, round(limit * len(mem) / len(items))))
	return mem[:: max(1, len(mem) // k)][:k] + div[:: max(1, len(div) // (limit - k))][: limit - k]


def _run_id(text: str) -> str:
	"""'r3', 'R3', '3' and ' r3 ' are the same run (review, Nov 1: an answer keyed '3' was lost)."""
	t = text.strip().lower()
	return t if t.startswith("r") else f"r{t}"


def label(out_dir: str, limit: int | None, batch: int, workers: int, budget: float) -> None:
	from preflight.harness.spark import ModelClient

	items = _inputs()
	if limit is not None:
		items = _pilot(items, limit)
	chunks = _chunks(items, batch)
	probe = ModelClient.baseten_spark_open(max_output_tokens=MAX_OUTPUT)
	if probe.model != MODEL:
		raise SystemExit(f"expected {MODEL}, the client would call {probe.model} (SPARK_OPEN_MODEL)")
	rate_in, _cached, rate_out = probe.prices.get(MODEL, (1.25, 0.15, 4.25))
	# worst case per call: the full output cap and twice the estimated prompt, with one retry
	worst = 2 * (MAX_OUTPUT * rate_out + 2 * 600 * batch * rate_in) / 1e6
	cond = threading.Condition()
	state = {"spent": 0.0, "in_flight": 0, "stopped": None}
	local = threading.local()
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	rows_fh = open(out / "spark_all_labels.jsonl", "w")  # noqa: SIM115  (written from the workers, closed below)
	calls_fh = open(out / "spark_all_calls.jsonl", "w")  # noqa: SIM115

	def run(job: tuple[int, tuple[str, list[dict]]]) -> None:
		n, (task, chunk) = job
		with cond:
			# review, Nov 1: wait for calls in flight to settle rather than drop the chunk; once a chunk can't start
			# with nothing in flight, the budget is spent and every later chunk stops too
			while (
				state["stopped"] is None
				and state["in_flight"] > 0
				and state["spent"] + (state["in_flight"] + 1) * worst > budget
			):
				cond.wait()
			if state["stopped"] is not None or state["spent"] + (state["in_flight"] + 1) * worst > budget:
				state["stopped"] = n if state["stopped"] is None else min(state["stopped"], n)
				return
			state["in_flight"] += 1
		# re-check, Nov 1: everything from here to the release is in try/finally, so a failure outside the call (the
		# client, the call log) can't leave in_flight raised and hang every waiting worker; such a failure keeps the
		# whole reservation as spent
		cost, calls, got, err = worst, [], {}, None
		try:
			if not hasattr(local, "client"):
				local.client = ModelClient.baseten_spark_open(max_output_tokens=MAX_OUTPUT)
				local.client.timeout_s, local.client.deadline_s = TIMEOUT_S, 2.5 * TIMEOUT_S
			client = local.client
			before = len(client.calls)
			try:
				ans, _ = client.structured(
					_messages(task, chunk),
					_schema(MEM_LABELS if task == "memory" else DIV_LABELS),
					purpose="label",
					reasoning_effort="low",
				)
				for one in ans.labels:
					got.setdefault(
						_run_id(one.run),
						{"label": one.label, "confidence": one.confidence, "why": one.why[:300]},
					)
			except Exception as exc:  # a refusal, a timeout or a schema miss: logged, the rows unlabelled
				err = f"{type(exc).__name__}: {exc}"[:300]
			calls = [c.model_dump(mode="json") for c in client.calls[before:]]
			# a call that timed out records $0 though the provider may have billed it to the output cap: counted at
			# one whole generation's worst case, half of `worst` (review, Nov 1: worst / 4 under-counted it)
			cost = sum(c["cost_usd"] if c["ok"] or c["output_tokens"] else worst / 2 for c in calls)
		except BaseException as exc:
			err = err or f"worker: {type(exc).__name__}: {exc}"[:300]
			with (
				cond
			):  # review, Nov 1: a broken worker stops the run instead of paying for every queued chunk
				state["stopped"] = n if state["stopped"] is None else min(state["stopped"], n)
			raise
		finally:
			with cond:
				state["spent"] += cost
				state["in_flight"] -= 1
				cond.notify_all()
				for c in calls:
					calls_fh.write(json.dumps({"chunk": n, "task": task, "rows": len(chunk), **c}) + "\n")
				for k, item in enumerate(chunk):
					lab = got.get(f"r{k + 1}") or {"label": None, "error": err or "no answer for this run id"}
					rows_fh.write(
						json.dumps({"row_id": item["row_id"], "task": task, "chunk": n, **lab}) + "\n"
					)
				rows_fh.flush()
				calls_fh.flush()

	t0 = time.monotonic()
	failed = None
	try:
		with ThreadPoolExecutor(workers) as pool:
			list(pool.map(run, enumerate(chunks)))
	except Exception as exc:  # the summary is written whatever a worker raised; the run then fails
		failed = exc
	finally:
		rows_fh.close()
		calls_fh.close()
	done = [json.loads(x) for x in (out / "spark_all_labels.jsonl").read_text().splitlines()]
	summary = {
		"rows": len(items),
		"chunks": len(chunks),
		"rows_sent": len(done),
		"labelled": sum(1 for r in done if r.get("label")),
		"counted_usd": round(state["spent"], 4),
		"stopped_at_chunk": state["stopped"],
		"worst_call_usd": round(worst, 4),
		"budget_usd": budget,
		"wall_s": round(time.monotonic() - t0, 1),
	}
	summary["worker_error"] = None if failed is None else f"{type(failed).__name__}: {failed}"[:300]
	(out / "spark_all_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
	print(json.dumps(summary))
	if failed is not None:
		raise SystemExit(f"a worker failed: {summary['worker_error']}")


# ---------------------------------------------------------------- score (local)


def _agree(pairs: list[tuple]) -> dict:
	comp = [(a, b) for a, b in pairs if a is not None and b is not None]
	k, n = sum(1 for a, b in comp if a == b), len(comp)
	lo, hi = first._wilson(k, n) if n else (None, None)
	confusion: dict[str, dict[str, int]] = {}
	for a, b in comp:
		confusion.setdefault(a, {}).setdefault(b, 0)
		confusion[a][b] += 1
	return {
		"compared": n,
		"agree": k,
		"rate": k / n if n else None,
		"wilson_95": [lo, hi],
		"existing_vs_spark": confusion,
	}


def score(labels_path: str, calls_path: str, run: str) -> None:
	items = {i["row_id"]: i for i in _inputs()}
	got = [json.loads(x) for x in Path(labels_path).read_text().splitlines()]
	calls = [json.loads(x) for x in Path(calls_path).read_text().splitlines()]
	mem = [(items[r["row_id"]], r.get("label")) for r in got if r["task"] == "memory"]
	div = [(items[r["row_id"]], r.get("label")) for r in got if r["task"] == "divergence"]
	rep: dict = {
		"model": MODEL,
		"effort": "low",
		"batch": BATCH,
		"run": run,
		"corpus_rows": len(items),
		"rows_sent": len(got),
		"answered": sum(1 for r in got if r.get("label")),
		"memory": {
			"rows": len(mem),
			"no_rule_label": sum(1 for i, _ in mem if i["rule"] is None),
			"no_answer": sum(1 for _, b in mem if b is None),
			"all": _agree([(i["rule"], b) for i, b in mem]),
			"out_of_memory_rows": _agree([(i["rule"], b) for i, b in mem if i["status"] == "oom"]),
			"fitted_rows": _agree([(i["rule"], b) for i, b in mem if i["status"] != "oom"]),
		},
		"divergence": {
			"rows": len(div),
			"no_answer": sum(1 for _, b in div if b is None),
			"label": _agree([(i["rule"], b) for i, b in div]),
			# "other" includes "can't tell": left out of the yes-or-no comparison
			"diverged_or_not": _agree(
				[(i["rule"] != "stable", None if b in (None, "other") else b != "stable") for i, b in div]
			),
			# review, Nov 1: a non-finite loss stops the run early, so 37 of 44 divergences read straight off the
			# evidence; the runs whose loss stayed finite are the harder test, and "always stable" is the baseline
			"finite_loss_runs": _agree(
				[
					(i["rule"], b)
					for i, b in div
					if (i["evidence"].get("loss") or {}).get("first_non_finite_step") is None
				]
			),
			"always_stable_baseline": _agree([(i["rule"], "stable") for i, _ in div]),
			"mode_where_both_say_diverged": _agree(
				[(i["rule"], b) for i, b in div if i["rule"] != "stable" and b not in (None, "stable")]
			),
		},
	}
	allp = [(i["rule"], b) for i, b in mem + div]
	rep["overall"] = _agree(allp)
	rep["overall"].pop("existing_vs_spark")
	# the rows the first run labelled one at a time (73 matched by GPU and key): does batching change the answer?
	# its divergence question had no "stable" option, so some of that disagreement is the question, not the batch
	if FIRST_ROWS.exists():
		one = {
			r["row_id"]: r.get("label") for r in (json.loads(x) for x in FIRST_ROWS.read_text().splitlines())
		}
		same = []
		for r in got:
			_kind, gpu, key = r["row_id"].split("/", 2)
			prev = one.get(f"{gpu}/{key}")
			if prev is not None and r.get("label") is not None:
				same.append((prev, r["label"]))
		rep["batched_vs_first_run"] = _agree(same)
	rep["calls"] = len(calls)
	rep["failed_calls"] = sum(1 for c in calls if not c["ok"])
	rep["cost_usd"] = round(
		sum(c["cost_usd"] for c in calls), 4
	)  # what the calls recorded; timeouts record $0
	summary = Path(calls_path).with_name("spark_all_summary.json")
	if summary.exists():
		rep["guard"] = json.loads(summary.read_text())  # the guard's count, timeouts at their worst case
	rep["tokens"] = {
		"input": sum(c["input_tokens"] for c in calls),
		"output": sum(c["output_tokens"] for c in calls),
	}
	REPORT.write_text(json.dumps(rep, indent=1) + "\n")
	PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
	PROVENANCE.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in calls))
	print(json.dumps({k: v for k, v in rep.items()}, indent=1))


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	p = sub.add_parser("prepare")
	p.add_argument("--corpus", required=True)
	lab = sub.add_parser("label")
	lab.add_argument("--out", default="out")
	lab.add_argument("--limit", type=int, default=None)
	lab.add_argument("--batch", type=int, default=BATCH)
	lab.add_argument("--workers", type=int, default=WORKERS)
	lab.add_argument("--budget", type=float, default=BUDGET_USD)
	s = sub.add_parser("score")
	s.add_argument("--labels", required=True)
	s.add_argument("--calls", required=True)
	s.add_argument("--run", default="")
	a = ap.parse_args(argv)
	if a.cmd == "prepare":
		prepare(a.corpus)
	elif a.cmd == "label":
		if a.budget > BUDGET_USD:
			raise SystemExit(f"--budget {a.budget} is over the approved ${BUDGET_USD:.2f}")
		label(a.out, a.limit, a.batch, a.workers, a.budget)
	else:
		score(a.labels, a.calls, a.run)
	return 0


if __name__ == "__main__":
	sys.exit(main())
