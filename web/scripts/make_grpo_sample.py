"""The GRPO sample (PF510): A's L4 smoke run of zoo/grpo_trl as a flight record.

Nothing here is made up or re-measured. The map and knobs come from A's static reader on the exact
script that ran (zoo/grpo_trl/train.py at the run's commit); the one probe is A's measured
recorder pass from that run, field for field (GitHub run 41260610897, `a-gt-grpo-l4-3`,
out/ground_truth_grpo_trl.json on the remote-runs branch); the predictions are B's engine
(`fill_predictions` with its RL targets on). One correction, from A's own log of the run:
truncation wasn't measured (the recorder didn't read the 512-token cap TRL sets in its generation
config), so the run's 0.0 is stored as not measured; the longest completion reached the cap.

	git show <run sha>:zoo/grpo_trl/train.py > /tmp/grpo_train.py
	git show origin/remote-runs:runs/<run dir>/out/ground_truth_grpo_trl.json > /tmp/gt.json
	uv run python web/scripts/make_grpo_sample.py /tmp/grpo_train.py /tmp/gt.json \\
		web/src/workspace/fixtures/grpo-l4.record.json

Swap in A's 50-step record when it's posted (the same three steps, from its own files).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from preflight.engine import api as engine
from preflight.harness.skeleton import skeleton_record
from preflight.harness.static import map_static
from preflight.probes.gpu_choice import usd_per_hour
from preflight.schema.record import (
	GPU,
	Costs,
	Env,
	GenerationStats,
	Memory,
	Probe,
	ProbeCost,
	RLSignals,
	Source,
	StepSample,
	StepTime,
)

RUN = "GitHub run 41260610897 (a-gt-grpo-l4-3, Nov 1)"
# Source: A's run reported the device as "NVIDIA L4"; B's table reads 23,689,363,456 bytes for a
# real L4 on Modal (B's request of Oct 27).
L4_BYTES = 23_689_363_456


def main(script: str, ground_truth: str, out_path: str) -> None:
	files = {"train.py": Path(script).read_text()}
	sm = map_static(files, entrypoint="train.py")
	gt = json.loads(Path(ground_truth).read_text())
	# A's pipeline folds the running model's shape into the map (untrusted.with_shape); this run's
	# recorder measured one field of it, the parameter count, so that one goes in and nothing else.
	params = int(gt["recorder"]["extras"]["params_total"])
	sm = sm.model_copy(update={"model": sm.model.model_copy(update={"param_count": params})})
	source = Source(
		kind="repo",
		uri="https://github.com/uchuuronin/preflight/tree/602edab1dbde6f8018368871d9c7b5790af3d249/zoo/grpo_trl",
		commit="602edab1dbde6f8018368871d9c7b5790af3d249",
		entrypoint="train.py",
	)
	rec = skeleton_record(sm, source, gpu="L4")
	r = gt["recorder"]
	rl = dict(r["rl"])
	gen = dict(r["generation"])
	# Not measured on this run (A's log): the cap wasn't read, so 0.0 means "unknown", not "none".
	rl["truncation_rate"] = None
	gen["truncation_rate"] = None
	gen["max_new_tokens"] = None
	current = {k.id: k.current for k in rec.knobs}
	probe = Probe(
		id="p0",
		knobs=current,
		status="ok",
		steps=r["steps"],
		duration_s=r["seconds"],
		memory=Memory(
			peak_allocated_bytes=r["peak_allocated_bytes"],
			peak_reserved_bytes=gt["bare"]["peak_reserved_bytes"],
			device_total_bytes=L4_BYTES,
		),
		step_time=StepTime(
			n_steps=r["steps"],
			p50_s=r["p50_s"],
			by_phase=r["by_phase"],
			first_step_s=r["step_log"][0]["wall_s"],
		),
		rl=RLSignals(**rl),
		generation=GenerationStats(**gen),
		extras={k: v for k, v in r["extras"].items() if v},
		step_log=[StepSample(**s) for s in r["step_log"]],
		# As A's runner prices a probe (pipeline._price): its GPU seconds at the GPU's hourly rate.
		cost=ProbeCost(gpu_seconds=r["seconds"], usd=round(r["seconds"] * usd_per_hour("L4") / 3600, 6)),
	)
	rec = rec.model_copy(
		update={
			"probes": [probe],
			# What making this record cost: the probe, as A's _merge totals it.
			"costs": Costs(gpu_usd=probe.cost.usd, breakdown={"probes": probe.cost.usd}),
			"env": Env(backend="cuda", provider="modal", gpu=GPU(name="L4", count=1, memory_bytes=L4_BYTES)),
		}
	)
	engine.FEATURES["rl"] = True
	rec = engine.fill_predictions(rec)
	rec = rec.model_copy(
		update={
			"record_id": "01JBGRPOL4SMOKE00000000000",
			"job": rec.job.model_copy(update={"title": "GRPO on GSM8K with Qwen2.5-0.5B, measured on an L4"}),
		}
	)
	data = json.loads(rec.model_dump_json(exclude_none=True))
	data.pop("share_token", None)  # share tokens never go in the repo
	with open(out_path, "w") as f:
		json.dump(data, f, indent="\t")
		f.write("\n")
	est = rec.predictions.estimates
	print(f"wrote {out_path}: {len(rec.knobs)} knobs, {len(rec.graph.nodes)} nodes; targets {sorted(est)}")


if __name__ == "__main__":
	main(*sys.argv[1:4])
