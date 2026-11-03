"""The Checks page's "Preflight vs asking Muse Spark" data (web/src/coverage/vs_model.json; Chiya and C, Nov 3).

	uv run --extra probe python scripts/vs_model_from_final.py > web/src/coverage/vs_model.json

Per script, the six full runs `docs/validation/preflight-vs-model.md` compares (the four zoo full runs, the zoo LoRA,
BEIR): the full run's steady step time and peak reserved memory, Preflight's prediction at the full run's own values
(today's engine, production flags: the same scoring as the final table and web/src/coverage/coverage.json), Muse
Spark's answer from the code alone (`ask-model@1`, `python/preflight/corpus/reports/ask_model*_nov2.json`), and each
prediction over what the run did (a ratio: 1 is exact, 2 is twice too high, 0.5 half). No unseen-set lines (Chiya,
Nov 3): BEIR is here as the retrieval script whose full run ran out of memory, not as a harnessing count.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from preflight.corpus import validate as V
from preflight.engine import api

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "docs" / "validation" / "final"
REPORTS = ROOT / "python" / "preflight" / "corpus" / "reports"
GIB = 2**30
SCRIPTS = {  # manifest script id -> (label, Muse Spark's answer file, notes)
	"gnn_pyg": ("GNN (PyG GraphSAGE)", "ask_model_zoo_nov2.json", []),
	"retrieval_st": ("Retrieval (sentence-transformers)", "ask_model_zoo_nov2.json", []),
	"vision_timm": ("Vision (timm ResNet-50)", "ask_model_zoo_nov2.json",
		["the full run got an A10G, about 11% slower than the A10 both sides predicted for"]),
	"pretrain_nanogpt": ("Pretrain (nanoGPT)", "ask_model_zoo_nov2.json",
		["the full run got an A10G, about 11% slower than the A10 both sides predicted for"]),
	"finetune_lora_hf": ("LoRA fine-tune (zoo)", "ask_model_nov2.json",
		["ran out of memory at its first eval (step 200), so its full peak was never seen; its step time was used "
		"to calibrate Preflight's step band, so Preflight's step isn't a fair test and is left out"]),
	"unseen-7": ("Retrieval (BEIR)", "ask_model_nov2.json",
		["ran out of memory on the A10 before its first step; Muse Spark was asked for the L4, a 24 GB card like the "
		"A10; Preflight had no probe of it (its only check-in got no GPU), so it gave no numbers"]),
}  # fmt: skip


def _num(x, nd=4):
	return None if x is None else round(float(x), nd)


def _est(e, scale=1.0):
	if e is None or e.status == "abstain" or e.point is None:
		return {"status": "abstain", "reason": getattr(e, "reason", None)} if e is not None else None
	return {
		"point": _num(e.point / scale),
		"lo": _num(e.lo / scale),
		"hi": _num(e.hi / scale),
		"status": e.status,
	}


def _ratio(pred, actual):
	return _num(pred / actual, 4) if pred and actual else None


def main() -> int:
	manifest = json.loads((FINAL / "main.json").read_text())
	flags = V.production_flags(ROOT)
	token = api._FEATURES_NOW.set(api.features_from(lambda k: flags.get(k, False)))
	answers = {}
	for f in {v[1] for v in SCRIPTS.values()}:
		for row in json.loads((REPORTS / f).read_text())["rows"]:
			answers[row["id"]] = row
	rows = []
	try:
		for run in manifest["runs"]:
			sid = run.get("script")
			if sid not in SCRIPTS or any(r["script"] == sid for r in rows):
				continue
			label, _, notes = SCRIPTS[sid]
			gt = V._load(FINAL / run["ground_truth"]) if run.get("ground_truth") else {}
			bare = gt.get("bare") or {}
			ran_out = V.ran_out(gt)
			step = V.steady_step_s(gt) if not ran_out or sid == "finetune_lora_hf" else None
			peak = None if ran_out else (bare.get("peak_reserved_bytes") or 0) / GIB or None
			pre = {}
			if run.get("record") and (FINAL / run["record"]).exists():
				rec = V._record(FINAL / run["record"])
				pred = api.predict(rec, run.get("values") or None).root
				pre = {"step_s": None if "step" in (run.get("calibrates") or ()) else _est(pred.get("time.step")),
					"peak_gib": _est(pred.get("memory.peak"), GIB),
					"p_oom": _num(pred["risk.oom"].point, 3) if pred.get("risk.oom") and pred["risk.oom"].status != "abstain" else None}  # fmt: skip
			else:
				pre = {"step_s": None, "peak_gib": None, "p_oom": None, "why": "no probe ran"}
			a = (answers.get(sid) or {}).get("answer") or {}
			muse = {"step_s": a.get("step_time_s") if a.get("step_applies") else None,
				"peak_gib": a.get("peak_memory_gib"), "p_oom": a.get("p_oom")}  # fmt: skip
			ps = (pre.get("step_s") or {}).get("point")
			pp = (pre.get("peak_gib") or {}).get("point")
			rows.append({
				"script": "beir" if sid == "unseen-7" else sid,
				"label": label, "gpu": gt.get("gpu") or run.get("values", {}).get("gpu_type"),
				"full_run": {"step_s": _num(step), "peak_gib": _num(peak, 2), "ran_out_of_memory": ran_out,
					"reserved_when_it_ran_out_gib": _num((bare.get("peak_reserved_bytes") or 0) / GIB, 2) if ran_out else None,
					"run": run.get("$ground_truth", "")[:160]},
				"preflight": pre, "muse_spark": muse,
				"ratio": {"preflight_step": _ratio(ps, step), "muse_step": _ratio((muse["step_s"] or {}).get("point"), step),
					"preflight_peak": _ratio(pp, peak), "muse_peak": _ratio((muse["peak_gib"] or {}).get("point"), peak)},
				"notes": notes,
			})  # fmt: skip
	finally:
		api._FEATURES_NOW.reset(token)
	step_r = [r["ratio"] for r in rows if r["ratio"]["muse_step"] and r["ratio"]["preflight_step"]]

	def off(x):
		return max(x, 1 / x)

	muse_rng = [min(off(r["muse_step"]) for r in step_r), max(off(r["muse_step"]) for r in step_r)]
	pre_rng = [
		min(abs(r["preflight_step"] - 1) for r in step_r),
		max(abs(r["preflight_step"] - 1) for r in step_r),
	]
	summary = (f"Too few to rank formally. On step time, on the {len(step_r)} full runs both sides could be scored on, "
		f"Muse Spark from the code alone was off by {muse_rng[0]:.1f}x to {muse_rng[1]:.0f}x (every one too slow), "
		f"Preflight by {round(100 * pre_rng[0])}% to {round(100 * pre_rng[1])}% (two low, two high). On peak memory, Preflight "
		"was within 10% on 2 of 4 and Muse Spark on 0 of 4; on running out of memory both caught the LoRA eval and both "
		"missed BEIR.")  # fmt: skip
	commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", "python/preflight/engine"], cwd=ROOT,
		capture_output=True, text=True).stdout.strip() or None  # fmt: skip
	out = {
		"generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
		"engine_commit": commit,
		"source": "docs/validation/preflight-vs-model.md; scripts/vs_model_from_final.py",
		"muse_spark": {
			"model": "muse-spark-1.3",
			"prompt": "ask-model@1",
			"prompt_hash": "211e2e084350",
			"given": "the repo's files (as Preflight's script map gets them), the command and the GPU; nothing measured",
		},
		"ratio_means": "prediction over what the full run did: 1 is exact, above 1 too high, below 1 too low",
		"summary": summary,
		"caveats": [
			"Too few to rank: 4 scripts for step time and memory, 6 for running out of memory; the hit counts' "
			"Wilson 95% intervals overlap (2 of 4: 15% to 85%; 0 of 4: 0% to 49%).",
			"Muse Spark's 90% intervals covered slightly more often than Preflight's on these four (4 of 8 against "
			"3 of 8, memory and step together).",
			"Two of the four full runs (vision, nanoGPT) got an A10G, about 11% slower than the A10 both sides "
			"predicted for.",
			"The memory comparison is of unlike quantities on Preflight's side: its point includes the CUDA "
			"context and the full run's reserved peak doesn't, so each Preflight memory error reads about 0.3 GB high.",
			"The zoo LoRA run calibrated Preflight's step band, so its Preflight step isn't scored; it and BEIR ran "
			"out of memory, so their full peaks were never seen.",
			"The GRPO zoo script is left out: its sample folder held that run's own record, which Muse Spark read.",
		],
		"rows": rows,
	}
	json.dump(out, sys.stdout, indent="\t")
	sys.stdout.write("\n")
	return 0


if __name__ == "__main__":
	sys.exit(main())
