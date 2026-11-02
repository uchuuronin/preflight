"""Muse Spark's script map on the three repositories too big to read whole (D, Nov 2).

Run through remote-run.yml (it holds META_API_KEY and META_API_BASE):
	python scripts/d_bigrepo_spark_map.py
With runner.read_closure on in this process only, fetches each pinned repository (from
docs/validation/unseen-scripts.md: torchtune, lm-evaluation-harness, vLLM) at its commit with its
entry point, as a check-in would, and maps it with Muse Spark (harness/mapper.map_script), the
static map beside it. Writes out/bigrepo_map.json after each: what was read, the static and the
merged map's lens, framework and knobs, the model calls and their cost at list price. Stops
before a repository that could take the run past US$0.80 [cap US$1, Chiya's list].
"""

import json
import os
from pathlib import Path

os.environ["PREFLIGHT_FLAGS"] = json.dumps({"runner.read_closure": True})

from preflight.harness.mapper import map_script
from preflight.harness.spark import ModelClient
from preflight.harness.static import map_static, mapped_files
from preflight.runner.sources import fetch

CASES = [
	("torchtune", "https://github.com/pytorch/torchtune", "bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1",
		"recipes/lora_finetune_single_device.py"),
	("lm-evaluation-harness", "https://github.com/EleutherAI/lm-evaluation-harness",
		"ddd67220430a2470529f25fd5c05a576ca1057a0", "lm_eval/__main__.py"),
	("vllm", "https://github.com/vllm-project/vllm", "5df63a7203cdb9328d824a27b682e77c7b9928be",
		"vllm/benchmarks/throughput.py"),
]  # fmt: skip
STOP_USD = 0.80
WORST_CASE_USD = 0.15  # guess: a 60,000-character prompt and an 8,192-token reply at list price


def brief(sm) -> dict:
	return {"lens": sm.lens, "framework": sm.framework, "produced_by": sm.produced_by,
		"knobs": [(k.id, k.current) for k in sm.knobs], "risks": sm.risks[:6]}  # fmt: skip


def main() -> None:
	Path("out").mkdir(exist_ok=True)
	client = ModelClient.meta()
	rows, spent = [], 0.0
	for name, uri, sha, entry in CASES:
		if spent + WORST_CASE_USD > STOP_USD:
			rows.append({"repo": name, "skipped": f"cap: US${spent:.3f} spent"})
			break
		row: dict = {"repo": name, "entry": entry}
		before = len(client.calls)
		try:
			f = fetch({"kind": "repo", "uri": uri, "commit": sha, "entrypoint": entry})
			_, m = mapped_files(f.files, f.entrypoint)
			row["read"] = {"files": len(f.files), "partial": f.partial, "map_partial": m}
			row["static"] = brief(map_static(f.files, entrypoint=f.entrypoint))
			sm, _ = map_script(f.files, entrypoint=f.entrypoint, client=client)
			row["merged"] = brief(sm)
		except Exception as exc:  # recorded; the next repository still runs
			row["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
		calls = client.calls[before:]
		cost = round(sum(c.cost_usd for c in calls), 6)
		spent += cost
		row["calls"] = [{"ok": c.ok, "in": c.input_tokens, "out": c.output_tokens, "usd": c.cost_usd,
			"s": c.latency_s} for c in calls]  # fmt: skip
		row["cost_usd"] = cost
		row["last_error"] = client.last_error if calls and not all(c.ok for c in calls) else None
		rows.append(row)
		print(json.dumps({k: row.get(k) for k in ("repo", "cost_usd", "error")} | {
			"merged": (row.get("merged") or {}).get("lens")}), flush=True)  # fmt: skip
		Path("out/bigrepo_map.json").write_text(
			json.dumps({"spent_usd": round(spent, 6), "rows": rows}, indent=1)
		)
	Path("out/bigrepo_map.json").write_text(
		json.dumps({"spent_usd": round(spent, 6), "rows": rows}, indent=1)
	)
	print(f"spent US${spent:.4f} at list price")


if __name__ == "__main__":
	main()
