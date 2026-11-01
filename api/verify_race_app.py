"""Verify the first fix race's three candidates through the production job store (Chiya, Nov 1:
three L4 verify jobs, under US$1 in all), then pick the winner with the same checks the fix loop
uses (preflight_api/verify.py, options from fix/templates.py).

Each candidate is one `mode: verify` job on the Qwen 1.5B sample's script (`sample:lora-qwen`, A's
SAMPLE_REFS: zoo/finetune_lora_hf, the script the race patched) with its diff as `patch`, at the
script's own settings. A verify job runs the patched setting and the current one beside it, at
fixed seeds (A's pipeline._followup_plan: v, v-base, v-seed), so each record answers "did the fix
help, and did nothing else change?" on its own.

Spend: the jobs run one at a time, and the next is submitted only while the records' own costs
so far stay under the cap with room for one more like the dearest so far. GPUs: L4, or an A10 if
an L4 won't schedule (both under the standing yes), never anything dearer.

Share tokens are never printed: the log is published to the remote-runs branch.

	modal run --env main api/verify_race_app.py::verify_race
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("preflight-verify-race")

# Source: Chiya, Nov 1 (three L4 verify jobs, under US$1).
CAP_USD = 1.0
# Source: guess, a job waits up to 20 min for a GPU (A's fallback ladder) and runs three short probes.
JOB_TIMEOUT_S = 40 * 60
POLL_S = 15
RACE = ROOT / "api" / "preflight_api" / "fix" / "races" / "nov1_lora.json"
# Source: zoo/finetune_lora_hf/train.py's own settings, as the live check-in's record read them
# (record 01M6C75GZ9BR4250GCX0BYY5X8, Nov 1).
KNOBS = {
	"checkpoint_every": 500, "eval_every": 200, "gpu_type": "L4", "log_every": 10, "epochs": 1,
	"packing": False, "batch_size": 4, "eval_batch_size": 32, "eval_no_grad": False,
	"gradient_checkpointing": False, "quantization": "none", "seq_len": 1024, "lora_alpha": 32,
	"lora_dropout": 0.05, "lora_r": 16, "lr": 0.0001, "warmup_steps": 50, "attn": "sdpa",
	"grad_accum": 4, "precision": "bf16",
}  # fmt: skip


def _record_cost(rec: dict) -> float:
	"""GPU (every attempt, and the sandbox's billed lifetime when known), model calls, and the
	sandbox's CPU and memory estimate, which A keeps apart from the GPU figure (pipeline.py)."""
	c = rec.get("costs") or {}
	extra = (c.get("breakdown") or {}).get("sandbox_cpu_memory_estimate") or 0
	return float(c.get("gpu_usd") or 0) + float(c.get("llm_usd") or 0) + float(extra)


def _read_record(vol, token: str) -> dict | None:
	try:
		return json.loads(b"".join(vol.read_file(f"{token}.json")))
	except Exception:
		return None


def _pair(rec: dict) -> tuple[str | None, str | None]:
	"""The verify probe and its base, by A's ids (v<n>, v<n>-base)."""
	ids = [p.get("id") for p in rec.get("probes") or []]
	base = next((i for i in ids if i and i.endswith("-base")), None)
	return (base[: -len("-base")] if base else None), base


@app.local_entrypoint()
def verify_race(only: str = "") -> None:
	sys.path.insert(0, str(ROOT / "api"))
	from preflight_api import verify
	from preflight_api.fix.templates import verify_options

	race = json.loads(RACE.read_text())
	cands = [c for c in race["candidates"] if not only or c["id"] in only.split(",")]
	svc = modal.Cls.from_name("preflight-runner", "JobStoreService", environment_name="main")()
	vol = modal.Volume.from_name("preflight-records", environment_name="main")
	spent, dearest, results = 0.0, 0.0, []
	for c in cands:
		if spent + max(dearest, 0.0) >= CAP_USD:
			results.append({"id": c["id"], "status": "not_run", "why": f"cap: US${spent:.4f} spent"})
			continue
		req = {
			"source": {"kind": "record", "record_ref": "sample:lora-qwen"},
			"mode": "verify",
			"knobs": KNOBS,
			"patch": c["diff"],
			"gpu": "L4",
			"constraints": {"gpu_types": ["L4", "A10"]},
			"purpose": "c.fix_verify",
		}
		t0 = time.monotonic()
		try:
			job = svc.submit.remote(req, "c.fix_verify.nov1", f"c-fix-verify-{c['id']}-nov1-b")
		except Exception as e:
			results.append({"id": c["id"], "status": "refused", "why": f"{type(e).__name__}: {str(e)[:300]}"})
			continue
		jid = job.get("job_id")
		print(f"{c['id']} ({c['template_id']}): job {jid} submitted", flush=True)
		st: dict = {}
		while time.monotonic() - t0 < JOB_TIMEOUT_S:
			st = svc.status.remote(jid)
			if st.get("status") in ("done", "failed", "cancelled"):
				break
			time.sleep(POLL_S)
		else:
			svc.cancel.remote(jid)
			st = {**st, "status": "timed_out"}
		out = {"id": c["id"], "template_id": c["template_id"], "job_id": jid, "job_status": st.get("status"),
			"record_id": st.get("record_id"), "error_code": st.get("error_code"), "error": (st.get("error") or "")[:300],
			"minutes": round((time.monotonic() - t0) / 60, 1)}  # fmt: skip
		rec = _read_record(vol, st["share_token"]) if st.get("share_token") else None
		if rec is not None:
			cost = _record_cost(rec)
			spent += cost
			dearest = max(dearest, cost)
			vid, base = _pair(rec)
			opts = verify_options(c["template_id"])
			res = verify.check(rec, rec, before_probe=base, after_probe=vid, **opts) if vid and base else None
			probes = {p["id"]: {"status": p.get("status"), "gpu": (p.get("env") or {}).get("gpu"),
				"peak_gib": round((p.get("memory") or {}).get("peak_allocated_bytes", 0) / 2**30, 2)}
				for p in rec.get("probes") or []}  # fmt: skip
			finding = next(
				(f for f in rec.get("findings") or [] if f.get("id", "").startswith("verify")), None
			)
			out |= {"usd": round(cost, 4), "gpu": (rec.get("env") or {}).get("gpu", {}).get("name"), "probes": probes,
				"verify": res and {k: res[k] for k in ("status", "metric", "before", "after", "fixed_seed_loss_delta")},
				"checks": res and [(x["id"], x["ok"], x.get("allowed", False)) for x in res["checks"]],
				"a_finding": finding and {k: finding.get(k) for k in ("code", "severity", "title")}}  # fmt: skip
		results.append(out)
		print(json.dumps(out, indent=1), flush=True)
	passed = [r for r in results if (r.get("verify") or {}).get("status") == "passed"]
	# The winner: a passing fix with the lowest memory after it (the race's own target).
	winner = min(passed, key=lambda r: r["verify"]["after"] or float("inf")) if passed else None
	print(json.dumps({"spent_usd": round(spent, 4), "winner": winner and winner["id"],
		"results": [{k: r.get(k) for k in ("id", "template_id", "job_status", "usd")} | {"verify": (r.get("verify") or {}).get("status")} for r in results]}, indent=1))  # fmt: skip
