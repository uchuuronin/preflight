"""Check-in help, end to end through the production job store (PM, Nov 2 8:06 PM: check `c.checkin_help`
after A's redeploy, before it goes on). The two ways past a stopped check-in, each as the web sends them
(web/src/app/checkinTrouble.ts `withPackages`, `fromFile`):

    packages  an upload whose script imports einops (on Preflight's list, not in the probe image), sent
              again with `extra_packages: ["einops"]`. The first send (live site, job
              01M6FFGWR940ATWXXFPZ0BAP24, Nov 2) stopped at the import as it should; the resend from the
              same browser hit the visitor's daily limit (3 jobs), so it goes from here.
    scope     NVIDIA/DeepLearningExamples (5,403 files, about 63 MB of text: past the 2,000-file and 25 MB
              read limits) with `source.scope: "entrypoint"` on PyTorch/Classification/ConvNets/main.py.
              The check is the read: the job must get past "the source is larger than Preflight reads".
              What it does after is the script's own (it needs ImageNet and NVIDIA's dllogger).

    modal run --env main api/checkin_help_app.py::send --which packages
    modal run --env main api/checkin_help_app.py::send --which scope
    modal run --env main api/checkin_help_app.py::look --which packages

`look` is the diagnosis live check (PM, Nov 2 9:13 PM, item 3): it finds the newest record this script
made for `which` (by its description, among the newest records on the volume), opens its cockpit on the
live site in a headless browser and prints each finding's title and the line that says who found it, as
a visitor sees them. The share token stays in memory: it is never printed.

Spend: one job a run, under A's per-job cap (US$0.60); a job still running at the time limit is cancelled.
Each run has its own visitor (the job store's daily limit is per visitor). Printed: the job's status,
error code, each probe's status and error code, and cost; never a share token or a record's text.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("preflight-checkin-help-check")

POLL_S = 15
# Source: guess, as campaign_probe_app.py: a probe job is minutes; past this the job is cancelled.
JOB_TIMEOUT_S = 25 * 60

TRAIN = '''"""A tiny classifier on random data; uses einops to reshape. C's check-in help test (Nov 2)."""
import argparse
import torch
import einops

p = argparse.ArgumentParser()
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--steps", type=int, default=400)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
for step in range(a.steps):
	x = torch.randn(a.batch_size, 64, device=dev)
	x = einops.rearrange(x, "b (h d) -> b h d", h=4).reshape(a.batch_size, 64)
	y = torch.randint(0, 10, (a.batch_size,), device=dev)
	loss = torch.nn.functional.cross_entropy(model(x), y)
	opt.zero_grad()
	loss.backward()
	opt.step()
	if step % 50 == 0:
		print(step, round(loss.item(), 4))
'''

# Source: the GitHub API, Nov 2 (the master branch's head; the runner fetches by full hash, as the API
# resolves a branch before queuing, preflight_api/repos.py).
NVIDIA_COMMIT = "729963dd47e7c8bd462ad10bfac7a7b0b604e6dd"

REQUESTS = {
	"packages": {
		"source": {"kind": "upload", "files": {"train.py": TRAIN}, "entrypoint": "train.py"},
		"objective": {"kind": "min_cost", "description": None},
		"gpu": None,
		"description": "C's check-in help test: a script that needs einops, sent again with it",
		"extra_packages": ["einops"],
	},
	"scope": {
		"source": {
			"kind": "repo",
			"uri": "https://github.com/NVIDIA/DeepLearningExamples",
			"commit": NVIDIA_COMMIT,
			"entrypoint": "PyTorch/Classification/ConvNets/main.py",
			"scope": "entrypoint",
		},
		"objective": {"kind": "min_cost", "description": None},
		"gpu": None,
		"description": "C's check-in help test: a repo too big to read whole, from one file",
	},
}


SITE = "https://preflight-cqn.pages.dev"
# Source: a version from before this conversation's cutoff, pinned so the check doesn't drift.
PLAYWRIGHT = "playwright==1.47.0"
NEWEST = 80  # records to read when looking for this script's own


def summary(rec: dict | None) -> dict:
	if rec is None:
		return {}
	src = (rec.get("job") or {}).get("source") or {}
	return {
		"source": {k: src.get(k) for k in ("kind", "entrypoint", "scope", "extra_packages")},
		"lens": (rec.get("job") or {}).get("lens"),
		"probes": [
			{
				"id": p.get("id"),
				"status": p.get("status"),
				"error_code": p.get("error_code"),
				"missing_module": p.get("missing_module"),
				"steps": p.get("steps"),
			}
			for p in rec.get("probes") or []
		],
		# Muse Spark's diagnosis runs on a finished record now (harness.spark_diagnosis and the claim check
		# on since the PM's 7:25 PM merge): its findings, who wrote them, and whether Spark Open checked them.
		"findings": [
			{
				"source": f.get("source"),
				"code": f.get("code"),
				"severity": f.get("severity"),
				"checked_by_spark_open": any(
					(e or {}).get("metric") == "spark_open_check" for e in f.get("evidence") or []
				),
				"title": (f.get("title") or "")[:120],
			}
			for f in rec.get("findings") or []
		],
	}


@app.local_entrypoint()
def send(which: str = "packages", attempt: str = "a") -> None:
	sys.path.insert(0, str(ROOT / "api"))  # this runs on the caller's machine, from the repo
	import verify_race_app as vr

	if which not in REQUESTS:
		raise SystemExit(f"which: {', '.join(REQUESTS)}")
	svc = modal.Cls.from_name("preflight-runner", "JobStoreService", environment_name="main")()
	vol = modal.Volume.from_name("preflight-records", environment_name="main")
	t0 = time.monotonic()
	try:
		job = svc.submit.remote(
			REQUESTS[which],
			f"c.checkin_help.nov2.{which}.{attempt}",
			f"c-checkin-help-{which}-nov2-{attempt}",
		)
	except Exception as e:
		print(json.dumps({"which": which, "refused": f"{type(e).__name__}: {str(e)[:400]}"}, indent=1))
		raise SystemExit(1) from None
	jid = job.get("job_id")
	print(f"{which}: submitted", flush=True)
	st: dict = {}
	done = False
	try:
		while time.monotonic() - t0 < JOB_TIMEOUT_S:
			st = svc.status.remote(jid)
			if st.get("status") in ("done", "failed", "cancelled"):
				done = True
				break
			time.sleep(POLL_S)
	finally:
		if not done:
			try:
				svc.cancel.remote(jid)
			except Exception as e:
				print(f"cancel failed ({type(e).__name__}); A's job cap still applies", flush=True)
	token = st.get("share_token")
	rec = vr._read_record(vol, token) if token else None
	cost = vr._record_cost(rec) if rec and done else None
	out = {
		"which": which,
		"job_status": st.get("status") if done else "timed_out",
		"error_code": st.get("error_code"),
		"error": (st.get("error") or "")[:300],
		"minutes": round((time.monotonic() - t0) / 60, 1),
		"usd": round(cost, 4) if cost is not None else f"unknown, at most {vr.JOB_CAP_USD}",
		"record": summary(rec),
	}
	print(json.dumps(out, indent=1))


def _is_own(which: str, rec: dict) -> bool:
	"""Whether a record is this script's job for `which`, by its source (a record keeps no description)."""
	src = (rec.get("job") or {}).get("source") or {}
	if which == "packages":
		return (
			src.get("kind") == "upload"
			and src.get("entrypoint") == "train.py"
			and src.get("extra_packages") == ["einops"]
		)
	want = REQUESTS[which]["source"]
	return (
		src.get("uri") == want["uri"]
		and src.get("entrypoint") == want["entrypoint"]
		and src.get("scope") == "entrypoint"
	)


def _own_record(vol, which: str) -> tuple[str, dict] | None:
	"""The newest record of this script's job for `which`, among the newest on the volume."""
	entries = [e for e in vol.listdir("/") if e.path.endswith(".json")]
	entries.sort(key=lambda e: e.mtime, reverse=True)
	import verify_race_app as vr

	for e in entries[:NEWEST]:
		token = e.path.rsplit("/", 1)[-1][: -len(".json")]
		rec = vr._read_record(vol, token)
		if rec and _is_own(which, rec):
			return token, rec
	return None


@app.local_entrypoint()
def look(which: str = "packages") -> None:
	import subprocess

	sys.path.insert(0, str(ROOT / "api"))
	if which not in REQUESTS:
		raise SystemExit(f"which: {', '.join(REQUESTS)}")
	vol = modal.Volume.from_name("preflight-records", environment_name="main")
	found = _own_record(vol, which)
	if not found:
		raise SystemExit(f"{which}: no record of this script's among the newest {NEWEST}")
	token, rec = found
	print(json.dumps({"which": which, "record": summary(rec)["findings"]}, indent=1), flush=True)
	subprocess.run(["uv", "pip", "install", "--python", sys.executable, "-q", PLAYWRIGHT], check=True)  # noqa: S603, S607  fixed args
	subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=True,
		stdout=subprocess.DEVNULL)  # fmt: skip
	from playwright.sync_api import sync_playwright

	# The band lists the findings that sit off the map (Findings.tsx); one on a node shows in its
	# detail instead, so a finding missing from `shown` may be on the map.
	# Playwright's errors quote the page's address, which holds the share token: only the error's
	# type is printed.
	try:
		with sync_playwright() as pw:
			browser = pw.chromium.launch()
			page = browser.new_page(viewport={"width": 1440, "height": 900})
			page.goto(f"{SITE}/#/cockpit/{token}", wait_until="networkidle")
			page.wait_for_selector(".pf-dial-wrap", timeout=60_000)  # the cockpit is up
			page.wait_for_timeout(3_000)  # the findings band renders with the map's layout
			shown = page.evaluate(
				"""() => [...document.querySelectorAll('.pf-findings__list > li')].map((li) => ({
					title: (li.querySelector('b')?.textContent || '').trim().slice(0, 120),
					source: (li.querySelector('.pf-findings__source')?.textContent || '').trim(),
				}))"""
			)
			chip = page.evaluate(
				"() => (document.querySelector('.pf-findings-chip')?.textContent || '').trim()"
			)
			browser.close()
	except Exception as e:
		print(json.dumps({"which": which, "browser_failed": type(e).__name__}))
		raise SystemExit(1) from None
	print(json.dumps({"which": which, "shown": shown, "chip": chip[:200]}, indent=1))
