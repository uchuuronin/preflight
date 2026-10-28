"""Fly whole live-guide tours against real Muse Spark and measure them.

Runs the real POST /v1/guide/turns route (typed input, signed state, ledger, guards, number
check) inside a Modal container, the place the production API calls Muse Spark from, and
plays the visitor's browser: it runs the browser tools the guide asks for, sends their typed
results back, applies every move the guide proposes, declines nothing and approves the first
spending tool once (so the "measuring isn't open yet" path is exercised too).

Measured per round: wall-clock latency of the model call (seen from the API), input, output
and cached tokens, cost at the guide's price table, tools called, withheld replies and any
fallback. Per tour: rounds, decisions, tokens and dollars. These set muse_spark.TIMEOUT_S,
MAX_OUTPUT_TOKENS and the ledger's caps (docs/decisions/guide-limits.md).

The browser tools here are a Python stand-in for web/src/guide/liveGuide.ts that reads the
record's grids exactly at grid points (every knob choice is a grid point, so no interpolation
is needed). Its results have the same shape and about the same size as the browser's; the
numbers can differ slightly from the cockpit's where the TS provider extrapolates.

Through GitHub Actions (the key never leaves Modal's "preflight-meta" secret):

	gh workflow run remote-run.yml -f name=c-guide-tour -f ref=ws-c -f timeout=30 \\
		-f cmd="modal run api/scripts/guide_tour.py --tours 2"

Writes out/guide_tour.json (every round) and prints a summary. The tour itself lives in
tour_play.py, which also runs locally against a stand-in model (see its docstring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tour_play, next to this file

# Locally (on the GitHub runner) the repo root; inside the container this file sits at /root and
# the image's local files are already baked in, so the path is never read there.
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root/repo")
SAMPLE_FILES = [
	"web/src/workspace/fixtures/dev-finetune.record.json",
	"schema/examples/finetune.json",
	"schema/examples/pretrain.json",
	"schema/examples/agent.json",
]

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install("fastapi>=0.115", "pydantic>=2.8,<3", "httpx>=0.27", "numpy>=1.26")
	.env({"PREFLIGHT_ROOT": "/root/repo", "PREFLIGHT_FLAGS": json.dumps({"c.guide.live": True})})
	.add_local_file(ROOT / "flags.json", "/root/repo/flags.json")
	.add_local_dir(ROOT / "python" / "preflight", "/root/repo/python/preflight")
	.add_local_dir(ROOT / "api" / "preflight_api", "/root/repo/api/preflight_api")
	.add_local_file(Path(__file__).resolve().parent / "tour_play.py", "/root/repo/api/scripts/tour_play.py")
)
for rel in SAMPLE_FILES:
	image = image.add_local_file(ROOT / rel, f"/root/repo/{rel}")

app = modal.App("preflight-guide-tour", image=image)


@app.function(secrets=[modal.Secret.from_name("preflight-meta")], timeout=1800, cpu=1.0)
async def play(samples: list[str], tours: int, aim: str, variants: tuple[str, ...] = ("current",)) -> dict:
	sys.path[:0] = ["/root/repo/api/scripts", "/root/repo/api", "/root/repo/python"]
	from tour_play import run_tours

	return await run_tours(Path("/root/repo"), samples, tours, aim, variants)


@app.local_entrypoint()
def main(
	tours: int = 2,
	samples: str = "lora-7b,lora-qwen,pretrain,agent",
	aim: str = "min_cost",
	variants: str = "current",
) -> None:
	from tour_play import report

	res = play.remote(
		[s for s in samples.split(",") if s], tours, aim, tuple(v for v in variants.split(",") if v)
	)
	Path("out").mkdir(exist_ok=True)
	Path("out/guide_tour.json").write_text(json.dumps(res, indent="\t") + "\n")
	print(report(res))
