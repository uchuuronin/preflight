"""One real Muse Spark round trip: verify the wire format, measure latency and token usage.

Runs in GitHub Actions with the key from repository secrets (never printed). Makes a handful
of small calls (about two cents in total):
1. a plain Responses request, to check the endpoint, model name and usage fields;
2. the guide's real first round (instructions, tools built from the lora-7b sample, the start
   event), repeated N times, to check that the model calls tools and to measure latency;
3. if /responses isn't served, one /chat/completions request, to learn the format to switch to.

Prints one JSON summary, and the same as GitHub annotations (readable through the API), so
the result can be checked without the logs.

	cd api && uv run python scripts/muse_spark_smoke.py --rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preflight_api.guide import muse_spark
from preflight_api.guide.prompt import INSTRUCTIONS, language_line
from preflight_api.guide.record_view import RecordView
from preflight_api.guide.tools import build_tools

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "web/src/workspace/fixtures/dev-finetune.record.json"


def redact(text: str) -> str:
	key = os.environ.get("META_API_KEY", "")
	return text.replace(key, "<key>") if key else text


def shape(body: dict) -> dict:
	"""The response's structure without its content."""
	out = body.get("output")
	return {
		"top_keys": sorted(body.keys()),
		"output_types": [o.get("type") for o in out] if isinstance(out, list) else None,
		"usage_keys": sorted((body.get("usage") or {}).keys()),
		"model": body.get("model"),
	}


async def raw_post(
	client: httpx.AsyncClient, path: str, payload: dict
) -> tuple[int, dict | str, float, str | None]:
	base = os.environ["META_API_BASE"].rstrip("/")
	headers = {"Authorization": f"Bearer {os.environ['META_API_KEY']}"}
	t0 = time.perf_counter()
	res = await client.post(f"{base}{path}", json=payload, headers=headers)
	dt = time.perf_counter() - t0
	try:
		body: dict | str = res.json()
	except ValueError:
		body = redact(res.text[:300])
	return res.status_code, body, dt, res.headers.get("x-request-id")


async def main(rounds: int) -> dict:
	if not muse_spark.configured():
		return {"ok": False, "skipped": "META_API_KEY or META_API_BASE not set"}
	summary: dict = {"model": muse_spark.MODEL, "base_host": httpx.URL(os.environ["META_API_BASE"]).host}
	async with httpx.AsyncClient(timeout=60) as client:
		status, body, dt, rid = await raw_post(
			client,
			"/responses",
			{
				"model": muse_spark.MODEL,
				"input": "Reply with the single word: ready",
				"max_output_tokens": 16,
				"store": False,
			},
		)
		summary["plain"] = {
			"status": status,
			"latency_s": round(dt, 2),
			"request_id": rid,
			**(shape(body) if isinstance(body, dict) else {"error": body}),
		}
		if status == 200 and isinstance(body, dict):
			summary["plain"]["text"] = muse_spark._parse(body, rid).text[:40]
		if status in (404, 405):
			cstatus, cbody, cdt, _ = await raw_post(
				client,
				"/chat/completions",
				{
					"model": muse_spark.MODEL,
					"messages": [{"role": "user", "content": "Reply with: ready"}],
					"max_tokens": 8,
				},
			)
			summary["chat_completions"] = {
				"status": cstatus,
				"latency_s": round(cdt, 2),
				**({"top_keys": sorted(cbody.keys())} if isinstance(cbody, dict) else {"error": cbody}),
			}
			return summary

		rec = json.loads(SAMPLE.read_text())
		view = RecordView.from_record(rec)
		tools = build_tools(view, set(), move_pending=False)
		items = [
			{
				"role": "user",
				"content": f"[event] The person opened the cockpit on {view.title}; their aim is min_cost. Start the tour.",
			},
			{
				"role": "user",
				"content": "[event] Current setting: "
				+ json.dumps({"knobs": {k: v.current for k, v in view.knobs.items()}}),
			},
		]
		latencies, usages, called = [], [], []
		errors: list[str] = []
		for _ in range(rounds):
			t0 = time.perf_counter()
			try:
				turn = await muse_spark.create_turn(
					instructions=INSTRUCTIONS + language_line(None),
					input_items=items,
					tools=tools,
					max_output_tokens=900,
					client=client,
				)
			except muse_spark.GuideModelError as e:
				errors.append(redact(str(e)))
				continue
			latencies.append(time.perf_counter() - t0)
			usages.append(
				{
					"in": turn.input_tokens,
					"out": turn.output_tokens,
					"cached": turn.cached_tokens,
					"usd": round(turn.cost_usd, 6),
				}
			)
			called.append([c.name for c in turn.calls])
		summary["guide_round"] = {
			"rounds": rounds,
			"ok": len(latencies),
			"errors": errors[:3],
			"latency_s": {
				"p50": round(statistics.median(latencies), 2) if latencies else None,
				"max": round(max(latencies), 2) if latencies else None,
			},
			"usage": usages,
			"tools_called": called,
			"tool_count": len(tools),
		}
	summary["ok"] = bool(
		summary.get("plain", {}).get("status") == 200 and summary.get("guide_round", {}).get("ok")
	)
	return summary


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--rounds", type=int, default=5)
	result = asyncio.run(main(ap.parse_args().rounds))
	text = redact(json.dumps(result))
	print(text)
	if os.environ.get("GITHUB_ACTIONS"):
		# Annotations are readable through the API without log access; keep each under 4 KB.
		for i in range(0, len(text), 3500):
			print(f"::notice title=muse-spark-smoke part {i // 3500 + 1}::{text[i : i + 3500]}")
	sys.exit(0)
