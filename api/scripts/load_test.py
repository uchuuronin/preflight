"""Twenty visitors at once against the deployed app: what each would load in its first minute.

Each visitor gets its own session (cookie jar) and walks: the web app's page and its scripts,
the API's status, then one record by share link (when one is given), a few times over. Nothing
here calls a model or starts a job, so it spends no tokens and no GPU; the only cost is the
API containers' CPU while it runs (one to two minutes). Run it only with Chiya's yes, like any
call against the deployed app:

	uv run python api/scripts/load_test.py --web https://<pages host> --api https://<api host>
		[--record-token $TOKEN] [--visitors 20] [--rounds 5]

A share token is a capability: pass it from the environment, never write it into the repo.
Exits 1 if any request failed (5xx, a timeout, or a refusal other than 429) or the p95 is over
the bar.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict

import httpx

# Source: guess, the product bar for "the cockpit opens without waiting": a page or record that
# takes longer than this at the 95th percentile, with 20 visitors, is a problem to look at.
P95_BAR_S = 1.5
# Source: guess, a stalled request; well past the bar so a slow one is counted, not dropped.
TIMEOUT_S = 15.0


async def visitor(n: int, web: str, api: str, token: str | None, rounds: int, out: dict) -> None:
	async with httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True) as c:

		async def hit(name: str, url: str) -> httpx.Response | None:
			t0 = time.perf_counter()
			try:
				r = await c.get(url)
			except httpx.HTTPError as e:
				out[name].append((time.perf_counter() - t0, f"error {type(e).__name__}"))
				return None
			out[name].append((time.perf_counter() - t0, r.status_code))
			return r

		page = await hit("web page", web + "/")
		scripts = re.findall(r'src="(/assets/[^"]+\.js)"', page.text) if page is not None else []
		for s in scripts[:3]:
			await hit("web script", web + s)
		for _ in range(rounds):
			await hit("api status", api + "/v1/status")
			if token:
				await hit("api record", f"{api}/v1/records/{token}")
			await asyncio.sleep(0.5 + (n % 5) * 0.1)  # a person reading, not a tight loop


def summary(out: dict) -> tuple[dict, bool]:
	ok, report = True, {}
	for name, rows in out.items():
		secs = sorted(s for s, _ in rows)
		bad = [code for _, code in rows if not (isinstance(code, int) and (code < 400 or code == 429))]
		p95 = secs[min(len(secs) - 1, int(0.95 * len(secs)))] if secs else None
		report[name] = {
			"requests": len(rows),
			"p50_s": round(statistics.median(secs), 3) if secs else None,
			"p95_s": round(p95, 3) if p95 is not None else None,
			"max_s": round(secs[-1], 3) if secs else None,
			"failed": len(bad),
			"refused_429": sum(1 for _, c in rows if c == 429),
		}
		ok = ok and not bad and (p95 is None or p95 <= P95_BAR_S)
	return report, ok


async def main(argv: list[str]) -> int:
	ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	ap.add_argument("--web", required=True)
	ap.add_argument("--api", required=True)
	ap.add_argument("--record-token", default=os.environ.get("PREFLIGHT_LOAD_TOKEN"))
	ap.add_argument("--visitors", type=int, default=20)
	ap.add_argument("--rounds", type=int, default=5)
	a = ap.parse_args(argv)
	out: dict = defaultdict(list)
	t0 = time.perf_counter()
	await asyncio.gather(
		*(
			visitor(i, a.web.rstrip("/"), a.api.rstrip("/"), a.record_token, a.rounds, out)
			for i in range(a.visitors)
		)
	)
	report, ok = summary(out)
	print(
		json.dumps(
			{"visitors": a.visitors, "seconds": round(time.perf_counter() - t0, 1), **report}, indent=1
		)
	)
	print("load test:", "passed" if ok else "FAILED", f"(p95 bar {P95_BAR_S} s, no failures)")
	return 0 if ok else 1


if __name__ == "__main__":
	sys.exit(asyncio.run(main(sys.argv[1:])))
