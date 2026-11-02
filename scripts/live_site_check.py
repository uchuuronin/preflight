"""Check the live site after a deploy, over plain HTTP (C, Nov 2: the overnight deploys, when no
browser is at hand). Reads only public pages and the bundled samples; prints what it found and
exits 1 if a check fails. Nothing secret is read or printed.

	python scripts/live_site_check.py https://preflight-cqn.pages.dev
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

UA = {"User-Agent": "preflight-live-check"}


def get(url: str) -> tuple[int, str]:
	req = urllib.request.Request(url, headers=UA)
	try:
		with urllib.request.urlopen(req, timeout=30) as r:
			return r.status, r.read().decode("utf-8", "replace")
	except urllib.error.HTTPError as e:
		return e.code, e.read().decode("utf-8", "replace")


def main(base: str) -> int:
	base = base.rstrip("/")
	results: list[tuple[str, bool, str]] = []

	def check(name: str, ok: bool, detail: str = "") -> None:
		results.append((name, ok, detail))

	status, html = get(base + "/")
	check("index loads", status == 200, f"HTTP {status}")
	scripts = re.findall(r'src="(/assets/[^"]+\.js)"', html)
	check("index names its bundle", bool(scripts), ", ".join(scripts))
	bundle = "".join(get(base + s)[1] for s in scripts)
	# the bundle splits into chunks: read the ones the entry imports too
	for chunk in sorted(set(re.findall(r'"\./([\w.-]+\.js)"', bundle)))[:80]:
		bundle += get(f"{base}/assets/{chunk}")[1]
	check("abstain label is 'Not measured here'", '"Not measured here"' in bundle)
	check(
		"no 'Not trusted' readout label",
		'"readout.abstain_short":"Not trusted"' not in bundle and '"Not trusted"' not in bundle,
	)
	check("between-measured reason shipped", "Between measured points; no precomputed value here." in bundle)
	check("PF102 says measured on an L4", "Alpaca, measured on an L4 (1 Nov)" in bundle)
	check("private-triage copy shipped (flag off)", "Private triage" in bundle)
	status, health = get(base + "/v1/health")
	check("API health", status == 200, f"HTTP {status} {health[:120]}")
	status, body = get(base + "/v1/records/lora-qwen")
	rec = json.loads(body) if status == 200 else {}
	check(
		"API's PF102 sample is the Nov 2 live record",
		rec.get("record_id") == "01M6D5N3VSJZ4EFEHW7MDDK6Z3",
		f"HTTP {status} {rec.get('record_id')}",
	)
	grids = (rec.get("predictions") or {}).get("grids") or []
	check(
		"PF102 carries B's pair grids",
		sum(1 for g in grids if str(g.get("id", "")).endswith(".clears")) == 15,
	)
	for name, ok, detail in results:
		print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
	return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
	sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "https://preflight-cqn.pages.dev"))
