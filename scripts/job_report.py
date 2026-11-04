"""What happened to one production job, read over plain HTTP through the public site (PM, Nov 4: a check-in
ended "Nothing was measured" with the recorder error, and the job's own error text says why). Prints the job's
status, every event's kind and message, and each probe's status, error code and error from its record. The
share token is masked: it never reaches the log. Reads only; spends nothing.

	python scripts/job_report.py <job id> [https://preflight-cqn.pages.dev]
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

UA = {"User-Agent": "preflight-job-report"}
TOKEN = re.compile(r"[A-Za-z0-9_-]{20,}")


def get(url: str) -> tuple[int, str]:
	req = urllib.request.Request(url, headers=UA)
	try:
		with urllib.request.urlopen(req, timeout=60) as r:
			return r.status, r.read().decode("utf-8", "replace")
	except urllib.error.HTTPError as e:
		return e.code, e.read().decode("utf-8", "replace")
	except (urllib.error.URLError, TimeoutError, OSError) as e:
		return 0, type(e).__name__


def mask(text: str, token: str | None) -> str:
	return text.replace(token, "<share token>") if token else text


def main(job: str, base: str) -> int:
	code, body = get(f"{base}/v1/jobs/{job}")
	print(f"status HTTP {code}")
	if code != 200:
		print(body[:500])
		return 1
	status = json.loads(body)
	token = status.get("share_token")
	print(json.dumps({k: v for k, v in status.items() if k != "share_token"}, indent=1)[:2000])
	code, body = get(f"{base}/v1/jobs/{job}/events?after=-1")
	print(f"\nevents HTTP {code}")
	for line in body.splitlines():
		if line.startswith("data: "):
			ev = json.loads(line[6:])
			keep = {
				k: ev.get(k)
				for k in (
					"seq",
					"stage",
					"kind",
					"message",
					"detail",
					"error",
					"error_code",
					"probe_id",
					"gpu",
				)
				if ev.get(k)
			}
			print(mask(json.dumps(keep)[:600], token))
	if token:
		code, body = get(f"{base}/v1/records/{token}")
		print(f"\nrecord HTTP {code}")
		if code == 200:
			rec = json.loads(body)
			for p in rec.get("probes", []):
				print(
					mask(
						json.dumps(
							{k: p.get(k) for k in ("id", "status", "steps", "error_code", "error", "gpu")}
						)[:800],
						token,
					)
				)
			for f in rec.get("findings", [])[:12]:
				print(
					mask(
						json.dumps({k: f.get(k) for k in ("id", "severity", "title", "detail")})[:400], token
					)
				)
			print(
				"env:",
				json.dumps(
					{k: (rec.get("env") or {}).get(k) for k in ("torch", "cuda", "gpu", "python", "image")}
				),
			)
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "https://preflight-cqn.pages.dev"))
