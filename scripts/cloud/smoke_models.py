"""Model API smoke test and latency measurement for Baseten (Muse Spark Open) and the Meta Model API.

Run through GitHub Actions (see docs/setup/cloud.md):
	gh workflow run remote-run.yml -f name=models -f cmd="python scripts/cloud/smoke_models.py --n 20"

For each provider it lists the available model ids, then sends --n tiny chat requests and records
latency (p50, p95, max), token usage and HTTP status. Writes out/smoke_models.json. The latencies
ground the guide's timeout (about 2x measured p95) instead of a guessed 30 s. Never prints a key.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx


def _pct(xs: list[float], q: float) -> float | None:
	if not xs:
		return None
	xs = sorted(xs)
	return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 3)


def _body(api: str, model: str, extra: dict) -> dict:
	prompt = "Reply with the single word OK."
	if api == "responses":  # Meta Model API format, as in its dashboard's cURL example
		return {
			"model": model,
			"input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
			"stream": False,
			**extra,
		}
	return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, **extra}


def probe(name: str, base: str, key: str, model: str | None, n: int, extra: dict, api: str = "chat") -> dict:
	out: dict = {"provider": name, "api": api, "base": base, "configured": bool(base and key)}
	if not out["configured"]:
		out["note"] = "missing key or base URL secret"
		return out
	h = {"Authorization": f"Bearer {key}"}
	with httpx.Client(timeout=120) as c:
		try:
			r = c.get(f"{base.rstrip('/')}/models", headers=h)
			out["models_status"] = r.status_code
			if r.is_success:
				out["model_ids"] = sorted(m.get("id", "?") for m in r.json().get("data", []))
		except Exception as e:  # report, don't fail: this is a smoke test
			out["models_error"] = type(e).__name__
		model = model or (out.get("model_ids") or [None])[0]
		out["model"] = model
		lat, statuses, usage = [], [], []
		for _ in range(n if model else 0):
			body = _body(api, model, extra)
			path = "/responses" if api == "responses" else "/chat/completions"
			t0 = time.monotonic()
			try:
				r = c.post(f"{base.rstrip('/')}{path}", headers=h, json=body)
				statuses.append(r.status_code)
				if r.is_success:
					lat.append(time.monotonic() - t0)
					usage.append(r.json().get("usage", {}))
				else:
					out.setdefault("errors", []).append(r.text[:300])
			except Exception as e:
				statuses.append(type(e).__name__)
	out["statuses"] = statuses
	out["latency_s"] = {
		"n": len(lat),
		"p50": _pct(lat, 0.5),
		"p95": _pct(lat, 0.95),
		"max": round(max(lat), 3) if lat else None,
		"mean": round(statistics.fmean(lat), 3) if lat else None,
	}
	out["usage_samples"] = usage[:3]
	return out


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--n", type=int, default=5, help="requests per provider")
	a = ap.parse_args()
	res = {
		"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"baseten": probe(
			"baseten",
			os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
			os.environ.get("BASETEN_API_KEY", ""),
			os.environ.get("BASETEN_MODEL", "meta-models/muse-spark-open-1026"),
			a.n,
			{"reasoning_effort": "low"},
		),
		# Meta: which endpoint works for Muse Spark? A's harness uses /chat/completions, C's guide /responses.
		"meta_responses": probe(
			"meta",
			os.environ.get("META_API_BASE", "https://api.meta.ai/v1"),
			os.environ.get("META_API_KEY", ""),
			os.environ.get("META_MODEL", "muse-spark-1.3"),
			a.n,
			{},
			api="responses",
		),
		"meta_chat": probe(
			"meta",
			os.environ.get("META_API_BASE", "https://api.meta.ai/v1"),
			os.environ.get("META_API_KEY", ""),
			os.environ.get("META_MODEL", "muse-spark-1.3"),
			min(a.n, 3),
			{},
			api="chat",
		),
	}
	Path("out").mkdir(exist_ok=True)
	Path("out/smoke_models.json").write_text(json.dumps(res, indent="\t") + "\n")
	print(json.dumps(res, indent="\t"))


if __name__ == "__main__":
	main()
