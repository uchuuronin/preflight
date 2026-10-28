"""The visitor's side of a live-guide tour, for api/scripts/guide_tour.py: a Python stand-in for
the browser's guide tools (web/src/guide/liveGuide.ts) that reads a record's grids exactly at
grid points, and a driver that flies whole tours through the real POST /v1/guide/turns route.
Pure Python so it runs locally against a stand-in model as well as on Modal:

	cd api && META_API_KEY=local META_API_BASE=http://127.0.0.1:8099/v1 \
		uv run python scripts/tour_play.py --tours 1
"""

from __future__ import annotations

import json
import time
from pathlib import Path

FITS_BELOW = 0.01  # B's cut-offs, web/src/workspace/record/terms.ts
FAILS_AT = 0.8
HEADLINE = ("memory", "time", "cost", "risk")
MAX_REQUESTS_PER_TOUR = 40  # a stuck tour stops here
MAX_DECISIONS_PER_TOUR = 4  # four applied moves is a long tour for a visitor


# ---------------------------------------------------------------- the browser's part (on-grid)


class Grids:
	def __init__(self, rec: dict) -> None:
		import base64

		import numpy as np

		self.rec = rec
		self.knobs = {k["id"]: k for k in rec["knobs"]}
		self.current = {k["id"]: k["current"] for k in rec["knobs"]}
		self.targets = {t["id"]: t for t in rec["predictions"]["targets"]}
		self.estimates = rec["predictions"].get("estimates") or {}
		self.grids = []
		for g in rec["predictions"].get("grids") or []:
			shape = tuple(len(a["values"]) for a in g["axes"])
			n = int(np.prod(shape)) if shape else 1
			sc = g.get("scale") or 1

			def f16(data: str | None, n: int = n, sc: float = sc) -> np.ndarray:
				return np.frombuffer(base64.b64decode(data or ""), dtype="<f2").astype(float) * sc

			mask = np.zeros(n, dtype=bool)
			if g.get("abstain"):
				bits = np.unpackbits(
					np.frombuffer(base64.b64decode(g["abstain"]), dtype=np.uint8), bitorder="little"
				)
				mask = bits[:n].astype(bool)
			self.grids.append((g, shape, f16(g.get("point")), f16(g.get("lo")), f16(g.get("hi")), mask))

	def choices(self, kid: str) -> list:
		k = self.knobs[kid]
		if k.get("kind") == "bool":
			return [False, True]
		vals = list((k.get("domain") or {}).get("values") or [])
		for g, *_ in self.grids:
			for a in g["axes"]:
				if a["knob_id"] == kid:
					vals += [v for v in a["values"] if v not in vals]
		return vals

	def predict(self, knobs: dict) -> dict:
		full = {**self.current, **knobs}
		here = full == self.current
		out = {}
		for tid, t in self.targets.items():
			if here and tid in self.estimates:
				out[tid] = self.estimates[tid]
				continue
			out[tid] = {
				"point": None,
				"lo": None,
				"hi": None,
				"unit": t.get("unit", ""),
				"status": "abstain",
				"reason": "off_grid",
			}
			for g, shape, p, lo, hi, mask in self.grids:
				if g["target_id"] != tid:
					continue
				on_axis = {a["knob_id"] for a in g["axes"]}
				fixed = g.get("fixed") or {}
				if any(
					full.get(k) != (fixed[k] if k in fixed else self.current[k])
					for k in self.current
					if k not in on_axis
				):
					continue
				idx = 0
				for a, size in zip(g["axes"], shape, strict=True):
					v = full.get(a["knob_id"])
					if v not in a["values"]:
						idx = None
						break
					idx = idx * size + a["values"].index(v)
				if idx is None:
					out[tid]["reason"] = "outside_grid"
					break
				if mask[idx]:
					out[tid]["reason"] = "outside_trusted_range"
					break
				out[tid] = {
					"point": float(p[idx]),
					"lo": float(lo[idx]),
					"hi": float(hi[idx]),
					"unit": g.get("unit") or t.get("unit", ""),
					"status": "predicted",
				}
				break
		return out

	def readouts(self, pred: dict) -> dict:
		ids = [tid for tid, t in self.targets.items() if t.get("readout") in HEADLINE]
		ids += [x for x in ("throughput.tokens_per_s", "tokens.total") if x in self.targets]
		res = {}
		for tid in ids:
			e = pred.get(tid)
			if not e:
				continue
			ab = e.get("status") == "abstain" or e.get("point") is None
			r = {
				"point": None if ab else e["point"],
				"lo": None if ab else e.get("lo"),
				"hi": None if ab else e.get("hi"),
				"unit": e.get("unit") or "",
				"status": "abstain" if ab else e["status"],
			}
			res[tid] = r
		return res

	def risk_target(self) -> str | None:
		return next((t for t in ("risk.oom",) if t in self.targets), None)

	def fit(self, pred: dict) -> str:
		rt = self.risk_target()
		r = pred.get(rt) if rt else None
		if not r or r.get("point") is None:
			return "unknown"
		p = r["point"]
		return "fits" if p < FITS_BELOW else "tight" if p < FAILS_AT else "fails"

	def aim(self, kind: str) -> str | None:
		want = {"min_cost": "cost", "min_time": "time"}.get(kind, "cost")
		return next((tid for tid, t in self.targets.items() if t.get("readout") == want), None)

	def sensitivity(self, knobs: dict, pinned: set, kind: str) -> list[dict]:
		goal = self.aim(kind)
		base = self.predict(knobs)
		rt = self.risk_target()
		base_risk = (base.get(rt) or {}).get("point") or 0 if rt else 0
		failing = base_risk >= FITS_BELOW
		base_goal = (base.get(goal) or {}).get("point") if goal else None
		moves = []
		for kid in self.knobs:
			if kid in pinned:
				continue
			best = None
			for v in self.choices(kid):
				if v == knobs.get(kid):
					continue
				p = self.predict({**knobs, kid: v})
				f = self.fit(p)
				if f == "fails" or (f == "unknown" and rt):
					continue
				if failing:
					risk = (p.get(rt) or {}).get("point")
					if risk is None or risk >= base_risk:
						continue
					rel = (risk - base_risk) / max(base_risk, 1e-9)
					delta = risk - base_risk
				else:
					g = (p.get(goal) or {}).get("point") if goal else None
					if g is None or base_goal in (None, 0) or g >= base_goal:
						continue
					rel = (g - base_goal) / base_goal
					delta = g - base_goal
				if best is None or rel < best[0]:
					best = (rel, v, delta, "tight" if f == "tight" else "fits")
			if best:
				moves.append((kid, *best))
		moves.sort(key=lambda m: m[1])
		moves = moves[:8]
		top = max((abs(m[1]) for m in moves), default=1e-12) or 1e-12
		out = []
		for kid, rel, v, delta, fit in moves:
			cur = knobs.get(kid)
			num = (
				isinstance(v, int | float)
				and not isinstance(v, bool)
				and isinstance(cur, int | float)
				and not isinstance(cur, bool)
			)
			direction = ("increase" if v > cur else "decrease") if num else "change"
			out.append(
				{
					"knob_id": kid,
					"impact": min(1.0, abs(rel) / top),
					"direction": direction,
					"best_value": v,
					"delta": delta,
					"fit": fit,
				}
			)
		return out

	def guard(self, knobs: dict, pinned: set, kid: str, value) -> str | None:
		if kid not in self.knobs or value not in self.choices(kid):
			return "not_in_domain"
		if kid in pinned:
			return "pinned"
		p = self.predict({**knobs, kid: value})
		mem = next((tid for tid, t in self.targets.items() if t.get("readout") == "memory"), None)
		if mem and (p.get(mem) or {}).get("point") is None:
			return "not_trusted"
		f = self.fit(p)
		return "would_fail" if f == "fails" else "would_be_tight" if f == "tight" else None


def browser_results(
	g: Grids, calls: list[dict], knobs: dict, pinned: set
) -> tuple[list[dict], dict, dict | None]:
	"""Typed results for the browser calls, the knobs after them, and the move made (if any)."""
	results, move = [], None
	for c in calls:
		if c["runs_in"] != "browser":
			continue
		a, cid, name = c["arguments"], c["call_id"], c["name"]
		if name == "predict":
			ks = {k: v for k, v in (a.get("knobs") or {}).items() if k in g.knobs}
			results.append(
				{"tool": "predict", "call_id": cid, "readouts": g.readouts(g.predict({**knobs, **ks}))}
			)
		elif name == "sensitivity":
			results.append(
				{
					"tool": "sensitivity",
					"call_id": cid,
					"ranked": g.sensitivity(knobs, pinned, a.get("objective") or "min_cost"),
				}
			)
		elif name == "set_knob":
			kid, v = str(a.get("knob_id")), a.get("value")
			refused = "one_per_round" if move else g.guard(knobs, pinned, kid, v)
			if refused:
				results.append(
					{
						"tool": "set_knob",
						"call_id": cid,
						"moved": False,
						"knob_id": kid if kid in g.knobs else next(iter(g.knobs)),
						"from_value": knobs.get(kid),
						"to_value": None,
						"readouts": {},
						"refused": refused,
					}
				)
			else:
				move = {"knob_id": kid, "from": knobs[kid], "to": v}
				knobs = {**knobs, kid: v}
				results.append(
					{
						"tool": "set_knob",
						"call_id": cid,
						"moved": True,
						"knob_id": kid,
						"from_value": move["from"],
						"to_value": v,
						"readouts": g.readouts(g.predict(knobs)),
						"refused": None,
					}
				)
		elif name == "highlight":
			nodes = {n["id"] for n in (g.rec.get("graph") or {}).get("nodes") or []}
			results.append(
				{
					"tool": "highlight",
					"call_id": cid,
					"nodes": [n for n in (a.get("nodes") or []) if n in nodes][:24],
				}
			)
		elif name == "pin":
			kid = str(a.get("knob_id"))
			if kid in g.knobs:
				if a.get("pinned"):
					pinned.add(kid)
				else:
					pinned.discard(kid)
				results.append(
					{"tool": "pin", "call_id": cid, "knob_id": kid, "pinned": bool(a.get("pinned"))}
				)
	return results, knobs, move


# ---------------------------------------------------------------- one tour through the real route


async def one_tour(api, g: Grids, sample: str, aim: str, timing: list) -> dict:
	knobs, pinned = dict(g.current), set()
	state, rounds, decisions, approved = None, [], 0, False
	body: dict = {"start": True}
	tour = {"sample": sample, "aim": aim, "rounds": rounds, "end": None}
	for _ in range(MAX_REQUESTS_PER_TOUR):
		payload = {
			"record": sample,
			"setting": {"knobs": knobs, "pinned": sorted(pinned), "objective": {"kind": aim}},
			"state": state,
			**body,
		}
		timing.clear()
		res = await api.post("/v1/guide/turns", json=payload)
		data = res.json()
		if res.status_code != 200:
			err = data.get("error") or {}
			tour["end"] = f"http_{res.status_code}:{err.get('reason') or err.get('code')}"
			tour["error"] = (err.get("message") or "")[:200]
			break
		prov = data["provenance"]
		rounds.append(
			{
				"latency_s": round(timing[-1], 3) if timing else None,
				"in": prov["input_tokens"],
				"out": prov["output_tokens"],
				"cached": prov["cached_tokens"],
				"usd": prov["cost_usd"],
				"calls": [c["name"] for c in data["calls"]],
				"awaiting": data["awaiting"],
				"withheld": data["withheld"],
				"text": data["text"][:240],
			}
		)
		state = data["state"]
		aw = data["awaiting"]
		if aw == "results":
			results, knobs, move = browser_results(g, data["calls"], knobs, pinned)
			body = {"results": results}
			for res_ in results:
				if res_["tool"] == "sensitivity" and any(x["fit"] == "fits" for x in res_["ranked"]):
					tour["offered_fitting_move"] = True
			if move:
				tour.setdefault("moves", []).append(move)
		elif aw == "decision":
			if decisions >= MAX_DECISIONS_PER_TOUR:
				tour["end"] = "decision_cap"
				break
			mv = (tour.get("moves") or [None])[-1]
			if not mv:
				tour["end"] = "decision_without_move"
				break
			decisions += 1
			body = {"decision": {"knob_id": mv["knob_id"], "action": "applied", "value": mv["to"]}}
		elif aw == "approval":
			pend = [c for c in data["calls"] if c["runs_in"] == "approval"]
			if not pend or approved:
				tour["end"] = "approval_left"
				break
			approved = True
			body = {"approvals": [{"call_id": pend[0]["call_id"], "approved": True}]}
		else:
			tour["end"] = "done"
			break
	else:
		tour["end"] = "request_cap"
	tour["decisions"] = decisions
	tour["usd"] = round(sum(r["usd"] for r in rounds), 6)
	tour["tokens_in"] = sum(r["in"] for r in rounds)
	tour["tokens_out"] = sum(r["out"] for r in rounds)
	return tour


#: Lines of the guide's instructions that the prompt variants add or drop (prompt.py).
_SPEAK_ONCE = (
	"- While you are still calling tools, write nothing; speak once, when you stop for the person.\n"
)
_ACT_NOW = (
	"- When sensitivity returns a move that fits, call set_knob in that same reply. Never end a\n"
	"  reply with a plan you haven't carried out.\n"
)


def instructions_for(variant: str, current: str) -> str:
	"""Prompt variants (docs/decisions/guide-limits.md, run 3 compared four). "current" is prompt.py
	as shipped; "speak_once" is run 2's instructions, for a before and after."""
	if variant == "current":
		return current
	if variant == "speak_once":
		return current.replace(_ACT_NOW, _SPEAK_ONCE)
	raise ValueError(variant)


async def run_tours(
	root: Path, samples: list[str], tours: int, aim: str, variants: tuple[str, ...] = ("current",)
) -> dict:
	"""Fly `tours` tours on each sample (per prompt variant) through the real route."""
	import os
	import secrets

	import httpx

	from preflight_api.app import create_app
	from preflight_api.guide import muse_spark
	from preflight_api.ledger import LedgerCore, Limits, MemoryLedger
	from preflight_api.stores import SAMPLES

	os.environ.setdefault("PREFLIGHT_SESSION_KEYS", secrets.token_urlsafe(48))  # throwaway
	timing: list[float] = []

	async def on_request(request: httpx.Request) -> None:
		request.extensions["t0"] = time.perf_counter()

	async def on_response(response: httpx.Response) -> None:
		if response.request.url.path.endswith("/responses"):
			timing.append(time.perf_counter() - response.request.extensions["t0"])

	# A long client timeout so the measurement sees the real tail, not the guide's cut-off.
	model_http = httpx.AsyncClient(
		timeout=180, event_hooks={"request": [on_request], "response": [on_response]}
	)
	# The caps under test are measured here, so they must not stop the measurement.
	limits = Limits(
		usd_per_run=50, usd_per_day=50, turns_per_visitor_day=10_000, turns_per_ip_day=10_000, ip_burst=10_000
	)
	api_app = create_app(ledger=MemoryLedger(LedgerCore(limits)), http=model_http)
	out: dict = {"model": muse_spark.MODEL, "aim": aim, "tours": []}
	from preflight_api.routes import guide as route

	base_prompt = route.INSTRUCTIONS
	try:
		for variant in variants:
			route.INSTRUCTIONS = instructions_for(variant, base_prompt)
			for sample in samples:
				g = Grids(json.loads((root / SAMPLES[sample]).read_text()))
				for _ in range(tours):
					transport = httpx.ASGITransport(app=api_app)
					async with httpx.AsyncClient(transport=transport, base_url="https://tour.local") as api:
						t = await one_tour(api, g, sample, aim, timing)
						t["variant"] = variant
						out["tours"].append(t)
	finally:
		route.INSTRUCTIONS = base_prompt
		await model_http.aclose()
	out["summary"] = summarise(out)
	if len(variants) > 1:
		out["by_variant"] = {
			v: compare({"tours": [t for t in out["tours"] if t["variant"] == v]}) for v in variants
		}
	return out


def compare(res: dict) -> dict:
	"""Per prompt variant: how tours ended, and how often the guide stalled with a move on offer."""
	tours = res["tours"]
	rounds = [r for t in tours for r in t["rounds"]]
	offered = [t for t in tours if t.get("offered_fitting_move")]
	return {
		"tours": len(tours),
		"moved_when_offered": sum(1 for t in offered if t["decisions"] > 0),
		"offered": len(offered),
		"stalled": sum(1 for t in offered if t["decisions"] == 0 and t["end"] == "done"),
		"loops": sum(1 for t in tours if str(t["end"]).endswith("too_many_rounds")),
		"text_while_calling": sum(1 for r in rounds if r["text"] and r["awaiting"] == "results"),
		"rounds": len(rounds),
		"usd": round(sum(t["usd"] for t in tours), 4),
		"latency_p50": sorted(r["latency_s"] for r in rounds)[len(rounds) // 2] if rounds else None,
	}


def summarise(res: dict) -> dict:
	import statistics

	rounds = [r for t in res["tours"] for r in t["rounds"]]
	lat = sorted(r["latency_s"] for r in rounds if r["latency_s"] is not None)

	def pct(xs: list[float], q: float) -> float | None:
		return round(xs[min(len(xs) - 1, round(q * (len(xs) - 1)))], 3) if xs else None

	per_tour_usd = [t["usd"] for t in res["tours"]]
	per_tour_rounds = [len(t["rounds"]) for t in res["tours"]]
	return {
		"model": res["model"],
		"tours": len(res["tours"]),
		"rounds": len(rounds),
		"latency_s": {
			"p50": pct(lat, 0.5),
			"p90": pct(lat, 0.9),
			"p95": pct(lat, 0.95),
			"max": lat[-1] if lat else None,
		},
		"output_tokens": {
			"p50": pct(sorted(r["out"] for r in rounds), 0.5),
			"p95": pct(sorted(r["out"] for r in rounds), 0.95),
			"max": max((r["out"] for r in rounds), default=None),
		},
		"input_tokens": {
			"p50": pct(sorted(r["in"] for r in rounds), 0.5),
			"max": max((r["in"] for r in rounds), default=None),
		},
		"cached_share": round(sum(r["cached"] for r in rounds) / max(1, sum(r["in"] for r in rounds)), 3),
		"rounds_per_tour": {
			"mean": round(statistics.fmean(per_tour_rounds), 2) if per_tour_rounds else None,
			"max": max(per_tour_rounds, default=None),
		},
		"usd_per_tour": {
			"mean": round(statistics.fmean(per_tour_usd), 4) if per_tour_usd else None,
			"max": max(per_tour_usd, default=None),
		},
		"withheld_rounds": sum(1 for r in rounds if r["withheld"]),
		"ends": {e: sum(1 for t in res["tours"] if t["end"] == e) for e in {t["end"] for t in res["tours"]}},
	}


def report(res: dict) -> str:
	lines = [json.dumps(res["summary"], indent="\t")]
	if "by_variant" in res:
		lines.append(json.dumps(res["by_variant"], indent="\t"))
	for t in res["tours"]:
		lines.append(
			f"{t['sample']:<10} rounds={len(t['rounds']):<3} decisions={t['decisions']} usd={t['usd']:.4f} end={t['end']} {t.get('error', '')}"
		)
		for r in t["rounds"]:
			w = f"WITHHELD {r['withheld']} " if r["withheld"] else ""
			lines.append(
				f"    {r['latency_s']}s in={r['in']} out={r['out']} calls={r['calls']} -> {r['awaiting']} {w}{r['text'][:120]!r}"
			)
	return "\n".join(lines)


if __name__ == "__main__":
	import argparse
	import asyncio
	import sys

	here = Path(__file__).resolve()
	sys.path[:0] = [str(here.parents[1]), str(here.parents[2] / "python")]
	ap = argparse.ArgumentParser()
	ap.add_argument("--tours", type=int, default=1)
	ap.add_argument("--samples", default="lora-7b,lora-qwen,pretrain,agent")
	ap.add_argument("--aim", default="min_cost")
	ap.add_argument("--variants", default="current")
	a = ap.parse_args()
	res = asyncio.run(
		run_tours(here.parents[2], a.samples.split(","), a.tours, a.aim, tuple(a.variants.split(",")))
	)
	print(report(res))
