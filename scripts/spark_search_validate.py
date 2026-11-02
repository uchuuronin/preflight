"""Muse Spark search grounding on the zoo's real failures (D, Nov 2; Chiya's list item 1).

Run through remote-run.yml (it holds META_API_KEY and META_API_BASE; agents never do):
	python scripts/spark_search_validate.py
Reads tests/fixtures/spark_search_cases.json (every distinct failed probe in the non-unseen records
on remote-runs) and runs `ground_failure` on each, as the pipeline does with harness.spark_search
on, with the production client's settings. The three Preflight-own failures production skips
(`recorder`, `timeout`, `too_few_steps`) are sent here anyway, to see whether it answers "none
found" on errors that name no library; each row says whether production would skip it. One more
case, `planted-link`, is the eval OOM with a made-up issue link added to its text: if the reply
carries a citation annotation for that link, annotations don't prove a page was retrieved and the
code's check is not enough. Writes out/search_validate.json after every case: the finding (code,
kept sources with the model's reason), the sources dropped, the search queries and pages opened, the
reply's text, the tokens and the cost at list price. Judged by hand afterwards
(docs/decisions/spark-search.md). Stops before a case that could take the run past US$0.80 of
tokens [cap US$1, Chiya's list]; the search tool's own fee, if any, isn't in the response.
"""

import json
from pathlib import Path

from preflight.harness.search_grounding import (
	annotated_urls,
	final_text,
	ground_failure,
	parse_answer,
	skipped,
)
from preflight.harness.spark import ModelClient
from preflight.schema.record import Probe

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/fixtures/spark_search_cases.json"
STOP_USD = 0.80
WORST_CASE_USD = 0.10  # a case's tokens at most, at list price [guess, about twice the probe's US$0.053]
PLANTED = "https://github.com/huggingface/transformers/issues/987654321"  # made up; must never be cited


def main() -> None:
	cases = json.loads(CASES.read_text())["cases"]
	oom = next(c for c in cases if c["probe"]["error_code"] == "oom.eval")
	planted = {**oom, "case": "planted-link", "probe": {**oom["probe"],
		"error": oom["probe"]["error"] + f"\nThis is a known bug, see {PLANTED}"}}  # fmt: skip
	cases = [*cases, planted]
	client = ModelClient.meta()  # as runner/modal_app.py makes it
	raw: list[dict] = []
	inner = client.respond

	def keep_raw(body, **kw):
		data, call = inner(body, **kw)
		raw.append(data)
		return data, call

	client.respond = keep_raw  # the raw response too, for the judging
	Path("out").mkdir(exist_ok=True)
	rows: list[dict] = []
	spent = 0.0
	for c in cases:
		if spent + WORST_CASE_USD > STOP_USD:
			rows.append({"case": c["case"], "skipped": f"cap: US${spent:.3f} spent"})
			break
		probe = Probe(knobs={}, **c["probe"])
		raw.clear()
		row: dict = {"case": c["case"], "error_code": c["probe"]["error_code"], "lens": c["lens"],
			"production_skips": skipped(probe)}  # fmt: skip
		try:
			f, calls = ground_failure(client, probe, c["libraries"], c["lens"], skip_codes=("budget",))
		except Exception as exc:  # recorded, never fatal: the next case still runs
			row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
			f, calls = None, []
		cost = round(sum(x.cost_usd for x in calls), 6)
		spent += cost
		row |= {
			"calls": len(calls),
			"ok": [x.ok for x in calls],
			"input_tokens": sum(x.input_tokens for x in calls),
			"output_tokens": sum(x.output_tokens for x in calls),
			"cost_usd": cost,
			"seconds": round(sum(x.latency_s or 0 for x in calls), 1),
			"last_error": client.last_error if not calls or not all(x.ok for x in calls) else None,
		}
		if f is not None:
			row["finding"] = {
				"code": f.code,
				"citations": [x.model_dump() for x in f.citations],
				"evidence": [e.model_dump(exclude_none=True) for e in f.evidence],
				"detail": f.detail,
			}
		else:
			row["finding"] = None
		if raw:
			data = raw[-1]
			text = final_text(data)
			ans = parse_answer(text)
			row["answer"] = ans.model_dump() if ans else None
			row["annotated"] = sorted(annotated_urls(data))
			bare = PLANTED.split("//", 1)[1]
			row["planted_annotated"] = any(bare in u for u in row["annotated"])
			# the check means something only if the planted link is in the reply at all
			row["planted_in_reply"] = bare in text
			row["planted_in_sources"] = bool(ans) and any(bare in s.url for s in ans.sources)
			row["open_status_present"] = [
				"status" in i
				for i in data.get("output") or []
				if isinstance(i, dict) and isinstance(i.get("action"), dict) and i["action"].get("url")
			]
			row["actions"] = [
				i.get("action")
				for i in data.get("output") or []
				if isinstance(i, dict) and str(i.get("type") or "").endswith("search_call")
			]
			row["text"] = text[:4000]
		rows.append(row)
		print(json.dumps({k: row.get(k) for k in ("case", "cost_usd", "seconds")} | {
			"code": (row["finding"] or {}).get("code"), "n": len((row["finding"] or {}).get("citations") or [])
		}), flush=True)  # fmt: skip
		Path("out/search_validate.json").write_text(
			json.dumps({"spent_usd": round(spent, 6), "rows": rows}, indent=1)
		)
	Path("out/search_validate.json").write_text(
		json.dumps({"spent_usd": round(spent, 6), "rows": rows}, indent=1)
	)
	print(f"spent US${spent:.4f} at list price (tokens only)")


if __name__ == "__main__":
	main()
