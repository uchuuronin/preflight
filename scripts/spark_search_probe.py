"""Does Muse Spark's Responses API search the web? One question, a few tool spellings (D, Nov 2).

Run through remote-run.yml (it holds META_API_KEY and META_API_BASE; agents never do):
    python scripts/spark_search_probe.py
Writes out/search_probe.json: for each tool spelling, the HTTP status, the output items' types,
any url_citation annotations, the answer text (trimmed) and the token usage. Prints no secrets.
At most four small calls at low effort.
"""

import json
import os
import time
from pathlib import Path

import httpx

QUESTION = (
	"A Hugging Face Trainer fine-tune runs out of GPU memory only during evaluation "
	"(torch.OutOfMemoryError in the eval loop), not during training. Search the web for known "
	"issues or documented causes and cite at most 3 sources."
)
TOOLS = [[{"type": "web_search"}], [{"type": "web_search_preview"}], []]


def _scrub(text: str, *secrets: str) -> str:
	"""An error body as saved: the key, the base URL and its host replaced, so nothing echoed leaks."""
	for s in secrets:
		if s:
			text = text.replace(s, "***")
	return text


def main() -> None:
	base = os.environ["META_API_BASE"].rstrip("/")
	key = os.environ["META_API_KEY"]
	host = httpx.URL(base).host
	out: list[dict] = []
	Path("out").mkdir(exist_ok=True)
	for tools in TOOLS:
		body = {
			"model": "muse-spark-1.3",
			"input": QUESTION,
			"max_output_tokens": 2048,
			"reasoning": {"effort": "low"},
			"store": False,
			**({"tools": tools} if tools else {}),
		}
		t0 = time.monotonic()
		try:
			r = httpx.post(
				f"{base}/responses", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90
			)
		except httpx.HTTPError as e:
			out.append({"tools": tools, "error": type(e).__name__})
			continue
		row: dict = {"tools": tools, "status": r.status_code, "seconds": round(time.monotonic() - t0, 1)}
		try:
			j = r.json()
		except ValueError:
			row["body"] = _scrub(r.text, key, base, host)[:500]
			out.append(row)
			continue
		if r.status_code >= 400:
			err = j.get("error") if isinstance(j, dict) else None
			err = err if isinstance(err, dict) else {}
			msg = _scrub(str(err.get("message", "")), key, base, host)
			row["error"] = {"type": err.get("type"), "code": err.get("code"), "message": msg[:300]}
			out.append(row)
			continue
		items = [i for i in (j.get("output") or []) if isinstance(i, dict)]
		row["item_types"] = [i.get("type") for i in items]
		texts, cites = [], []
		for i in items:
			for c in i.get("content") or []:
				if c.get("type") in ("output_text", "text"):
					texts.append(c.get("text", ""))
					for a in c.get("annotations") or []:
						cites.append(
							{k: a.get(k) for k in ("type", "url", "title", "start_index", "end_index")}
						)
			if str(i.get("type") or "").endswith("search_call"):
				row.setdefault("search_calls", []).append({k: i.get(k) for k in ("status", "action")})
		row["text"] = "\n".join(texts)[:3000]
		row["annotations"] = cites[:20]
		row["usage"] = j.get("usage")
		out.append(row)
		print(json.dumps({k: row.get(k) for k in ("tools", "status", "item_types", "seconds")}), flush=True)
		# after every call, so the rows already paid for survive a crash on the next
		Path("out/search_probe.json").write_text(json.dumps(out, indent=1))
	Path("out/search_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
	main()
