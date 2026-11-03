"""Ask Muse Spark about each case from its code alone (corpus/ask_model.py), on Modal with the preflight-meta secret.

	modal run --env main python/preflight/corpus/ask_model_app.py::ask --cases python/preflight/corpus/data/ask_model_cases.json

Each case's files come from the runner's own fetcher (runner/sources.py), the same files the script map would get.
Where the runner refuses a repo (over its size limits), the model still gets a fair look: the entrypoint, the files
the command's arguments name and the entrypoint's own directory, read from a shallow clone; the answer says so.
Stops before a call would pass the dollar cap. Writes out/ask_model.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "preflight-ask-model"
MAX_USD = 2.0  # cited: PM, Nov 2, item 2 (Meta credit, cap $2)
# guess: a case's worst cost with its re-ask (60k characters in, 4k tokens out, twice) at harness/spark.py's prices, with room
WORST_CALL_USD = 0.25

# review, Nov 2: in the container this file sits at /root/<name>.py, with no repo above it; the local paths below are
# only read where the app is launched
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3] if len(_HERE.parents) > 3 else Path("/root")

app = modal.App(APP_NAME)
image = (
	modal.Image.debian_slim(python_version="3.11")
	.apt_install("git")
	.pip_install("pydantic>=2.8,<3", "numpy>=1.26", "httpx>=0.27")
	.env({"PREFLIGHT_ZOO": "/root/zoo"})  # review, Nov 2: the zoo samples, as runner/modal_app.py ships them
	.add_local_dir(str(ROOT / "zoo"), "/root/zoo")
	.add_local_python_source("preflight")
)


def _command(entrypoint: str, args: list[str]) -> str:
	import shlex

	return " ".join(["python", shlex.quote(entrypoint), *(shlex.quote(a) for a in args)])


def _narrow(uri: str, commit: str, entrypoint: str, args: list[str]) -> dict[str, str]:
	"""The entrypoint, the files the arguments name and the entrypoint's directory, from a shallow clone."""
	import subprocess
	import tempfile

	from preflight.corpus.ask_model import SUFFIXES

	tmp = Path(tempfile.mkdtemp())
	run = lambda *a: subprocess.run(a, check=True, capture_output=True, timeout=120).stdout  # noqa: E731
	run(
		"git",
		"clone",
		"--depth",
		"1",
		"--no-tags",
		"--filter=blob:limit=8m",
		"--no-checkout",
		uri,
		str(tmp / "r"),
	)
	run("git", "-C", str(tmp / "r"), "fetch", "--depth", "1", "origin", commit)
	names = run("git", "-C", str(tmp / "r"), "ls-tree", "-r", "--name-only", commit).decode().splitlines()
	here = str(Path(entrypoint).parent)
	named = {a for a in args if not a.startswith("-")}
	want = [
		n
		for n in names
		if n == entrypoint
		or (str(Path(n).parent) == here and n.endswith(SUFFIXES))
		or any(
			"/" in a and n.endswith((a, a + ".yaml", a + ".yml", a + ".json")) for a in named
		)  # torchtune names configs bare; a bare word ("random", "500") is no file name
	]
	out = {}
	for n in want:
		try:
			out[n] = run("git", "-C", str(tmp / "r"), "show", f"{commit}:{n}").decode("utf-8")
		except (subprocess.CalledProcessError, UnicodeDecodeError):
			continue
	return out


@app.function(image=image, secrets=[modal.Secret.from_name("preflight-meta")], timeout=40 * 60, cpu=2)
def ask_all(cases: list[dict], max_usd: float = MAX_USD) -> dict:
	from preflight.corpus import ask_model as am
	from preflight.harness.spark import ModelClient
	from preflight.probes.local import InvalidSubmission
	from preflight.runner.sources import fetch

	client = ModelClient.meta(model=am.MODEL, timeout_s=180.0, max_output_tokens=4096)
	rows, spent = [], 0.0
	for c in cases:
		row: dict = {"id": c["id"], "set": c["set"], "gpu": c["gpu"], "prompt": am.PROMPT_VERSION}
		if spent + WORST_CALL_USD > max_usd:
			row["skipped"] = f"the ${max_usd:.2f} cap: ${spent:.4f} spent"
			rows.append(row)
			continue
		src = c["source"]
		before = len(client.calls)
		try:  # review, Nov 2: one case's failure, its fetch included, is its own row
			try:
				got = fetch(src)
				files, entry, args, how = got.files, got.entrypoint, got.args, "the runner's fetch"
			except InvalidSubmission as exc:
				if src.get("kind") != "repo":
					raise
				files = _narrow(src["uri"], src["commit"], src["entrypoint"], src.get("args") or [])
				entry, args = src["entrypoint"], src.get("args") or []
				how = f"narrowed: the runner refused the repo ({str(exc)[:120]})"
			if src.get("kind") == "sample":
				# review, Nov 2: a zoo sample's own measured outputs (record_*.json and the like) stay out, as the
				# runner leaves them out with runner.hide_sample_outputs on; grpo_trl's answer read its own record
				from preflight.runner.sources import _SAMPLE_OUTPUT

				hidden = [k for k in files if _SAMPLE_OUTPUT.fullmatch(k)]
				files = {k: v for k, v in files.items() if k not in hidden}
				how += f", {len(hidden)} measured output files left out" if hidden else ""
			row["files"] = {"how": how, "n": len(files)}
			ans, _ = client.structured(
				am.messages(files, entry, _command(entry, args), c["gpu"], c.get("steps")),
				am.Answer,
				purpose="ask_model",
				reasoning_effort="medium",
			)
			row["answer"] = ans.model_dump()
		except Exception as exc:  # one case's failure is its own row
			row["error"] = f"{type(exc).__name__}: {exc}"[:400]
		calls = client.calls[before:]
		row["usd"] = round(sum(k.cost_usd for k in calls), 6)
		row["tokens"] = [sum(k.input_tokens for k in calls), sum(k.output_tokens for k in calls)]
		spent += row["usd"]
		rows.append(row)
	return {
		"prompt": am.PROMPT_VERSION,
		"prompt_hash": am.prompt_hash(),
		"model": am.MODEL,
		"usd": spent,
		"rows": rows,
	}


@app.local_entrypoint()
def ask(cases: str, max_usd: float = MAX_USD, only: str = ""):
	"""--only takes a comma-separated list of case ids (the others are left out)."""
	t0 = time.monotonic()
	data = json.loads(Path(cases).read_text())
	keep = {s for s in only.split(",") if s}
	todo = [c for c in data["cases"] if not keep or c["id"] in keep]
	got = ask_all.remote(todo, max_usd)
	got["wall_s"] = round(time.monotonic() - t0, 1)
	Path("out").mkdir(exist_ok=True)
	Path("out/ask_model.json").write_text(json.dumps(got, indent=1))
	print(json.dumps({k: v for k, v in got.items() if k != "rows"}))
	for r in got["rows"]:
		print(r["id"], r.get("answer") or r.get("error") or r.get("skipped"))
