"""The data-use notice and A's runner decide "Muse Spark reads your code" from the same flag.

The notice (web/src/app/DataUse.tsx) shows the line when `harness.spark_script_map` is on; the
runner sends a job's code to Muse Spark for the script map only under that same flag (A's
pipeline.py, f50999d). Chiya's decision, Nov 1.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _notice_flag() -> str:
	src = (ROOT / "web/src/app/DataUse.tsx").read_text()
	m = re.search(r'flag\("([a-z0-9_.]+)"\)\) lines\.push\(t\("datause\.script_map"\)\)', src)
	assert m, "the notice no longer gates its script-map line on a flag"
	return m.group(1)


def test_the_notice_and_the_runner_read_the_same_flag():
	flag = _notice_flag()
	assert flag == "harness.spark_script_map"
	pipeline = (ROOT / "python/preflight/runner/pipeline.py").read_text()
	gate = re.search(r"if deps\.spark is not None[^\n]*", pipeline)
	assert gate, "the runner's Muse Spark call moved; find its gate and check it here"
	used = re.findall(r'flag\("([a-z0-9_.]+)"\)', gate.group(0))
	if not used:
		pytest.skip("waits on A's runner change (f50999d on ws-a): the gate reads no flag on this branch yet")
	assert used == [flag], f"the runner gates Muse Spark on {used}, the notice on {flag}"


def test_the_flag_exists():
	assert "harness.spark_script_map" in json.loads((ROOT / "flags.json").read_text())


def test_the_notice_and_the_runner_read_the_same_triage_flag():
	"""Spark Open triage of probe logs (A, 2d7cabd): the notice's line and A's gate, one flag."""
	src = (ROOT / "web/src/app/DataUse.tsx").read_text()
	m = re.search(r'flag\("([a-z0-9_.]+)"\)\) lines\.push\(t\("datause\.triage"\)\)', src)
	assert m, "the notice no longer gates its triage line on a flag"
	if not (ROOT / "python/preflight/harness/triage.py").exists():
		pytest.skip("waits on A's triage (2d7cabd on ws-a): harness/triage.py isn't on this branch yet")
	runner = ROOT / "python/preflight/runner"
	gated = []
	for path in runner.glob("*.py"):
		text = path.read_text()
		# Every function that calls triage_probe must check a flag first; collect the flags it reads.
		for fn in re.finditer(r"\ndef (\w+)\(.*?(?=\ndef |\Z)", text, re.S):
			body = fn.group(0)
			if "triage_probe(" in body or "deps.triage" in body:
				flags = re.findall(r'flag\("([a-z0-9_.]+)"\)', body)
				assert flags, f"{path.name}:{fn.group(1)} calls triage with no flag check"
				gated.extend(flags)
	assert gated, "no runner function calls triage; find its gate and check it here"
	# Round five (PM, Nov 1): A's Spark diagnosis also calls Spark Open (its claim check) under
	# harness.spark_diagnosis. The rule is that production never sends a visitor's data under a flag
	# the notice doesn't speak for: every gating flag has a notice line, or is off in production.
	# Every flag named in a line's condition counts (`if (flag("a") || flag("b")) lines.push(...)`).
	notice = {
		f
		for cond in re.findall(r"if \(([^\n]*?)\) lines\.push", src)
		for f in re.findall(r'flag\("([a-z0-9_.]+)"\)', cond)
	}
	prod = json.loads((ROOT / "flags.production.json").read_text())
	assert m.group(1) in gated, f"the triage notice reads {m.group(1)}, which no runner gate reads"
	silent = {f for f in gated if f not in notice and prod.get(f) is True}
	assert not silent, f"on in production without a data-use line: {silent}"


# ---- Muse Spark's diagnosis, its claim check and its search (PM, Nov 2): one notice line each,
# shown only under its flag, and never quieter than the runner that sends the data.


def _notice_conditions() -> dict[str, set[str]]:
	"""Each data-use line's key, and every flag in the condition that shows it."""
	src = (ROOT / "web/src/app/DataUse.tsx").read_text()
	out: dict[str, set[str]] = {}
	for cond, key in re.findall(r"if \((.*?)\) lines\.push\(t\(\s*\"(datause\.[a-z_]+)\"", src):
		out[key] = set(re.findall(r'flag\("([a-z0-9_.]+)"\)', cond))
	return out


def _runner_gates(call: str) -> set[str]:
	"""The flags read by every runner function that makes `call`."""
	gated: set[str] = set()
	for path in (ROOT / "python/preflight/runner").glob("*.py"):
		text = path.read_text()
		for fn in re.finditer(r"\ndef (\w+)\(.*?(?=\ndef |\Z)", text, re.S):
			body = fn.group(0)
			if call in body:
				flags = set(re.findall(r'flag\("([a-z0-9_.]+)"\)', body))
				assert flags, f"{path.name}:{fn.group(1)} calls {call} with no flag check"
				gated |= flags
	return gated


@pytest.mark.parametrize(
	"key,flag",
	[
		("datause.diagnosis", "harness.spark_diagnosis"),
		("datause.claim_check", "harness.spark_claim_check"),
		("datause.search", "harness.spark_search"),
		("datause.fix", "c.fix_loop"),
	],
)
def test_each_line_has_its_flag_off_by_default(key, flag):
	cond = _notice_conditions()
	assert key in cond, f"the notice has no line {key}"
	assert flag in cond[key], f"{key} isn't shown under {flag}"
	assert json.loads((ROOT / "flags.json").read_text()).get(flag) is False, (
		f"{flag} must be in flags.json, off"
	)


def test_the_diagnosis_line_reads_the_runners_flag():
	gates = _runner_gates("diagnose_record(")
	assert gates, "no runner function calls the diagnosis; find its gate and check it here"
	assert gates <= _notice_conditions()["datause.diagnosis"], f"the runner gates the diagnosis on {gates}"


def test_the_claim_check_line_shows_whenever_the_runner_checks():
	"""Spark Open gets each finding's numbers whenever the runner runs the claim check, so the line
	shows under every flag that gates it (today harness.spark_diagnosis, in pipeline._diagnose)."""
	gates = _runner_gates("check_diagnosis(")
	assert gates, "no runner function calls the claim check; find its gate and check it here"
	cond = _notice_conditions()["datause.claim_check"]
	assert gates & cond, f"the claim check runs under {gates}; the notice's line under {cond}"
	missing = {g for g in gates if g.startswith("harness.spark") and g not in cond}
	assert not missing, f"the claim check runs under {missing} with no notice line"


def test_the_search_line_shows_whenever_the_runner_searches():
	"""A search by Muse Spark is a model tool A's harness would add (web_search); the retrieval
	lens's own index search (probes, recorder) isn't one."""
	harness = ROOT / "python/preflight/harness"
	searching = [p.name for p in harness.rglob("*.py") if re.search(r"web_search|search_web", p.read_text())]
	if not searching:
		pytest.skip("no Muse Spark search in A's harness yet: the line waits on harness.spark_search")
	# D's runner entry point is ground_failure (harness/search_grounding.py), called from pipeline._search
	gates = _runner_gates("web_search") | _runner_gates("search_web") | _runner_gates("ground_failure(")
	assert "harness.spark_search" in gates, (
		f"the runner searches under {gates}, the notice under harness.spark_search"
	)
