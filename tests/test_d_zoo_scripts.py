"""D's bug-zoo scripts (Oct 31): each is read by A's static reader exactly as an upload would be.

No GPU and no API calls: A's reader only parses the code. Each script must land in its lens,
expose its planted settings with their planted values, and raise the planted problems that can be
seen in the code; problems that only show when the script runs are listed in the header as probe
findings and checked by A's probes and full runs.
"""

from __future__ import annotations

import ast
import contextlib
import re
from pathlib import Path

import pytest

from preflight.harness.skeleton import read_time_findings, skeleton_graph
from preflight.harness.static import map_static

ZOO = Path(__file__).resolve().parents[1] / "zoo"

# dir, entry point, lens, planted settings {knob id: value}, findings read from the code
SCRIPTS = [
	(
		"agent_eval_spark",
		"run.py",
		"agent",
		{"max_turns": 40, "concurrency": 4, "n_tasks": 500},
		{"agent.tool_retry_loop"},
	),
	(
		"eval_sweep_reasoning",
		"run.py",
		"eval",
		{"max_new_tokens": 2048, "dump_hidden_states": True},
		{"eval.activation_dump"},
	),
	(
		"serving_vllm",
		"serve.py",
		"serving",
		{"max_model_len": 32768, "max_num_seqs": 256},
		set(),
	),
	(
		"retrieval_st",
		"train.py",
		"retrieval",
		{"batch_size": 8},
		{"retrieval.small_batch"},
	),
	(
		"gnn_pyg",
		"train.py",
		"gnn",
		{"fanout.hop1": 15, "fanout.hop2": 25, "fanout.hop3": 25},
		set(),
	),
	(
		"vision_timm",
		"train.py",
		"vision",
		{"num_workers": 2},
		{"vision.no_channels_last", "vision.cpu_augment"},
	),
]

IDS = [s[0] for s in SCRIPTS]

# Planted problems A's reader should raise but can't yet, with why. Each is checked by a strict
# xfail below, so the gap shows here until A's fix lands and the xfail then turns red.
KNOWN_GAPS = {
	("vision_timm", "vision.no_channels_last"): (
		"A's reader looks for 'channels_last' anywhere in the file, comments and docstrings "
		"included, so the header naming this finding hides it (request to A in from-d.md)"
	),
}


def _codes_in(doc: str, label: str) -> set[str]:
	m = re.search(rf"{label}: (.*)", doc)
	return set(re.findall(r"[a-z]+\.[a-z_]+", m.group(1))) if m else set()


def _read(d: str, entry: str):
	path = ZOO / d / entry
	files = {entry: path.read_text()}
	return map_static(files, entrypoint=entry), path.read_text()


@pytest.mark.parametrize(("d", "entry", "lens", "knobs", "findings"), SCRIPTS, ids=IDS)
def test_header_lists_planted_problems_and_expected_findings(d, entry, lens, knobs, findings):
	_, text = _read(d, entry)
	doc = ast.get_docstring(ast.parse(text)) or ""
	assert "BUG ZOO" in doc
	assert re.search(r"^\s+1\. ", doc, re.M), "numbered planted problems"
	read = re.search(r"Read from the code: (.*)", doc)
	assert read, "header says which findings are read from the code"
	assert _codes_in(doc, "Read from the code") == findings
	assert "Found by the probe:" in doc
	assert "Ground truth:" in doc


@pytest.mark.parametrize(("d", "entry", "lens", "knobs", "findings"), SCRIPTS, ids=IDS)
def test_lands_in_its_lens(d, entry, lens, knobs, findings):
	sm, _ = _read(d, entry)
	assert sm.lens == lens


@pytest.mark.parametrize(("d", "entry", "lens", "knobs", "findings"), SCRIPTS, ids=IDS)
def test_planted_settings_are_found_with_their_values(d, entry, lens, knobs, findings):
	sm, _ = _read(d, entry)
	found = {k.id: k.current for k in sm.knobs}
	for kid, value in knobs.items():
		assert kid in found, (kid, sorted(found))
		assert found[kid] == value, (kid, found[kid])


@pytest.mark.parametrize(("d", "entry", "lens", "knobs", "findings"), SCRIPTS, ids=IDS)
def test_planted_problems_seen_in_the_code_are_raised(d, entry, lens, knobs, findings):
	sm, _ = _read(d, entry)
	_, text = _read(d, entry)
	codes = {f.code for f in read_time_findings(sm, skeleton_graph(sm))}
	gaps = {code for (sd, code) in KNOWN_GAPS if sd == d}
	also = _codes_in(ast.get_docstring(ast.parse(text)) or "", "Also read, not planted")
	# Exact: every finding the reader raises is either planted or named in the header.
	assert codes - gaps == (findings | also) - gaps, (codes, findings, also)


@pytest.mark.parametrize(("d", "code"), list(KNOWN_GAPS), ids=[f"{d}:{c}" for d, c in KNOWN_GAPS])
def test_known_gaps_in_the_reader(d, code):
	entry = next(s[1] for s in SCRIPTS if s[0] == d)
	sm, _ = _read(d, entry)
	codes = {f.code for f in read_time_findings(sm, skeleton_graph(sm))}
	if code not in codes:
		pytest.xfail(KNOWN_GAPS[(d, code)])
	pytest.fail(f"{code} is now raised: drop it from KNOWN_GAPS")


@pytest.mark.parametrize(("d", "entry", "lens", "knobs", "findings"), SCRIPTS, ids=IDS)
def test_requirements_are_plain_pinned_pypi_lines(d, entry, lens, knobs, findings):
	req = ZOO / d / "requirements.txt"
	assert req.exists()
	for line in req.read_text().splitlines():
		line = line.split("#", 1)[0].strip()
		if not line:
			continue
		assert re.fullmatch(r"[A-Za-z0-9_.\-\[\],]+==[0-9][A-Za-z0-9.+\-]*", line), line


def _consts(path: Path) -> dict:
	"""Module-level NAME = literal assignments, read without importing the script."""
	out = {}
	for node in ast.parse(path.read_text()).body:
		if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
			with contextlib.suppress(ValueError):
				out[node.targets[0].id] = ast.literal_eval(node.value)
	return out


def test_agent_script_is_the_golden_agent_records_job():
	import json

	rec = json.loads((ZOO.parent / "schema/examples/agent.json").read_text())
	assert rec["job"]["source"]["entrypoint"] == "zoo/agent_eval_spark/run.py"
	knobs = {k["id"]: k["current"] for k in rec["knobs"]}
	c = _consts(ZOO / "agent_eval_spark/run.py")
	assert (c["MAX_TURNS"], c["CONCURRENCY"], c["N_TASKS"]) == (
		knobs["max_turns"],
		knobs["concurrency"],
		knobs["n_tasks"],
	)
	assert c["MODEL"].startswith(knobs["model"])
	# Known difference, raised with A: the golden record says 32,768; the script keeps a
	# 26,000-token budget; by Muse Spark's count it overflows at turn 11 (header), and its own check at 12.
	assert (c["CONTEXT_LIMIT"], knobs["context_limit"]) == (26_000, 32768)


def test_agent_script_stays_the_size_a_measured():
	"""The script echoes its own source every turn, so its size sets the growth. A measured the
	growth behind turn 11 on Muse Spark (Oct 31) with the 7,641-byte version; keep within 300
	bytes (about 4% of the growth) of it, or ask A to measure again."""
	size = len((ZOO / "agent_eval_spark/run.py").read_bytes())
	assert abs(size - 7641) <= 300, size


def test_agent_scripts_own_check_stops_at_turn_12_with_short_replies():
	"""Replay the history the script builds, offline: the user's task, then each turn one short
	run_python call and its result, which echoes the script's own source. Uses the script's own
	token estimate (characters of each JSON message / 4) and its own text as the echo. By Muse
	Spark's count the context passes the budget at turn 11 for a typical task (the header; derived
	from A's per-turn counts, Oct 31), so the script's own check stops it a turn later, at 12."""
	import json

	path = ZOO / "agent_eval_spark/run.py"
	harness = path.read_text()
	limit = _consts(path)["CONTEXT_LIMIT"]
	task = {
		"text": "Write a function to find the shared elements from the given two lists.",
		"test_list": [
			"assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
			"assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))",
			"assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))",
		],
	}
	system = {"role": "system", "content": "Current time: 2026-10-31T09:00:00.000000+00:00\n" + "x" * 190}
	messages = [
		{
			"role": "user",
			"content": f"{task['text']}\nYour code should pass these tests:\n" + "\n".join(task["test_list"]),
		}
	]
	code = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))\nprint(similar_elements((3, 4), (4, 5)))"
	for turn in range(1, 41):
		if sum(len(json.dumps(m)) for m in [system, *messages]) // 4 > limit:
			break
		call = {
			"id": f"call_{turn}",
			"type": "function",
			"function": {"name": "run_python", "arguments": json.dumps({"code": code})},
		}
		messages.append({"role": "assistant", "tool_calls": [call]})
		out = json.dumps({"stdout": "(4,)\n", "stderr": "", "harness": harness, "task": task["text"]})
		messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
	assert turn == 12
