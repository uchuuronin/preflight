"""D5 parity: D's notebook preview (web/src/addons/notebook/convert.ts) and A's converter agree.

The fixtures' expected outputs were produced by A's code. convert.test.ts holds D's port to them;
this holds A's code to them, so a change on either side fails a test. Skipped until A's
converter (preflight.runner.sources) is in the branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

sources = pytest.importorskip("preflight.runner.sources", reason="A's notebook converter is not merged yet")

FIX = Path(__file__).resolve().parents[1] / "web/src/addons/notebook/fixtures"


def _named():
	return sorted(p for p in FIX.glob("*.ipynb"))


@pytest.mark.parametrize("path", _named(), ids=lambda p: p.name)
def test_a_converts_each_fixture_as_recorded(path: Path):
	expected = json.loads(path.with_name(path.stem + ".expected.json").read_text())
	text = path.read_text()
	script = sources._notebook_to_script(text)
	assert script == expected["script"]
	assert sources.notebook_requirements(script) == expected["requirements"]
	fetched = sources.fetch({"kind": "notebook", "files": {path.name: text}})
	assert fetched.files == expected["files"]
	assert fetched.entrypoint == expected["entrypoint"]


def test_a_converts_the_random_notebooks_as_recorded():
	data = json.loads((FIX / "random.expected.json").read_text())
	for i, case in enumerate(data["cases"]):
		if "error" in case:
			with pytest.raises(sources.InvalidSubmission):
				sources._notebook_to_script(case["text"])
			continue
		script = sources._notebook_to_script(case["text"])
		assert script == case["script"], i
		assert sources.notebook_requirements(script) == case["requirements"], i


def test_a_runs_what_d_sends_as_it_would_the_original():
	"""D leaves outputs and other cells' text out of what it sends; A must run the same script."""
	random_cases = json.loads((FIX / "random.expected.json").read_text())["cases"]
	sent = json.loads((FIX / "sent.expected.json").read_text())["cases"]
	assert len(sent) > 200
	for case in sent:
		if case["from"].startswith("random#"):
			original = random_cases[int(case["from"].removeprefix("random#"))]["text"]
		else:
			original = (FIX / case["from"]).read_text()
		fetched = sources.fetch({"kind": "notebook", "files": {case["name"]: case["sent"]}})
		assert fetched.files[fetched.entrypoint] == sources._notebook_to_script(original), case["from"]


def test_a_accepts_every_name_d_sends():
	text = (FIX / "no_requirements.ipynb").read_text()
	for given, sent in json.loads((FIX / "names.json").read_text())["names"].items():
		fetched = sources.fetch({"kind": "notebook", "files": {sent: text}})
		assert fetched.entrypoint == sent.removesuffix(".ipynb") + ".py", given
