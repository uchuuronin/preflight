"""runner.probe_without_map (off; the PM, Nov 3): when the whole source doesn't map, the entry point alone
is mapped and the job probes on, with a finding that says the map was skipped and why."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import preflight.runner.pipeline as P
from preflight.harness.static import ScriptParseError
from test_pipeline import SCRIPT, FakeRunner, deps

BROKEN = {"train.py": SCRIPT, "tools/old.py": "def broken(:\n    pass\n"}


@pytest.fixture
def on(monkeypatch):
	real = P._flag
	monkeypatch.setattr(P, "_flag", lambda name: name == "runner.probe_without_map" or real(name))


def _src(files):
	return {"source": {"kind": "upload", "files": files}}


def test_off_a_file_that_doesn_t_parse_still_ends_the_job(tmp_path):
	with pytest.raises(ScriptParseError):
		P.run_pipeline(_src(BROKEN), P._Ctx(), deps(tmp_path, FakeRunner()))


def test_on_the_job_probes_and_says_the_map_was_skipped_not_syntax_error(tmp_path, on):
	rec = P.run_pipeline(_src(BROKEN), P._Ctx(), deps(tmp_path, FakeRunner()))
	assert rec.probes and any(p.status == "ok" for p in rec.probes)
	f = next(f for f in rec.findings if f.id == "source.map_skipped")
	assert "didn't map the whole source" in f.detail and "tools/old.py" in f.detail
	assert "syntax error" not in f.title.lower()


def test_on_an_entry_point_that_doesn_t_parse_is_still_the_error(tmp_path, on):
	with pytest.raises(ScriptParseError):
		P.run_pipeline(_src({"train.py": "def broken(:\n"}), P._Ctx(), deps(tmp_path, FakeRunner()))


def test_over_two_mb_of_python_probes_the_entry_point(tmp_path, on):
	pad = "# " + "x" * 1000 + "\n"
	files = {"train.py": SCRIPT, **{f"vendor/m{i}.py": pad * 100 for i in range(25)}}
	rec = P.run_pipeline(_src(files), P._Ctx(), deps(tmp_path, FakeRunner()))
	f = next(f for f in rec.findings if f.id == "source.map_skipped")
	assert "MB of Python" in f.detail and rec.probes
