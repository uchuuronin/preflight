"""The fresh scripts' fixes (Nov 3, before the freeze): a terminal for probes (F3, probes.tty), login-only notebook
cells skipped (F2, probes.skip_login_cells) and their error labelled right, and a newer torch said plainly (F1,
probes.torch_hint)."""

import json
import os

import pytest

from preflight.recorder.errors import classify
from preflight.runner.sources import _login_cell, _notebook_to_script, fetch

STTY = "import os\n_, w = os.popen('stty size', 'r').read().split()\nprint('width', w)\n"


def _nb(*cells):
	return json.dumps({"cells": [{"cell_type": "code", "source": c} for c in cells]})


# F2: login cells
def test_a_login_only_cell_is_skipped_with_the_flag_and_kept_without():
	nb = _nb("from huggingface_hub import notebook_login\n\nnotebook_login()", "import torch\nprint(torch)")
	skipped = []
	assert "notebook_login" not in _notebook_to_script(nb, skip_login=True, skipped=skipped) and skipped == [
		1
	]
	assert "notebook_login()" in _notebook_to_script(nb)  # off: as before
	assert _login_cell("import huggingface_hub\nhuggingface_hub.login()")
	assert not _login_cell(
		"from huggingface_hub import notebook_login\nnotebook_login()\nmodel = load()"
	)  # more


def test_the_fetch_says_how_many_it_skipped(monkeypatch):
	import preflight.runner.flags as F

	monkeypatch.setattr(F, "flag", lambda name: name == "probes.skip_login_cells")
	got = fetch(
		{
			"kind": "notebook",
			"files": {
				"nb.ipynb": _nb("from huggingface_hub import notebook_login\nnotebook_login()", "x = 1")
			},
		}
	)
	assert got.login_skipped == 1 and "notebook_login" not in got.files["nb.py"]
	from preflight.runner.pipeline import _reading_findings

	f = next(f for f in _reading_findings(got) if f.id == "source.login_skipped")
	assert "Preflight skipped your login cell" in f.detail and "public models need no login" in f.detail


def test_notebook_login_outside_a_notebook_isnt_a_missing_package():
	exc = ImportError("The `notebook_login` function can only be used in a notebook (Jupyter or Colab) and you need "
		"the `ipywidgets` module: `pip install ipywidgets`.")  # fmt: skip
	c = classify(exc, steps=0, in_eval=False)
	assert c.code == "notebook_only" and "only works inside a notebook" in c.message


# F1: a newer torch
def test_a_torch_attribute_only_a_newer_torch_has_is_said_plainly():
	pytest.importorskip("torch")
	from preflight.recorder.run import _newer_torch

	msg = _newer_torch(AttributeError("module 'torch' has no attribute 'accelerator'"))
	assert (
		msg.startswith("This code needs a newer torch than it declares: torch.accelerator") and "2.7.1" in msg
	)
	assert _newer_torch(
		ModuleNotFoundError("No module named 'torch.nn.attention.flex'", name="torch.nn.attention.flex")
	)
	assert _newer_torch(AttributeError("'Tensor' object has no attribute 'foo'")) is None
	assert _newer_torch(AttributeError("module 'numpy' has no attribute 'float'")) is None


# F3: a terminal
def test_a_probe_with_the_flag_gets_a_terminal_size_and_without_it_doesnt():
	pytest.importorskip("torch")
	from preflight.probes.local import ProbeSpec, run_probe

	loop = STTY + ("import torch\nm = torch.nn.Linear(2, 2)\nopt = torch.optim.SGD(m.parameters(), lr=0.1)\n"
		"for _ in range(10):\n    m(torch.randn(1, 2)).sum().backward()\n    opt.step()\n")  # fmt: skip
	off = run_probe(
		ProbeSpec(probe_id="p", files={"t.py": loop}, entrypoint="t.py", warmup=1, timed=2, timeout_s=120)
	)
	assert (
		off.probe.status == "error" and "not enough values to unpack" in off.probe.error
	)  # F3's own failure
	on = run_probe(ProbeSpec(probe_id="p", files={"t.py": loop}, entrypoint="t.py", warmup=1, timed=2, timeout_s=120,
		tty=True))  # fmt: skip
	assert on.probe.status == "ok" and "width 120" in on.log_tail


def test_a_probe_reading_stdin_with_the_terminal_gets_end_of_file_not_a_hang():
	pytest.importorskip("torch")
	from preflight.probes.local import ProbeSpec, run_probe

	code = "import sys\nprint('read', repr(sys.stdin.read()))\nraise SystemExit(3)\n"
	r = run_probe(ProbeSpec(probe_id="p", files={"t.py": code}, entrypoint="t.py", warmup=1, timed=2, timeout_s=60,
		tty=True))  # fmt: skip
	assert r.probe.status != "timeout" and "read ''" in r.log_tail


# D's note: the Windows probe environment
def test_the_windows_probe_environment_prints_utf8(monkeypatch, tmp_path):
	from preflight.probes import local

	monkeypatch.setattr(local.os, "name", "nt")
	assert local.scrubbed_env(tmp_path)["PYTHONIOENCODING"] == "utf-8"
	monkeypatch.setattr(local.os, "name", os.name)
