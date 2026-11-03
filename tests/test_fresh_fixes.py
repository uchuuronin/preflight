"""The fresh scripts' fixes (Nov 3, before the freeze): a terminal for probes (F3, probes.tty), login-only notebook
cells skipped (F2, probes.skip_login_cells) and their error labelled right, and a newer torch said plainly (F1,
probes.torch_hint)."""

import json
import os

import pytest

from preflight.recorder.errors import classify
from preflight.runner.sources import _notebook_to_script, fetch

STTY = "import os\n_, w = os.popen('stty size', 'r').read().split()\nprint('width', w)\n"


def _nb(*cells):
	return json.dumps({"cells": [{"cell_type": "code", "source": c} for c in cells]})


# F2: login cells
def test_a_login_call_is_skipped_with_the_flag_and_kept_without():
	nb = _nb("from huggingface_hub import notebook_login\n\nnotebook_login()", "import torch\nprint(torch)")
	skipped = []
	got = _notebook_to_script(nb, skip_login=True, skipped=skipped)
	assert (
		"notebook_login()" not in got
		and "from huggingface_hub import notebook_login" in got
		and skipped == [1]
	)
	compile(got, "nb.py", "exec")
	assert "notebook_login()" in _notebook_to_script(nb)  # off: as before


def test_only_hugging_faces_own_bare_login_calls_go_and_the_imports_stay():
	def conv(*cells):
		skipped = []
		return _notebook_to_script(_nb(*cells), skip_login=True, skipped=skipped), skipped

	# the review's cases (Nov 3): another import in the same cell stays usable
	got, sk = conv(
		"from huggingface_hub import notebook_login, hf_hub_download\nnotebook_login()",
		"hf_hub_download('a', 'b')",
	)
	assert sk == [1] and "import notebook_login, hf_hub_download" in got and "notebook_login()" not in got
	got, sk = conv("import huggingface_hub as hh\nhh.login()")
	assert sk == [1] and "hh.login()" not in got
	# not Hugging Face's, or with a token: kept
	for cell in ("import wandb\nwandb.login()", "wandb.login()", "client.login('u', 'p')",
		"from huggingface_hub import login\nlogin(token=os.environ['HF_TOKEN'])", "def login():\n    pass\nlogin()"):  # fmt: skip
		got, sk = conv(cell)
		assert sk == [] and got.strip() == cell.strip(), cell
	# a name imported in an earlier cell counts; one shadowed by another import doesn't
	got, sk = conv("from huggingface_hub import notebook_login", "notebook_login()")
	assert sk == [2]
	got, sk = conv(
		"from huggingface_hub import notebook_login", "from mylib import notebook_login", "notebook_login()"
	)
	assert sk == []


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
	assert got.login_skipped == 1 and "notebook_login()" not in got.files["nb.py"]
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
	import types

	from preflight.recorder.run import _newer_torch

	# torch 2.5.1's module, as the probe image has it: no accelerator (this machine's torch may be newer)
	old = types.SimpleNamespace(cuda=1, autocast=1, compile=1)
	msg = _newer_torch(
		AttributeError("module 'torch' has no attribute 'accelerator'", name="accelerator", obj=old)
	)
	assert (
		msg.startswith("This code needs a newer torch than it declares: torch.accelerator") and "2.7.1" in msg
	)
	assert _newer_torch(
		ModuleNotFoundError("No module named 'torch.nn.attention.flex'", name="torch.nn.attention.flex")
	)
	assert _newer_torch(AttributeError("'Tensor' object has no attribute 'foo'")) is None
	assert _newer_torch(AttributeError("module 'numpy' has no attribute 'float'")) is None
	# not a newer torch (review, Nov 3): a typo, a private name, code for an older torch
	assert _newer_torch(AttributeError("module 'torch.nn' has no attribute 'Relu'")) is None
	assert _newer_torch(AttributeError("module 'torch' has no attribute 'symeig'")) is None
	assert _newer_torch(ModuleNotFoundError("No module named 'torch._six'", name="torch._six")) is None
	assert _newer_torch(AttributeError("module 'torch._C' has no attribute '_cuda_getDeviceCount'")) is None


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

	# getpass, as huggingface_hub.login() and wandb.login() call it in a script (review, Nov 3: it flushed queued ^D
	# bytes and waited for the timeout); then 20 reads, each end-of-file at once
	code = ("import getpass, sys\ntry:\n    getpass.getpass('token: ')\nexcept EOFError:\n    print('getpass eof')\n"
		"print('read', repr(sys.stdin.read()))\nprint('inputs', sum(1 for _ in range(20) if not sys.stdin.readline()))\n"
		"raise SystemExit(3)\n")  # fmt: skip
	r = run_probe(ProbeSpec(probe_id="p", files={"t.py": code}, entrypoint="t.py", warmup=1, timed=2, timeout_s=60,
		tty=True))  # fmt: skip
	assert r.probe.status != "timeout" and "read ''" in r.log_tail and "inputs 20" in r.log_tail, r.log_tail
	assert "getpass eof" in r.log_tail


# D's note: the Windows probe environment
def test_the_windows_probe_environment_prints_utf8(monkeypatch, tmp_path):
	from preflight.probes import local

	monkeypatch.setattr(local.os, "name", "nt")
	assert local.scrubbed_env(tmp_path)["PYTHONIOENCODING"] == "utf-8"
	monkeypatch.setattr(local.os, "name", os.name)


def test_the_full_runs_bare_run_gets_the_same_terminal_with_the_rigs_tty(tmp_path):
	import sys

	from preflight.probes import local
	from preflight.recorder.ground_truth import _stream

	(tmp_path / "t.py").write_text(STTY)
	cmd = [sys.executable, str(tmp_path / "t.py")]
	code, tail = _stream(cmd, env=dict(os.environ), cwd=tmp_path, timeout_s=30, tty=True)
	assert code == 0 and "width 120" in tail and not local._TTYS
	code, tail = _stream(cmd, env=dict(os.environ), cwd=tmp_path, timeout_s=30)
	assert code != 0  # as before: no terminal
