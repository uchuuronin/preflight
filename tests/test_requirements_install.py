"""A submission's requirements.txt is installed in the probe, on top of the image's pinned stack."""

import pytest

from preflight.probes.local import ProbeSpec, run_probe

LOOP = """
import tomli_w, torch
m = torch.nn.Linear(4, 1)
opt = torch.optim.SGD(m.parameters(), 0.1)
print("TOMLI_W", tomli_w.dumps({"a": 1}).strip())
for _ in range(20):
	loss = m(torch.randn(8, 4)).pow(2).mean()
	opt.zero_grad(); loss.backward(); opt.step()
"""


@pytest.mark.torch
@pytest.mark.network
def test_requirements_are_installed_before_the_script_runs(tmp_path):
	files = {"train.py": LOOP, "requirements.txt": "tomli-w==1.0.0\ntorch==0.0.1\n"}
	r = run_probe(ProbeSpec(probe_id="p1", files=files, entrypoint="train.py", timeout_s=240,
		install_requirements=True, out_dir=tmp_path))  # fmt: skip
	assert r.probe.status == "ok", (r.probe.error, r.log_tail[-2000:])
	assert "TOMLI_W a = 1" in r.log_tail and "kept the image's torch" in r.log_tail


@pytest.mark.torch
def test_a_requirement_that_wont_install_is_a_missing_package_error(tmp_path):
	files = {"train.py": LOOP, "requirements.txt": "-e .\n"}
	r = run_probe(ProbeSpec(probe_id="p1", files=files, entrypoint="train.py", timeout_s=60,
		install_requirements=True, out_dir=tmp_path))  # fmt: skip
	# nothing installable: the line is skipped with a note, and the import fails as the script's own
	assert r.probe.status == "error" and "only plain PyPI" in r.log_tail


def test_pip_runs_isolated_from_the_probes_pythonpath(tmp_path, monkeypatch):
	"""An unprivileged probe's PYTHONPATH can name a directory it can't list (Modal's /root, Nov 1);
	pip walks sys.path, so it runs with -I and from its own prefix."""
	import subprocess

	from preflight.recorder import requirements

	(tmp_path / "requirements.txt").write_text("some-package-not-installed==1.0\n")
	seen = {}

	def fake_run(cmd, **kw):
		seen["cmd"], seen["cwd"] = cmd, kw.get("cwd")
		return subprocess.CompletedProcess(cmd, 0, "", "")

	monkeypatch.setattr(requirements.subprocess, "run", fake_run)
	monkeypatch.delenv("PREFLIGHT_DEPS_DIR", raising=False)
	err, _ = requirements.install(tmp_path, tmp_path / "deps", log=lambda m: None)
	assert (
		err is None
		and seen["cmd"][1:5] == ["-I", "-m", "pip", "install"]
		and seen["cwd"] == tmp_path / "deps"
	)
