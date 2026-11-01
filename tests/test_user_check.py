"""The droplet's probe-user startup check: what passes and what fails (probes/user_check.py)."""

import pytest

from preflight.probes import user_check
from preflight.probes.user_check import CHILD_IMPORTS, verdict

OK = {"exit": 0, "uid": 61000, "user": "preflight-probe", "groups": [44, 109], "compiled": 7.27,
	"imported": list(CHILD_IMPORTS), "kfd": True, "gpu": {"name": "AMD Instinct MI300X", "hip": "6.2"},
	"gpu_compiled": 1.0}  # fmt: skip


def test_passes():
	assert verdict(OK, 61000, True, [44, 109]) == []
	assert verdict({**OK, "gpu": None, "gpu_compiled": None, "kfd": None}, 61000, False, [44]) == []


def test_each_failure_is_named():
	assert verdict({**OK, "uid": 0}, 61000, False, [])[0].startswith("ran as uid 0")
	assert "GPU groups [109] missing" in verdict({**OK, "groups": [44]}, 61000, False, [44, 109])
	assert "no CPU compile" in verdict({**OK, "compiled": None}, 61000, False, [])
	assert verdict({**OK, "imported": []}, 61000, False, [])[0].startswith("imported")
	assert "no GPU compile" in verdict({**OK, "gpu_compiled": None}, 61000, True, [])
	assert "no GPU matmul" in verdict({**OK, "gpu": None}, 61000, True, [])


def test_a_crash_reports_its_stderr():
	assert verdict({"exit": 1, "stderr": "RuntimeError: no HIP device"}, 61000, True, []) == [
		"exit 1: RuntimeError: no HIP device"
	]


def test_not_root_skips_the_uid_checks():
	assert verdict({**OK, "uid": 1000, "user": "me", "groups": []}, None, False, [44]) == []


def test_a_cuda_torch_fails_the_gpu_check_where_rocm_devices_exist(monkeypatch):
	monkeypatch.setattr(user_check.os.path, "exists", lambda p: p == "/dev/kfd")
	assert "not a ROCm torch" in verdict({**OK, "gpu": {"hip": None, "cuda": "12.4"}}, 61000, True, [])
	assert "/dev/kfd not opened" in verdict({**OK, "kfd": None}, 61000, True, [])
	assert verdict(OK, 61000, True, [44, 109]) == []


def test_a_wrong_user_name_fails():
	assert "user name 'root'" in verdict({**OK, "user": "root"}, 61000, False, [])


def test_a_check_that_cant_launch_still_prints_its_line(monkeypatch, capsys):
	def boom(*a, **k):
		raise PermissionError("setgroups")

	monkeypatch.setattr(user_check, "probe_user_check", boom)
	assert user_check.main(["--uid", "61000"]) == 1
	out = capsys.readouterr().out
	assert out.startswith("USER_CHECK ") and "PermissionError: setgroups" in out


def test_the_uid_must_be_unprivileged():
	with pytest.raises(SystemExit):
		user_check.main(["--uid", "0"])
