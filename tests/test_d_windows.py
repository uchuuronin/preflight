"""Windows guards on the CLI's path (D, Nov 3; the gaps listed in docs/requests/from-d.md).

Each test takes away what Windows lacks (`os.geteuid`, `os.killpg`, `resource`, the `O_` flags) or
says the platform is Windows, and checks the guarded code still does its job; the Linux path is
checked alongside, unchanged. The whole path is proved on a real Windows runner by
`.github/workflows/d-windows-watch.yml`.
"""

import os
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from preflight.cli import main as cli
from preflight.probes import local
from preflight.recorder import inflight
from preflight.recorder import run as recorder_run

ROOT = Path(__file__).resolve().parents[1]


# 1 and 4: no os.geteuid on Windows, so no probe uid and no root caches
def test_no_geteuid_means_no_root_caches_and_no_probe_uid(monkeypatch):
	monkeypatch.delattr(os, "geteuid", raising=False)
	assert cli._run_caches_when_root() is None
	assert local._probe_uid() is None


def test_as_a_user_on_linux_nothing_changes(monkeypatch):
	monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
	assert cli._run_caches_when_root() is None
	assert local._probe_uid() is None


# 2: Windows has no time-zone database; the package brings one there only
def test_tzdata_is_a_dependency_on_windows_only():
	deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
	tz = [d for d in deps if d.startswith("tzdata")]
	assert tz and "sys_platform == 'win32'" in tz[0]


# 3: no process groups on Windows
def test_kill_tree_uses_taskkill_on_windows_while_alive_and_nothing_after(monkeypatch):
	monkeypatch.delattr(os, "killpg", raising=False)
	calls = []
	monkeypatch.setattr(subprocess, "run", lambda args, **kw: calls.append(args))
	local.kill_tree(4321, alive=True)
	assert calls == [["taskkill", "/F", "/T", "/PID", "4321"]]
	local.kill_tree(4321, alive=False)  # exited: its pid may be someone else's now
	assert len(calls) == 1


def test_kill_tree_on_linux_kills_the_group(monkeypatch):
	got = []
	monkeypatch.setattr(os, "killpg", lambda pid, sig: got.append((pid, sig)), raising=False)
	local.kill_tree(55, alive=False)
	assert got == [(55, signal.SIGKILL)]


def test_kill_tree_without_taskkill_doesnt_raise(monkeypatch):
	monkeypatch.delattr(os, "killpg", raising=False)

	def missing(*a, **k):
		raise FileNotFoundError("taskkill")

	monkeypatch.setattr(subprocess, "run", missing)
	local.kill_tree(1, alive=True)


# 5: no `resource` module on Windows
def test_set_limits_without_resource_applies_nothing(monkeypatch):
	monkeypatch.setitem(sys.modules, "resource", None)  # import raises ImportError
	assert recorder_run.set_limits() == {}


# 6: "/tmp" isn't a folder on Windows
def test_the_sandbox_default_is_the_temp_folder_on_windows_and_tmp_elsewhere():
	assert local.sandbox_default("nt") == __import__("tempfile").gettempdir()
	assert local.sandbox_default("posix") == "/tmp"


# 7: no O_NOFOLLOW or O_NONBLOCK on Windows
def test_reading_a_probe_file_without_the_unix_flags(monkeypatch, tmp_path):
	f = tmp_path / "result.json"
	f.write_text('{"ok": 1}')
	link = tmp_path / "link.json"
	link.symlink_to(f)
	monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
	monkeypatch.delattr(os, "O_NONBLOCK", raising=False)
	assert local._read_untrusted(f) == '{"ok": 1}'
	with pytest.raises(OSError):
		local._read_untrusted(link)  # still never through a link


def test_reading_a_probe_file_on_linux_refuses_a_link(tmp_path):
	f = tmp_path / "result.json"
	f.write_text("x")
	link = tmp_path / "link.json"
	link.symlink_to(f)
	with pytest.raises(OSError):
		local._read_untrusted(link)


# 8: the agent sampler's kill, without process groups
def test_the_agent_samplers_kill_on_windows(monkeypatch):
	from preflight.recorder import agent_sampler

	monkeypatch.delattr(os, "killpg", raising=False)
	calls = []
	monkeypatch.setattr(subprocess, "run", lambda args, **kw: calls.append(args))

	class Proc:
		pid, killed = 77, False

		def poll(self):
			return None if not self.killed else -9

		def kill(self):
			self.killed = True

		def wait(self, timeout=None):
			return -9

	p = agent_sampler._LocalProc.__new__(agent_sampler._LocalProc)
	p.p = Proc()
	p.kill()
	assert calls == [["taskkill", "/F", "/T", "/PID", "77"]] and p.p.killed


# 9: the probe's environment keeps what Windows needs, and swaps the user's folders for its home
def test_windows_env_keeps_systemroot_and_moves_the_users_folders(tmp_path):
	src = {"SYSTEMROOT": r"C:\Windows", "USERNAME": "chiya", "APPDATA": r"C:\Users\chiya\AppData\Roaming",
		"USERPROFILE": r"C:\Users\chiya", "META_API_KEY": "secret", "CUDA_CACHE_PATH": r"E:\pf9b\cuda-cache"}  # fmt: skip
	env = local.windows_env(tmp_path, src)
	assert env["SYSTEMROOT"] == r"C:\Windows" and env["USERNAME"] == "chiya"
	assert env["CUDA_CACHE_PATH"] == r"E:\pf9b\cuda-cache"
	assert env["USERPROFILE"] == str(tmp_path) and env["APPDATA"].startswith(str(tmp_path))
	assert env["TEMP"] == env["TMP"] == str(tmp_path / "tmp")
	assert "META_API_KEY" not in env


def test_scrubbed_env_adds_the_windows_variables_only_on_windows(monkeypatch, tmp_path):
	monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
	assert "SYSTEMROOT" not in local.scrubbed_env(tmp_path)  # Linux: as before
	monkeypatch.setattr(local.os, "name", "nt")
	try:
		env = local.scrubbed_env(tmp_path)
	finally:
		monkeypatch.undo()
	assert env["SYSTEMROOT"] == r"C:\Windows" and env["USERPROFILE"] == str(tmp_path)


# 10: the live record's replace while a browser has it open
def test_a_running_write_retries_then_skips_and_the_final_write_raises(monkeypatch, tmp_path):
	monkeypatch.setattr(inflight.time, "sleep", lambda s: None)
	path = tmp_path / "live.json"
	real = os.replace
	fails = {"n": 2}

	def flaky(a, b):
		if fails["n"] > 0:
			fails["n"] -= 1
			raise PermissionError(13, "in use")
		real(a, b)

	monkeypatch.setattr(inflight.os, "replace", flaky)
	assert inflight._atomic(path, '{"a": "–"}', final=False) is True  # landed on the third try
	assert path.read_text(encoding="utf-8") == '{"a": "–"}'
	fails["n"] = 99
	assert inflight._atomic(path, "{}", final=False) is False  # skipped; the run goes on
	with pytest.raises(PermissionError):
		inflight._atomic(path, "{}")  # the final write says so


def test_the_scripts_files_are_written_as_utf8(monkeypatch, tmp_path):
	# Linux's locale is UTF-8 already, so check the encoding is asked for, not left to the locale
	seen = []
	real = Path.write_text

	def spy(self, data, encoding=None, *a, **k):
		seen.append(encoding)
		return real(self, data, encoding, *a, **k)

	monkeypatch.setattr(Path, "write_text", spy)
	local._write_files(tmp_path, {"train.py": "print('café ≈ ok')\n"})
	assert seen == ["utf-8"]
	assert (tmp_path / "train.py").read_bytes() == "print('café ≈ ok')\n".encode()


# the probe's own output: a pipe on Windows is cp1252 unless told otherwise
def test_the_probe_prints_utf8_on_windows():
	assert local.windows_env(Path("h"), {})["PYTHONIOENCODING"] == "utf-8"


# the child's own replaces: the result (must land) and the progress (may be skipped)
def _flaky_replace(monkeypatch, mod, n):
	real = os.replace
	fails = {"n": n}

	def flaky(a, b):
		if fails["n"] > 0:
			fails["n"] -= 1
			raise PermissionError(13, "in use")
		real(a, b)

	monkeypatch.setattr(mod.os, "replace", flaky)
	monkeypatch.setattr(mod.time, "sleep", lambda s: None)
	return fails


def test_the_probes_result_write_retries_then_raises(monkeypatch, tmp_path):
	fails = _flaky_replace(monkeypatch, recorder_run, 3)
	out = str(tmp_path / "result.json")
	recorder_run._write(out, {"a": "≈"})
	assert Path(out).read_text(encoding="utf-8") == '{"a": "\\u2248"}'
	fails["n"] = 99
	with pytest.raises(PermissionError):
		recorder_run._write(out, {})


def test_the_progress_write_retries_then_skips(monkeypatch, tmp_path):
	runtime = pytest.importorskip("preflight.recorder.runtime")
	fails = _flaky_replace(monkeypatch, runtime, 2)
	dst = str(tmp_path / "progress.json")
	(tmp_path / "progress.json.tmp").write_text("{}")
	assert runtime._replace(dst + ".tmp", dst, final=False) is True
	fails["n"] = 99
	(tmp_path / "progress.json.tmp").write_text("{}")
	assert runtime._replace(dst + ".tmp", dst, final=False) is False
