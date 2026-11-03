"""Native Windows (D's ten gaps, from-d.md, Nov 3): each guard, then `preflight run` and `preflight watch` end to
end with the POSIX-only calls taken away in this process. These run on Linux; D's Windows workflow runs the same
commands on a real Windows runner."""

import contextlib
import os
import subprocess
import sys
from datetime import UTC
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


@pytest.fixture
def no_posix(monkeypatch):
	"""What Windows' os module lacks, gone; taskkill recorded instead of run."""
	for name in ("geteuid", "killpg", "O_NOFOLLOW", "O_NONBLOCK"):
		monkeypatch.delattr(os, name, raising=False)
	killed = []
	real = subprocess.run

	def run(cmd, *a, **kw):
		if isinstance(cmd, list) and cmd and cmd[0] == "taskkill":
			killed.append(cmd)
			return subprocess.CompletedProcess(cmd, 0, "", "")
		return real(cmd, *a, **kw)

	monkeypatch.setattr(subprocess, "run", run)
	return killed


# 1, 4: no euid
def test_no_euid_is_never_root(no_posix):
	from preflight.cli.main import _run_caches_when_root
	from preflight.probes.local import _probe_uid

	assert _run_caches_when_root() is None and _probe_uid() is None


# 2: no time-zone database
def test_eastern_time_without_a_zone_database_matches_the_database(monkeypatch):
	from datetime import datetime
	from zoneinfo import ZoneInfo

	import preflight.runner.queue as Q

	et = ZoneInfo("America/New_York")
	stamps = [datetime(y, m, d, h, mi, tzinfo=UTC).timestamp() for y in (2025, 2026, 2027)
		for m, d in ((1, 15), (3, 8), (3, 9), (3, 14), (7, 4), (11, 1), (11, 2), (11, 7), (11, 8), (12, 31))
		for h in (0, 4, 5, 6, 7, 12, 23) for mi in (0, 59)]  # fmt: skip
	want = [(datetime.fromtimestamp(t, et).date().isoformat(), Q.seconds_to_midnight_et(t)) for t in stamps]
	monkeypatch.setattr(Q, "_et_zone", lambda: None)
	monkeypatch.setattr(Q, "_ET", None)
	got = [(Q.et_day(t), Q.seconds_to_midnight_et(t)) for t in stamps]
	assert got == want


# 3, 8: no killpg
def test_a_process_group_is_ended_with_taskkill_where_there_is_no_killpg(no_posix):
	from preflight.probes.local import kill_group

	kill_group(1234)
	kill_group(1234, exited=True)  # already gone: nothing to find
	assert no_posix == [["taskkill", "/F", "/T", "/PID", "1234"]]


def test_the_agent_sampler_kills_through_the_same_guard(no_posix):
	from preflight.recorder.agent_sampler import _LocalProc

	p = _LocalProc.__new__(_LocalProc)
	p.p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
	try:
		p.kill()
		assert no_posix and no_posix[0][-1] == str(p.p.pid)
	finally:
		with contextlib.suppress(Exception):
			p.p.kill()
			p.p.wait(5)


# 5: no resource module
def test_the_probe_runs_without_limits_where_there_is_no_resource_module(monkeypatch):
	import builtins

	from preflight.recorder import run

	real = builtins.__import__

	def imp(name, *a, **kw):
		if name == "resource":
			raise ImportError("No module named 'resource'")
		return real(name, *a, **kw)

	monkeypatch.setattr(builtins, "__import__", imp)
	assert run.set_limits() == {}


# 7: no O_NOFOLLOW, O_NONBLOCK
def test_reading_a_probe_file_works_without_the_posix_open_flags(no_posix, tmp_path):
	from preflight.probes.local import _read_untrusted

	f = tmp_path / "result.json"
	f.write_text('{"ok": "café"}', encoding="utf-8")
	assert _read_untrusted(f) == '{"ok": "café"}'


# 9: the probe's environment on Windows
def test_the_probe_keeps_what_windows_needs_and_its_user_folders_point_into_its_home(monkeypatch, tmp_path):
	from preflight.probes import local

	monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
	monkeypatch.setenv("APPDATA", r"C:\Users\chiya\AppData\Roaming")
	monkeypatch.setattr(local.os, "name", "nt")
	env = local.scrubbed_env(tmp_path)
	assert env["SYSTEMROOT"] == r"C:\Windows"
	assert env["APPDATA"].startswith(str(tmp_path)) and env["USERPROFILE"] == str(tmp_path)
	assert env["TEMP"] == env["TMP"] == str(tmp_path / "tmp")


def test_on_linux_the_probe_environment_is_as_before(tmp_path, monkeypatch):
	from preflight.probes import local

	monkeypatch.setenv("SYSTEMROOT", "x")
	assert "SYSTEMROOT" not in local.scrubbed_env(tmp_path) and "APPDATA" not in local.scrubbed_env(tmp_path)


# 10: the live record while the browser has it open
def test_the_live_record_waits_out_a_reader_and_skips_a_write_rather_than_stop(monkeypatch, tmp_path):
	from preflight.recorder import inflight

	monkeypatch.setattr(inflight, "REPLACE_WAIT_S", 0.0)
	fails = {"n": 2}
	real = os.replace

	def replace(a, b):
		if fails["n"]:
			fails["n"] -= 1
			raise PermissionError(5, "Access is denied")
		real(a, b)

	monkeypatch.setattr(inflight.os, "replace", replace)
	rec = tmp_path / "r.json"
	assert inflight._atomic(rec, '{"t": "naïve"}') and rec.read_text(encoding="utf-8") == '{"t": "naïve"}'
	fails["n"] = 99
	assert inflight._atomic(rec, "{}") is False  # skipped, not raised: the watched run goes on
	assert rec.read_text(encoding="utf-8") == '{"t": "naïve"}' and not list(tmp_path.glob("*.tmp"))


# end to end
def test_preflight_run_writes_its_record_with_windows_os_module(no_posix, tmp_path, monkeypatch):
	pytest.importorskip("torch")
	from preflight.cli import main as cli

	script = tmp_path / "tiny_loop.py"
	script.write_text((FIX / "tiny_loop.py").read_text())
	monkeypatch.chdir(tmp_path)
	code = cli.main(["run", str(script), "--out", str(tmp_path / "records"), "--no-predictions"])
	recs = list((tmp_path / "records").glob("*.json"))
	assert code == 0 and recs
	from preflight.schema import load_record

	assert any(p.status == "ok" for p in load_record(recs[0].read_text(encoding="utf-8")).probes)


def test_preflight_watch_streams_a_live_record_with_windows_os_module(tmp_path):
	pytest.importorskip("torch")
	script = tmp_path / "loop.py"
	script.write_text((FIX / "tiny_loop.py").read_text())
	parent = ROOT / "schema" / "examples" / "finetune.json"
	# watch runs the script in its own process: take Windows' missing calls away there, then run the CLI
	boot = ("import os, sys\n"
		"for n in ('geteuid', 'killpg', 'O_NOFOLLOW', 'O_NONBLOCK'):\n    os.__dict__.pop(n, None)\n"
		"from preflight.cli.main import main\nsys.exit(main(sys.argv[1:]))\n")  # fmt: skip
	r = subprocess.run([sys.executable, "-c", boot, "watch", str(parent), str(script), "--out", str(tmp_path / "live"),
		"--window", "2"], capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONPATH": str(ROOT / "python")})  # fmt: skip
	assert r.returncode == 0, r.stderr[-2000:]
	assert list((tmp_path / "live").glob("*.json"))
