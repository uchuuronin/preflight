"""Nov 4: a probe's spec on its sandbox's stdin. Modal's writer buffers at most 16 MiB between drains; this repository's
own repo check-in (711 files, a 16.96 MB spec) raised BufferError on its one write and every probe ended without a
result. The spec now goes in pieces with a drain after each."""

import pytest

from preflight.runner.sandbox_runner import STDIN_CHUNK, write_stdin

LIMIT = 16 * 2**20  # Modal's buffer, as ops_app.py::sandbox_stdin_check measured it


class ModalLikeStdin:
	def __init__(self):
		self.buffered, self.received, self.calls, self.eof = 0, [], [], False

	def write(self, text):
		self.calls.append("write")
		n = len(text.encode("utf-8"))
		if self.buffered + n > LIMIT:
			raise BufferError("Buffer size exceed limit. Call drain to flush the buffer.")
		self.buffered += n
		self.received.append(text)

	def drain(self):
		self.calls.append("drain")
		self.buffered = 0

	def write_eof(self):
		self.calls.append("eof")
		self.eof = True


def test_one_write_of_this_repos_spec_overflows_the_buffer():
	s = ModalLikeStdin()
	with pytest.raises(BufferError):
		s.write("a" * 16_957_300)


@pytest.mark.parametrize("n", [16_957_300, 24_000_000, 40_000_000])
def test_a_spec_past_the_buffer_arrives_whole_in_pieces(n):
	s = ModalLikeStdin()
	text = "a" * n
	write_stdin(s, text)
	assert "".join(s.received) == text and s.eof and s.calls[-2:] == ["eof", "drain"]


def test_wide_characters_stay_under_the_buffer():
	s = ModalLikeStdin()
	text = "\U0001f600" * (STDIN_CHUNK + 10)  # 4 bytes a character
	write_stdin(s, text)
	assert "".join(s.received) == text


def test_a_small_spec_is_one_write_as_before():
	s = ModalLikeStdin()
	write_stdin(s, '{"probe_id": "p1"}')
	assert s.calls == ["write", "drain", "eof", "drain"] and s.received == ['{"probe_id": "p1"}']
	s = ModalLikeStdin()
	write_stdin(s, "")
	assert s.calls == ["eof", "drain"]


def test_an_agent_runs_job_goes_in_pieces_too():
	"""Review, Nov 4: an agent run's job carries the same repo files as a probe's spec."""
	from preflight.runner.agent_jobs import _SandboxProc

	stdin = ModalLikeStdin()
	proc = type("P", (), {"stdin": stdin})()
	sb = type("SB", (), {"exec": lambda self, *a, **k: proc})()
	job = {"timeout_s": 60, "files": {"big.py": "a" * 17_000_000}}
	_SandboxProc(sb, job)
	assert stdin.eof and len("".join(stdin.received)) > 17_000_000


def test_the_deploy_gates_sandbox_check_requires_a_repo_sized_probe_to_measure():
	"""The startup check's ok needs both its tiny probe and its repo-sized one (over Modal's 16 MiB buffer) to measure."""
	import ast
	from pathlib import Path

	src = (Path(__file__).resolve().parents[1] / "python/preflight/runner/ops_app.py").read_text()
	tree = ast.parse(src)
	pad = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
		and any(getattr(t, "id", None) == "BIG_SPEC_PAD" for t in n.targets))  # fmt: skip
	assert ast.literal_eval(pad) > LIMIT
	fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "startup_check")
	body = ast.get_source_segment(src, fn)
	assert 'sb.get("big_probe_status") == "ok"' in body and 'sb.get("big_probe_steps", 0) > 0' in body


def test_a_repo_sized_spec_runs_through_the_sandbox_runner_in_pieces():
	"""SandboxRunner's own call path with a Modal-like stdin: the big spec arrives whole and the probe's result is read."""
	from preflight.probes.local import ProbeSpec
	from preflight.runner import sandbox_runner as SR
	from preflight.runner.wire import spec_to_wire

	stdin = ModalLikeStdin()
	spec = ProbeSpec(
		probe_id="big", files={"train.py": "x = 1\n", "docs/pad.txt": "a" * 17_500_000}, entrypoint="train.py"
	)
	assert len(spec_to_wire(spec)) > LIMIT
	SR.write_stdin(stdin, spec_to_wire(spec))
	assert "".join(stdin.received) == spec_to_wire(spec) and stdin.eof
