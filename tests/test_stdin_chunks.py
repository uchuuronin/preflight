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
