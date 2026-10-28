"""Records on A's Modal volume: the API mounts "preflight-records" read-only and must reload it to
see what A's workers committed, without a reload ever overlapping a read in the same container
(Modal: a container sees the volume empty while it reloads, and a reload fails on open files)."""

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from preflight_api.app import create_app
from preflight_api.stores import LocalRecordStore, StoreBusy, new_share_token


class Clock:
	def __init__(self) -> None:
		self.t = 1000.0

	def __call__(self) -> float:
		return self.t


def write(root: Path, token: str, rev: int) -> None:
	(root / f"{token}.json").write_text(
		json.dumps({"schema_version": "0.3.0", "record_id": "01X", "rev": rev})
	)


def test_miss_reloads_and_then_finds_a_record_another_container_wrote(tmp_path):
	token = new_share_token()
	calls = []

	def reload() -> None:
		calls.append(1)
		write(tmp_path, token, 1)  # the commit A's worker made becomes visible on reload

	store = LocalRecordStore(tmp_path, reload=reload, clock=Clock())
	assert store.by_share_token(token)["rev"] == 1
	assert len(calls) == 1


def test_a_hit_reloads_too_so_a_revised_record_is_not_stale(tmp_path):
	token = new_share_token()
	write(tmp_path, token, 1)
	clock = Clock()
	store = LocalRecordStore(tmp_path, reload=lambda: write(tmp_path, token, 2), clock=clock)
	assert store.by_share_token(token)["rev"] == 2


def test_reloads_are_rate_limited(tmp_path):
	token = new_share_token()
	write(tmp_path, token, 1)
	calls = []
	clock = Clock()
	store = LocalRecordStore(tmp_path, reload=lambda: calls.append(1), clock=clock)
	for _ in range(5):
		store.by_share_token(token)
	assert len(calls) == 1
	clock.t += LocalRecordStore.RELOAD_MIN_S + 0.01
	store.by_share_token(token)
	assert len(calls) == 2


def test_a_failed_reload_serves_the_current_view(tmp_path):
	token = new_share_token()
	write(tmp_path, token, 1)

	def busy() -> None:
		raise RuntimeError("volume busy")

	store = LocalRecordStore(tmp_path, reload=busy, clock=Clock())
	assert store.by_share_token(token)["rev"] == 1


def test_reload_never_overlaps_a_read(tmp_path):
	token = new_share_token()
	write(tmp_path, token, 1)
	state = {"reloading": False, "overlap": False}
	real_read = Path.read_bytes

	def reload() -> None:
		state["reloading"] = True
		time.sleep(0.02)
		state["reloading"] = False

	def read_bytes(self):
		if state["reloading"]:
			state["overlap"] = True
		return real_read(self)

	store = LocalRecordStore(tmp_path, reload=reload)
	store.RELOAD_MIN_S = 0.0  # reload on every read, to force contention
	Path.read_bytes = read_bytes
	try:
		threads = [threading.Thread(target=store.by_share_token, args=(token,)) for _ in range(16)]
		for th in threads:
			th.start()
		for th in threads:
			th.join()
	finally:
		Path.read_bytes = real_read
	assert not state["overlap"]


def test_bad_or_huge_files_are_misses_not_errors(tmp_path):
	good, bad, big = new_share_token(), new_share_token(), new_share_token()
	write(tmp_path, good, 1)
	(tmp_path / f"{bad}.json").write_text("{not json")
	(tmp_path / f"{big}.json").write_bytes(b" " * (LocalRecordStore.MAX_READ_BYTES + 1))
	store = LocalRecordStore(tmp_path)
	assert store.by_share_token(good)["rev"] == 1
	assert store.by_share_token(bad) is None
	assert store.by_share_token(big) is None
	assert store.by_share_token("../" + good) is None


def test_record_responses_are_not_cached_while_a_job_can_still_revise_them(tmp_path):
	token = new_share_token()
	write(tmp_path, token, 1)
	with TestClient(create_app(records=LocalRecordStore(tmp_path)), base_url="https://testserver") as c:
		res = c.get(f"/v1/records/{token}")
		assert res.status_code == 200
		assert "no-cache" in res.headers["Cache-Control"]
		assert "public" not in res.headers["Cache-Control"]


def test_a_record_committed_just_after_a_reload_is_found_on_a_miss(tmp_path):
	# Review finding: a reload at t, A commits, a read at t + 0.6 s must still find the record.
	token = new_share_token()
	clock = Clock()
	committed = {"now": False}

	def reload() -> None:
		if committed["now"]:
			write(tmp_path, token, 1)

	store = LocalRecordStore(tmp_path, reload=reload, clock=clock)
	assert store.by_share_token(new_share_token()) is None  # a reload at t
	committed["now"] = True
	clock.t += 0.6
	assert store.by_share_token(token)["rev"] == 1


def test_a_miss_reloads_at_most_once_per_request(tmp_path):
	calls = []
	store = LocalRecordStore(tmp_path, reload=lambda: calls.append(1), clock=Clock())
	assert store.by_share_token(new_share_token()) is None
	assert len(calls) == 1  # the first reload already started after the request arrived


def test_a_stuck_reload_turns_other_reads_into_a_retryable_busy(tmp_path):
	token = new_share_token()
	write(tmp_path, token, 1)
	release = threading.Event()
	store = LocalRecordStore(tmp_path, reload=lambda: release.wait(5))
	store.LOCK_WAIT_S = 0.05
	stuck = threading.Thread(target=store.by_share_token, args=(token,))
	stuck.start()
	time.sleep(0.02)
	try:
		with pytest.raises(StoreBusy):
			store.by_share_token(token)
	finally:
		release.set()
		stuck.join()


def test_busy_store_is_a_503_the_browser_may_retry(tmp_path):
	class Busy:
		def by_share_token(self, token):
			raise StoreBusy("reloading")

		def healthy(self):
			return True

	with TestClient(create_app(records=Busy()), base_url="https://testserver") as c:
		res = c.get(f"/v1/records/{new_share_token()}")
		assert res.status_code == 503
		assert res.json()["error"]["code"] == "record_store_busy"
		assert res.json()["error"]["retryable"] is True
