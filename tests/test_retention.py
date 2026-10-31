"""Everything about a job expires with its record at 30 days (Chiya, Oct 30)."""

import os
import time

from preflight.runner.queue import InMemoryJobStore
from preflight.runner.remote_store import DirBlobs, StoreMethods
from preflight.runner.retention import TTL_S, cache_entry, cached_probe, prune_cache, prune_uploads

DAY = 86400
NOW = 1_800_000_000.0


def _age(p, days):
	t = NOW - days * DAY
	os.utime(p, (t, t))


def test_uploads_expire_30_days_after_their_newest_use(tmp_path):
	blobs = DirBlobs(tmp_path)
	blobs.put("sha256:old", {"train.py": "x"})
	blobs.put("sha256:reused", {"train.py": "y"})
	blobs.put("sha256:new", {"train.py": "z"})
	_age(tmp_path / "old.json", 31)
	_age(tmp_path / "reused.json", 31)
	_age(tmp_path / "new.json", 2)
	blobs.put("sha256:reused", {"train.py": "y"})  # checked in again today
	os.utime(tmp_path / "reused.json", (NOW, NOW))  # what put's touch does, at NOW
	(tmp_path / "half.tmp").write_text("{")
	_age(tmp_path / "half.tmp", 2)
	(tmp_path / "future.json").write_text("{}")
	os.utime(tmp_path / "future.json", (NOW + 5 * DAY, NOW + 5 * DAY))  # a clock gone wrong keeps it
	assert sorted(prune_uploads(tmp_path, now=NOW)) == ["half.tmp", "old.json"]
	assert sorted(p.name for p in tmp_path.iterdir()) == ["future.json", "new.json", "reused.json"]


def test_reusing_an_upload_renews_it(tmp_path):
	blobs = DirBlobs(tmp_path)
	blobs.put("sha256:a", {"f": "1"})
	old = time.time() - 40 * DAY
	os.utime(tmp_path / "a.json", (old, old))
	blobs.put("sha256:a", {"f": "1"})
	assert time.time() - (tmp_path / "a.json").stat().st_mtime < 60


def test_cached_probes_expire_and_untimed_entries_go(tmp_path):
	d = {
		"fresh": cache_entry({"id": "p1"}, now=NOW - 3 * DAY),
		"stale": cache_entry({"id": "p2", "error": "Traceback ... visitor text"}, now=NOW - 31 * DAY),
		"legacy": {"id": "p3"},  # from before the rule: no time
		"future": cache_entry({"id": "p4"}, now=NOW + DAY),
	}
	removed = prune_cache(list(d.items()), d.pop, now=NOW)
	assert sorted(removed) == ["legacy", "stale"] and sorted(d) == ["fresh", "future"]
	assert cached_probe(d["fresh"]) == {"id": "p1"} and cached_probe({"id": "p3"}) is None


def test_the_job_index_forgets_finished_jobs_after_30_days_and_never_live_ones():
	clock = [NOW - 40 * DAY]
	store = InMemoryJobStore(clock=lambda: clock[0])
	m = StoreMethods(store)
	old = store.submit(
		{"source": {"kind": "sample", "uri": "sample://zoo/x"}}, visitor="v1", idempotency_key="k1"
	)
	lease = store.lease("w")
	store.complete(lease, record_id="r1")
	clock[0] = NOW - 35 * DAY
	live = store.submit(
		{"source": {"kind": "sample", "uri": "sample://zoo/y"}}, visitor="v2", idempotency_key="k2"
	)
	clock[0] = NOW - DAY
	recent = store.submit(
		{"source": {"kind": "sample", "uri": "sample://zoo/z"}}, visitor="v3", idempotency_key="k3"
	)
	store.lease("w")  # v2's job: leased (live) but old
	l3 = store.lease("w2")
	store.complete(l3, record_id="r3")
	clock[0] = NOW
	assert m.expire(TTL_S) == 1
	assert old.job_id not in store._index and ("v1", "k1") not in store._keys
	assert recent.job_id in store._index and live.job_id in store._jobs  # recent and live jobs stay
	assert m.expire(TTL_S) == 0
	# the same key can be used again once its job is forgotten
	again = store.submit(
		{"source": {"kind": "sample", "uri": "sample://zoo/x"}}, visitor="v1", idempotency_key="k1"
	)
	assert again.created and again.job_id != old.job_id


def test_the_daily_cleanup_covers_every_kind_of_job_data():
	from pathlib import Path

	src = (
		Path(__file__).resolve().parents[1] / "python" / "preflight" / "runner" / "modal_app.py"
	).read_text()
	body = src[
		src.index("def _prune_records_daily") : src.index(
			"@app.function(", src.index("def _prune_records_daily")
		)
	]
	for part in ("prune(RECORDS_DIR)", "prune_uploads(UPLOADS_DIR)", "prune_cache(", ".expire.remote(TTL_S)"):
		assert part in body, part
	assert "UPLOADS_DIR: uploads_vol" in src[src.index("schedule=modal.Period(minutes=1)") - 200 :][:400]
