import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from preflight_api import session
from preflight_api.app import create_app
from preflight_api.ledger import LedgerCore, Limits, MemoryLedger


@pytest.fixture(autouse=True)
def keys(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", "k" * 40)
	session.reset_keys()
	yield
	session.reset_keys()


def client(**kw):
	return TestClient(create_app(**kw), base_url="https://testserver")


# ---- signed sessions


def test_forged_or_unsigned_cookies_get_a_fresh_session():
	sid = session.new_session_id()
	good = session.sign(sid)
	assert session.verify(good) == sid
	assert session.verify(f"v1.{sid}.AAAA") is None
	assert session.verify(sid) is None
	other = session.new_session_id()
	assert session.verify(good.replace(sid, other)) is None
	with client() as c:
		c.cookies.set(session.COOKIE, f"v1.{sid}.forged")
		res = c.get("/livez")
		assert "__Host-pf_session=v1." in res.headers["set-cookie"]
		assert sid not in res.headers["set-cookie"]


def test_key_rotation_keeps_sessions_and_reissues_them(monkeypatch):
	sid = session.new_session_id()
	old = session.sign(sid)
	monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", ",".join(["n" * 40, "k" * 40]))
	session.reset_keys()
	assert session.verify(old) == sid
	assert session.needs_resign(old)
	with client() as c:
		c.cookies.set(session.COOKIE, old)
		res = c.get("/livez")
		assert res.headers["set-cookie"].startswith(f"__Host-pf_session=v1.{sid}.")


def test_production_refuses_to_start_without_a_session_key(monkeypatch):
	monkeypatch.delenv("PREFLIGHT_SESSION_KEYS")
	monkeypatch.setenv("PREFLIGHT_ENV", "production")
	session.reset_keys()
	with pytest.raises(session.MissingSessionKey):
		create_app()


# ---- same site: only the edge may call the API in production


def test_edge_secret_required_in_production(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_REQUIRE_EDGE", "1")
	monkeypatch.setenv("PREFLIGHT_EDGE_SECRET", "s" * 40)
	with client() as c:
		assert c.get("/livez").status_code == 200  # health checks stay open
		assert c.get("/v1/status").status_code == 403
		assert c.get("/v1/status", headers={"X-Preflight-Edge": "wrong"}).status_code == 403
		assert c.get("/v1/status", headers={"X-Preflight-Edge": "s" * 40}).status_code == 200


def test_no_cross_site_cors_unless_configured():
	with client() as c:
		res = c.options(
			"/v1/status",
			headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
		)
		assert "access-control-allow-origin" not in {k.lower() for k in res.headers}


# ---- the ledger: reserve atomically, settle, expire


def test_concurrent_reservations_never_overshoot_the_daily_cap():
	core = LedgerCore(
		Limits(
			usd_per_day=1.0,
			usd_per_run=100,
			turns_per_visitor_day=10_000,
			turns_per_ip_day=10_000,
			ip_burst=10_000,
		)
	)
	granted = []

	def go(i):
		d = core.reserve(visitor=f"v{i}", run=f"r{i}", ip=f"10.0.0.{i % 250}", usd=0.07)
		if d.ok:
			granted.append(d.reservation)

	threads = [threading.Thread(target=go, args=(i,)) for i in range(200)]
	for t in threads:
		t.start()
	for t in threads:
		t.join()
	assert len(granted) == 14  # floor(1.0 / 0.07)
	assert core.spent_today() <= 1.0


def test_settle_replaces_the_worst_case_and_frees_the_rest():
	core = LedgerCore(Limits(usd_per_day=1.0))
	a = core.reserve(visitor="v", run="r", ip="1.1.1.1", usd=0.30)
	assert a.ok and core.spent_today() == pytest.approx(0.30)
	core.settle(a.reservation, 0.01)
	assert core.spent_today() == pytest.approx(0.01)
	core.settle(a.reservation, 5.0)  # idempotent: a second settle changes nothing
	assert core.spent_today() == pytest.approx(0.01)


def test_run_cap_is_per_visitor_and_record():
	core = LedgerCore(Limits(usd_per_run=0.10, usd_per_day=10))
	assert core.reserve(visitor="v", run="rec:v", ip="1.1.1.1", usd=0.08).ok
	d = core.reserve(visitor="v", run="rec:v", ip="1.1.1.1", usd=0.08)
	assert not d.ok and d.reason == "run_cap"
	assert core.reserve(visitor="w", run="rec:w", ip="2.2.2.2", usd=0.08).ok


def test_lost_reservations_expire_at_their_worst_case():
	core = LedgerCore(Limits(usd_per_day=1.0, reservation_ttl_s=10))
	core.reserve(visitor="v", run="r", ip="1.1.1.1", usd=0.4, now=1000.0)
	assert core.spent_today(now=1020.0) == pytest.approx(0.4)
	assert core.snapshot()["open"] == []


def test_ip_bucket_limits_bursts_and_refills():
	core = LedgerCore(Limits(ip_burst=3, ip_refill_per_s=1.0, turns_per_visitor_day=100))
	for i in range(3):
		assert core.reserve(visitor=f"v{i}", run=f"r{i}", ip="9.9.9.9", usd=0.0, now=100.0).ok
	d = core.reserve(visitor="v9", run="r9", ip="9.9.9.9", usd=0.0, now=100.0)
	assert not d.ok and d.reason == "ip_burst" and d.retry_after_s == pytest.approx(1.0)
	assert core.reserve(visitor="v9", run="r9", ip="9.9.9.9", usd=0.0, now=101.5).ok


def test_release_refunds_money_and_turn():
	core = LedgerCore(Limits(turns_per_visitor_day=1))
	d = core.reserve(visitor="v", run="r", ip="1.1.1.1", usd=0.2)
	core.release(d.reservation)
	assert core.spent_today() == 0
	assert core.reserve(visitor="v", run="r", ip="1.1.1.1", usd=0.2).ok


def test_snapshot_round_trips():
	core = LedgerCore()
	d = core.reserve(visitor="v", run="r", ip="1.1.1.1", usd=0.05)
	core.settle(d.reservation, 0.02)
	core.reserve(visitor="v", run="r", ip="1.1.1.1", usd=0.03)
	again = LedgerCore.restore(core.snapshot())
	assert again.spent_today() == pytest.approx(0.05)


def test_memory_ledger_is_async():
	async def go():
		led = MemoryLedger()
		d = await led.reserve(visitor="v", run="r", ip="1.1.1.1", usd=0.01)
		await led.settle(d.reservation, 0.005)
		return await led.spent_today()

	assert asyncio.run(go()) == pytest.approx(0.005)
