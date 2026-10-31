"""Deploy the API on Modal: `modal deploy api/app.py` (the deploy workflow runs it).

Two pieces:
- `api`: the FastAPI app, up to 4 containers. Reached only through the web app's own origin
  (a Cloudflare Pages Function at /v1/*, web/functions/v1/[[path]].ts), which adds the edge
  secret; PREFLIGHT_REQUIRE_EDGE=1 refuses anything else.
- `SpendLedger`: one container that holds the spend ledger, so every API container reserves
  against the same caps atomically (preflight_api/ledger.py). It saves a snapshot to a Modal
  Dict after each change and restores it on start.

Records: A's workers write <share_token>.json to the Modal volume "preflight-records"; the API
mounts it read-only at /records and reloads it before reads (preflight_api/stores.py). A's tick
deletes records older than 30 days.

Before any deploy, start the image's code in Modal on CPU (no deploy, nothing left running):

	modal run --env main api/app.py::startup_check

Secrets (never in the repo):
- "preflight-meta": META_API_KEY, META_API_BASE
- "preflight-api": PREFLIGHT_SESSION_KEYS (comma-separated, 32+ characters each),
  PREFLIGHT_EDGE_SECRET (the same value as the Pages project's PREFLIGHT_EDGE_SECRET)
"""

from pathlib import Path

import modal

# Read only while building the image, on the machine that deploys. Inside a container Modal loads
# this file on its own as /root/app.py, where the repo isn't, so nothing at import may need it.
_HERE = Path(__file__).resolve()
ROOT = _HERE.parent.parent if (_HERE.parent.parent / "flags.json").is_file() else Path("/repo-not-here")
RECORDS_DIR = "/records"
SAMPLE_FILES = [
	"web/src/workspace/fixtures/dev-finetune.record.json",
	"schema/examples/finetune.json",
	"schema/examples/pretrain.json",
	"schema/examples/agent.json",
]

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install("fastapi>=0.115", "pydantic>=2.8,<3", "httpx>=0.27", "numpy>=1.26")
	.env(
		{
			"PREFLIGHT_ROOT": "/root/repo",
			"PREFLIGHT_ENV": "production",
			"PREFLIGHT_REQUIRE_EDGE": "1",
			"PREFLIGHT_LEDGER": "modal",
			"PREFLIGHT_RECORDS_DIR": RECORDS_DIR,
			# A's job store (python/preflight/runner/modal_app.py), for /v1/jobs and approved guide jobs
			"PREFLIGHT_JOBSTORE_APP": "preflight-runner",
			"PREFLIGHT_JOBSTORE_CLS": "JobStoreService",
		}
	)
	.add_local_file(ROOT / "flags.json", "/root/repo/flags.json")
	# PREFLIGHT_ENV=production reads this over flags.json (preflight_api/flags.py). B's file; the
	# runner's image ships the same two (A's test holds both images to one list).
	.add_local_file(ROOT / "flags.production.json", "/root/repo/flags.production.json")
	.add_local_dir(ROOT / "python" / "preflight", "/root/repo/python/preflight")
	.add_local_dir(ROOT / "api" / "preflight_api", "/root/repo/api/preflight_api")
	.add_local_dir(ROOT / "api" / "addons", "/root/repo/api/addons")
)
for rel in SAMPLE_FILES:
	image = image.add_local_file(ROOT / rel, f"/root/repo/{rel}")

app = modal.App("preflight-api", image=image)
ledger_store = modal.Dict.from_name("preflight-ledger", create_if_missing=True)
# A's volume (python/preflight/runner/modal_app.py creates it); never created here, so a missing
# volume fails the startup check instead of serving an empty one.
records_vol = modal.Volume.from_name("preflight-records")


def _read_only(vol):
	try:
		return vol.with_mount_options(read_only=True)
	except AttributeError:  # older SDKs
		return vol.read_only()


SECRETS = [modal.Secret.from_name("preflight-meta"), modal.Secret.from_name("preflight-api")]


def _paths() -> None:
	import sys

	sys.path[:0] = ["/root/repo/api", "/root/repo/python"]


@app.cls(max_containers=1, min_containers=1, scaledown_window=600)
@modal.concurrent(max_inputs=200)
class SpendLedger:
	"""The one place spend is reserved. Atomic because there is exactly one of it."""

	@modal.enter()
	def load(self) -> None:
		_paths()
		from preflight_api.ledger import LedgerCore

		snap = ledger_store.get("snapshot")
		self.core = LedgerCore.restore(snap) if snap else LedgerCore()

	def _save(self) -> None:
		ledger_store["snapshot"] = self.core.snapshot()

	@modal.method()
	def reserve(self, *, visitor: str, run: str, ip: str, usd: float) -> dict:
		from dataclasses import asdict

		decision = self.core.reserve(visitor=visitor, run=run, ip=ip, usd=usd)
		if decision.ok:
			self._save()
		return asdict(decision)

	@modal.method()
	def settle(self, reservation: str, usd: float) -> None:
		self.core.settle(reservation, usd)
		self._save()

	@modal.method()
	def release(self, reservation: str) -> None:
		self.core.release(reservation)
		self._save()

	@modal.method()
	def spent_today(self) -> float:
		return self.core.spent_today()


def _records():
	from preflight_api.stores import LocalRecordStore

	return LocalRecordStore(Path(RECORDS_DIR), reload=records_vol.reload)


@app.function(
	secrets=SECRETS, min_containers=0, max_containers=4, volumes={RECORDS_DIR: _read_only(records_vol)}
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def api():
	_paths()
	from preflight_api.app import create_app

	return create_app(records=_records())


# ---------------------------------------------------------------- startup check (CPU, no deploy)


@app.function(secrets=SECRETS, volumes={RECORDS_DIR: _read_only(records_vol)}, timeout=180, retries=0)
def startup_check_api() -> dict:
	"""What an API container does when it starts, in the real image with the real secrets and the
	records volume: build the app as `api()` does, then answer the health routes, an edge-signed
	record miss (session keys, edge secret, volume reload) and a job-store lookup. Touches no
	shared state: the ledger is checked by a snapshot round trip, not by calling SpendLedger."""
	import os
	import time

	_paths()
	from fastapi.testclient import TestClient

	from preflight_api import edge
	from preflight_api.app import create_app
	from preflight_api.jobs import jobs_client, kind_of
	from preflight_api.ledger import LedgerCore, MemoryLedger

	out: dict = {"module": __name__, "file": __file__}
	t0 = time.monotonic()
	records_vol.reload()
	out["reload_s"] = round(time.monotonic() - t0, 3)
	out["records_seen"] = sum(1 for _ in Path(RECORDS_DIR).glob("*.json"))
	core = LedgerCore.restore(LedgerCore().snapshot())
	out["ledger_restores"] = core.spent_today() == 0.0
	app_ = create_app(records=_records(), ledger=MemoryLedger())
	with TestClient(app_, base_url="https://testserver") as c:
		out["livez"] = c.get("/livez").status_code
		out["readyz"] = c.get("/readyz").status_code
		out["unsigned_status"] = c.get("/v1/status").status_code  # refused without the edge
		miss = c.get(
			"/v1/records/" + "x" * 24, headers={edge.EDGE_HEADER: os.environ.get("PREFLIGHT_EDGE_SECRET", "")}
		)
		out["signed_miss"] = [miss.status_code, (miss.json().get("error") or {}).get("code")]
	store = jobs_client()
	out["jobstore"] = None
	if store is not None:
		import asyncio

		try:
			asyncio.run(store.status("check-" + "0" * 26))
			out["jobstore"] = "answered"
		except Exception as e:  # UnknownJob is the healthy answer
			out["jobstore"] = kind_of(e)
	return out


@app.local_entrypoint()
def startup_check() -> None:
	"""Exits non-zero unless the API starts and answers as it would deployed. Writes
	out/startup_check_api.json."""
	import json
	import sys

	try:
		res = startup_check_api.remote()
	except Exception as exc:  # the remote traceback is in the log above
		res = {"failed": f"{type(exc).__name__}: {exc}"[:2000]}
	res["ok"] = bool(
		"failed" not in res
		and res.get("livez") == 200
		and res.get("readyz") == 200
		and res.get("unsigned_status") == 403
		and res.get("signed_miss") == [404, "record_not_found"]
		and res.get("ledger_restores")
		and res.get("jobstore") == "UnknownJob"
	)
	Path("out").mkdir(exist_ok=True)
	Path("out/startup_check_api.json").write_text(json.dumps(res, indent=2))
	print(json.dumps(res, indent=2), flush=True)
	if not res["ok"]:
		sys.exit(1)
