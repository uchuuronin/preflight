"""Deploy the API on Modal: `modal deploy api/app.py` (the deploy workflow runs it).

Two pieces:
- `api`: the FastAPI app, up to 4 containers. Reached only through the web app's own origin
  (a Cloudflare Pages Function at /v1/*, web/functions/v1/[[path]].ts), which adds the edge
  secret; PREFLIGHT_REQUIRE_EDGE=1 refuses anything else.
- `SpendLedger`: one container that holds the spend ledger, so every API container reserves
  against the same caps atomically (preflight_api/ledger.py). It saves a snapshot to a Modal
  Dict after each change and restores it on start.

Secrets (never in the repo):
- "preflight-meta": META_API_KEY, META_API_BASE
- "preflight-api": PREFLIGHT_SESSION_KEYS (comma-separated, 32+ characters each),
  PREFLIGHT_EDGE_SECRET (the same value as the Pages project's PREFLIGHT_EDGE_SECRET)
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
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
		}
	)
	.add_local_file(ROOT / "flags.json", "/root/repo/flags.json")
	.add_local_dir(ROOT / "python" / "preflight", "/root/repo/python/preflight")
	.add_local_dir(ROOT / "api" / "preflight_api", "/root/repo/api/preflight_api")
	.add_local_dir(ROOT / "api" / "addons", "/root/repo/api/addons")
)
for rel in SAMPLE_FILES:
	image = image.add_local_file(ROOT / rel, f"/root/repo/{rel}")

app = modal.App("preflight-api", image=image)
ledger_store = modal.Dict.from_name("preflight-ledger", create_if_missing=True)
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


@app.function(secrets=SECRETS, min_containers=0, max_containers=4)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def api():
	_paths()
	from preflight_api.app import create_app

	return create_app()
