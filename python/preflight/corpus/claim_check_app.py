"""The planted-claims evaluation with Spark Open picking among candidate fields (corpus/claim_check_eval.py).

	modal run --env main python/preflight/corpus/claim_check_app.py::evaluate

Spark Open on Baseten (the preflight-baseten secret), one call per claim with more than one candidate field. Stops
before the next call could pass the cap. Writes out/claim_check_eval.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

MAX_USD = 1.0  # cited: PM, Nov 2, item 3 (Baseten credit, cap $1)
WORST_CALL_USD = (
	0.01  # guess: one short pick call (a sentence and a field list) at Spark Open's list price, with room
)

ROOT = Path(__file__).resolve().parents[3]
app = modal.App("preflight-claim-check-eval")
image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install("pydantic>=2.8,<3", "numpy>=1.26", "httpx>=0.27")
	.env({"PREFLIGHT_CLAIMS_ROOT": "/root/claims"})
)
from preflight.corpus.claim_check_eval import RECORDS  # noqa: E402

for _rel in RECORDS:
	image = image.add_local_file(str(ROOT / _rel), f"/root/claims/{_rel}")
image = image.add_local_python_source("preflight")


class _Capped:
	"""Spark Open, refusing a call once the next could pass the cap."""

	def __init__(self, client, max_usd: float):
		self.client, self.max_usd, self.skipped = client, max_usd, 0

	@property
	def spent(self) -> float:
		return sum(c.cost_usd for c in self.client.calls)

	def structured(self, *a, **kw):
		if self.spent + WORST_CALL_USD > self.max_usd:
			self.skipped += 1
			raise RuntimeError("the cap")
		return self.client.structured(*a, **kw)


@app.function(image=image, secrets=[modal.Secret.from_name("preflight-baseten")], timeout=30 * 60)
def evaluate_remote(max_usd: float = MAX_USD) -> dict:
	from preflight.corpus import claim_check_eval as E
	from preflight.harness.spark import ModelClient

	client = _Capped(ModelClient.baseten_spark_open(), max_usd)
	items = E.run(E.claims(), client=client)
	calls = client.client.calls
	return {
		"model": client.client.model,
		"usd": round(client.spent, 6),
		"calls": len(calls),
		"failed_calls": sum(not c.ok for c in calls),
		"skipped_at_cap": client.skipped,
		"summary": E.summary(items),
		"items": items,
	}


@app.local_entrypoint()
def evaluate(max_usd: float = MAX_USD):
	t0 = time.monotonic()
	got = evaluate_remote.remote(max_usd)
	got["wall_s"] = round(time.monotonic() - t0, 1)
	Path("out").mkdir(exist_ok=True)
	Path("out/claim_check_eval.json").write_text(json.dumps(got, indent=1))
	print(json.dumps({k: v for k, v in got.items() if k != "items"}, indent=1))
