from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..flags import flag
from ..guide import muse_spark
from ..guide.budget import LLM_USD_PER_DAY

router = APIRouter()


@router.get("/livez")
async def livez() -> dict:
	return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
	store_ok = request.app.state.records.healthy()
	body = {"status": "ok" if store_ok else "degraded", "checks": {"record_store": store_ok}}
	return JSONResponse(body, status_code=200 if store_ok else 503)


@router.get("/v1/status")
async def status(request: Request) -> dict:
	"""Public service status: what works right now, in words the status page shows."""
	guide_live = flag("c.guide.live") and muse_spark.configured()
	spent = request.app.state.ledger.spent_today()
	return {
		"status": "ok" if request.app.state.records.healthy() else "degraded",
		"guide": "muse_spark" if guide_live else "rule_based",
		"llm_spend_today_usd": round(spent, 2),
		"llm_spend_cap_usd": LLM_USD_PER_DAY,
		"samples_only": spent >= LLM_USD_PER_DAY,
	}
