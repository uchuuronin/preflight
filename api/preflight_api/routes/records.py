from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..errors import ApiError

router = APIRouter(prefix="/v1")


@router.get("/records/{share_token}")
async def get_record(share_token: str, request: Request) -> JSONResponse:
	"""A record by its share token. The ULID never appears in a public link."""
	# Off the event loop: the store may wait for a volume reload.
	record = await asyncio.to_thread(request.app.state.records.by_share_token, share_token)
	if record is None:
		raise ApiError(
			404, "record_not_found", "No flight record matches this link. Check the link or open a sample."
		)
	# A running job revises its record (skeleton, then each probe), so nothing may keep a copy.
	return JSONResponse(record, headers={"Cache-Control": "private, no-cache"})
