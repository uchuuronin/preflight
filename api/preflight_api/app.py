"""App factory. Middleware: request id, anonymous session cookie. Every router in
api/addons/ (owned by D) is mounted automatically, so add-on routes never touch this file.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import errors
from .errors import envelope
from .guide.budget import Ledger, MemoryLedger
from .guide.muse_spark import TIMEOUT_S
from .idempotency import IdempotencyStore, MemoryIdempotencyStore
from .routes import guide as guide_routes
from .routes import health as health_routes
from .routes import records as records_routes
from .stores import LocalRecordStore, RecordStore

log = logging.getLogger("preflight.api")

ADDONS_DIR = Path(__file__).resolve().parents[1] / "addons"
SESSION_COOKIE = "pf_session"


def mount_addons(app: FastAPI, directory: Path = ADDONS_DIR) -> list[str]:
	"""Import api/addons/<name>.py and api/addons/<name>/__init__.py; mount each `router`."""
	mounted: list[str] = []
	if not directory.is_dir():
		return mounted
	for path in sorted(directory.iterdir()):
		if path.name.startswith(("_", ".")):
			continue
		target = path / "__init__.py" if path.is_dir() else path
		if target.suffix != ".py" or not target.is_file():
			continue
		spec = importlib.util.spec_from_file_location(f"preflight_addons.{path.stem}", target)
		if spec is None or spec.loader is None:
			continue
		module = importlib.util.module_from_spec(spec)
		try:
			spec.loader.exec_module(module)
		except Exception:
			# One broken add-on must not take the API down.
			log.exception("addon %s failed to load", path.name)
			continue
		router = getattr(module, "router", None)
		if isinstance(router, APIRouter):
			app.include_router(router)
			mounted.append(path.stem)
	return mounted


def create_app(
	*,
	records: RecordStore | None = None,
	ledger: Ledger | None = None,
	idempotency: IdempotencyStore | None = None,
	http: httpx.AsyncClient | None = None,
	addons_dir: Path = ADDONS_DIR,
) -> FastAPI:
	@asynccontextmanager
	async def lifespan(app: FastAPI):
		own = http is None
		app.state.http = http or httpx.AsyncClient(timeout=TIMEOUT_S)
		yield
		if own:
			await app.state.http.aclose()

	app = FastAPI(
		title="Preflight API",
		version="0.1.0",
		lifespan=lifespan,
		docs_url="/v1/docs",
		openapi_url="/v1/openapi.json",
	)
	app.state.records = records or LocalRecordStore(Path(os.environ.get("PREFLIGHT_RECORDS_DIR", "records")))
	app.state.ledger = ledger or MemoryLedger()
	app.state.idempotency = idempotency or MemoryIdempotencyStore()
	app.state.http = http

	errors.install(app)

	origins = [
		o
		for o in os.environ.get("PREFLIGHT_WEB_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(
			","
		)
		if o
	]
	app.add_middleware(
		CORSMiddleware,
		allow_origins=origins,
		allow_credentials=True,
		allow_methods=["GET", "POST"],
		allow_headers=["Content-Type", "Idempotency-Key"],
		expose_headers=["X-Request-Id", "Retry-After"],
	)

	@app.middleware("http")
	async def request_context(request: Request, call_next):
		request.state.request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
		session = request.cookies.get(SESSION_COOKIE)
		fresh = not session or len(session) > 64
		request.state.session = secrets.token_urlsafe(18) if fresh else session
		try:
			response = await call_next(request)
		except Exception:
			log.exception("unhandled error", extra={"request_id": request.state.request_id})
			response = JSONResponse(
				envelope(
					request, "internal", "Something went wrong on our side. Try again in a minute.", True
				),
				status_code=500,
			)
		response.headers["X-Request-Id"] = request.state.request_id
		if fresh:
			response.set_cookie(
				SESSION_COOKIE,
				request.state.session,
				max_age=30 * 86400,
				httponly=True,
				secure=os.environ.get("PREFLIGHT_INSECURE_COOKIES") != "1",
				samesite="lax",
			)
		return response

	app.include_router(health_routes.router)
	app.include_router(records_routes.router)
	app.include_router(guide_routes.router)
	app.state.addons = mount_addons(app, addons_dir)
	return app
