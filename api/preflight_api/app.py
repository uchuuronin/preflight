"""App factory. Middleware: request id, the edge check (the API is reached through the web
app's own origin, edge.py) and the signed anonymous session cookie (session.py). Every router
in api/addons/ (owned by D) is mounted automatically, so add-on routes never touch this file.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import edge, errors, session
from .errors import envelope
from .guide.executors import Executor, job_executor
from .guide.muse_spark import TIMEOUT_S
from .idempotency import IdempotencyStore, MemoryIdempotencyStore
from .ledger import Ledger, MemoryLedger, ModalLedger
from .routes import guide as guide_routes
from .routes import health as health_routes
from .routes import records as records_routes
from .stores import LocalRecordStore, RecordStore, SampleStore

log = logging.getLogger("preflight.api")

ADDONS_DIR = Path(__file__).resolve().parents[1] / "addons"
REPO_ROOT = Path(os.environ.get("PREFLIGHT_ROOT", Path(__file__).resolve().parents[2]))


def default_ledger() -> Ledger:
	"""The shared single-container ledger in production; in-process for development and tests."""
	if os.environ.get("PREFLIGHT_LEDGER") == "modal":
		return ModalLedger()
	return MemoryLedger()


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
	jobs: Executor | None = None,
	samples: SampleStore | None = None,
	addons_dir: Path = ADDONS_DIR,
) -> FastAPI:
	session._keys()  # fail at start-up, not on the first visitor, if production has no key

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
	app.state.ledger = ledger or default_ledger()
	app.state.idempotency = idempotency or MemoryIdempotencyStore()
	app.state.http = http
	app.state.samples = samples or SampleStore(REPO_ROOT)
	app.state.jobs = jobs if jobs is not None else job_executor()

	errors.install(app)

	# Production serves the API from the web app's origin (same site), so there's no CORS at all.
	# Development may list the Vite dev server here.
	origins = [o for o in os.environ.get("PREFLIGHT_WEB_ORIGINS", "").split(",") if o]
	if origins:
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
		if not edge.allowed(request):
			return JSONResponse(
				envelope(request, "forbidden", "Use the Preflight app's own address.", False), status_code=403
			)
		raw = request.cookies.get(session.COOKIE)
		sid = session.verify(raw)
		fresh = sid is None
		resign = not fresh and session.needs_resign(raw)
		request.state.session = sid or session.new_session_id()
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
		if fresh or resign:
			response.set_cookie(
				session.COOKIE,
				session.sign(request.state.session),
				max_age=session.MAX_AGE_S,
				path="/",
				httponly=True,
				secure=True,  # required by the __Host- prefix; tests use an https base URL
				samesite="lax",
			)
			if request.cookies.get(session.LEGACY_COOKIE):
				response.delete_cookie(session.LEGACY_COOKIE)
		return response

	app.include_router(health_routes.router)
	app.include_router(records_routes.router)
	app.include_router(guide_routes.router)
	app.state.addons = mount_addons(app, addons_dir)
	return app
