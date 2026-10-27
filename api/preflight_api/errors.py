"""One error envelope for every failure: {"error": {code, message, request_id, retryable}}.

Messages say what happened and what to do next, in plain words; they never carry stack
traces, secrets or internal hostnames.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
	def __init__(
		self,
		status: int,
		code: str,
		message: str,
		*,
		retryable: bool = False,
		retry_after: int | None = None,
		extra: dict | None = None,
	) -> None:
		super().__init__(message)
		self.status = status
		self.code = code
		self.message = message
		self.retryable = retryable
		self.retry_after = retry_after
		# Machine-readable details the client acts on, e.g. {"fallback": "rule_based"}.
		self.extra = extra or {}


def envelope(request: Request, code: str, message: str, retryable: bool, extra: dict | None = None) -> dict:
	return {
		"error": {
			**(extra or {}),
			"code": code,
			"message": message,
			"request_id": getattr(request.state, "request_id", None),
			"retryable": retryable,
		}
	}


def quota_exceeded(message: str, retry_after: int) -> ApiError:
	return ApiError(429, "quota_exceeded", message, retryable=True, retry_after=retry_after)


_HTTP_CODES = {
	400: "bad_request",
	401: "unauthorised",
	403: "forbidden",
	404: "not_found",
	405: "method_not_allowed",
	409: "conflict",
	413: "too_large",
	422: "invalid_request",
	429: "quota_exceeded",
}


def install(app: FastAPI) -> None:
	@app.exception_handler(ApiError)
	async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
		headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
		return JSONResponse(
			envelope(request, exc.code, exc.message, exc.retryable, exc.extra), status_code=exc.status, headers=headers
		)

	@app.exception_handler(RequestValidationError)
	async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
		first = exc.errors()[0] if exc.errors() else {}
		where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
		msg = f"{where}: {first.get('msg', 'invalid value')}" if where else "The request body is not valid."
		return JSONResponse(envelope(request, "invalid_request", msg, False), status_code=422)

	@app.exception_handler(StarletteHTTPException)
	async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
		code = _HTTP_CODES.get(exc.status_code, "error")
		message = exc.detail if isinstance(exc.detail, str) else "Request failed."
		return JSONResponse(envelope(request, code, message, exc.status_code >= 500), status_code=exc.status_code)

	@app.exception_handler(Exception)
	async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
		# Logged with the request id by the middleware; the visitor gets no internals.
		return JSONResponse(
			envelope(request, "internal", "Something went wrong on our side. Try again in a minute.", True),
			status_code=500,
		)
