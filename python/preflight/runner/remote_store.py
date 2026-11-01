"""The job store as seen from a worker in another container: every call is an RPC to the one store.

Leases cross the wire as dicts. Errors keep their types (Modal re-raises the
remote exception), so LeaseLost still stops a worker and QueueFull still
carries retry_after_s.
"""

import copy
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .queue import LEASE_S, Lease
from .sources import content_hash


def _thin(lease: Lease) -> dict:
	"""A lease as the store needs it back: the request body stays behind (it can be megabytes)."""
	d = asdict(lease)
	d["request"] = {}
	return d


class Blobs(Protocol):
	"""Where uploaded files live while their job waits: the job state keeps only a reference."""

	def put(self, key: str, files: dict[str, str]) -> None: ...
	def get(self, key: str) -> dict[str, str]: ...


class DirBlobs:
	"""<root>/<key>.json; on Modal, a Volume (commit after write, reload before read)."""

	def __init__(self, root, *, commit=None, reload=None):
		self.root = Path(root)
		self.root.mkdir(parents=True, exist_ok=True)
		self._commit, self._reload = commit, reload

	def _path(self, key: str) -> Path:
		return self.root / (key.removeprefix("sha256:") + ".json")

	def put(self, key: str, files: dict[str, str]) -> None:
		p = self._path(key)
		if not p.exists():
			tmp = p.with_suffix(".tmp")
			tmp.write_text(json.dumps(files))
			tmp.replace(p)
		else:
			p.touch()  # uploaded again: its 30 days run from this job, the newest that uses it
		if self._commit:
			self._commit()

	def get(self, key: str) -> dict[str, str]:
		p = self._path(key)
		if not p.exists() and self._reload:
			self._reload()
		return json.loads(p.read_text())


def externalise_files(request: dict, blobs: Blobs | None) -> dict:
	"""The request with source.files moved to the blob store and replaced by files_ref."""
	src = request.get("source") if isinstance(request, dict) else None
	if blobs is None or not isinstance(src, dict) or not isinstance(src.get("files"), dict):
		return request
	files = src["files"]
	if not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
		return request  # malformed: leave it for fetch() to reject with a visitor-safe message
	key = content_hash(files)
	blobs.put(key, files)
	out = copy.deepcopy({**request, "source": {k: v for k, v in src.items() if k != "files"}})
	out["source"]["files_ref"] = key
	return out


class RemoteJobStore:
	"""call(method_name, *args, **kwargs) reaches JobStoreService; e.g. lambda m, *a, **k: getattr(svc, m).remote(*a, **k)."""

	def __init__(self, call: Callable, *, lease_s: float = LEASE_S):
		self._call = call
		self.lease_s = lease_s

	def lease(self, worker: str) -> Lease | None:
		d = self._call("lease", worker)
		return Lease(**d) if d else None

	def heartbeat(self, lease: Lease) -> float:
		return self._call("heartbeat", _thin(lease))

	def complete(self, lease: Lease, *, record_id: str, spent_usd: float | None = None) -> None:
		self._call("complete", _thin(lease), record_id=record_id, spent_usd=spent_usd)

	def fail(
		self,
		lease: Lease,
		*,
		error: str,
		retryable: bool,
		error_code: str | None = None,
		spent_usd: float | None = None,
	) -> None:
		self._call(
			"fail", _thin(lease), error=error, retryable=retryable, error_code=error_code, spent_usd=spent_usd
		)

	def emit(self, job_id: str, *, stage: str, token: int | None = None, **fields):
		return self._call("emit", job_id, stage, fields, token)


class StoreMethods:
	"""The service's methods over an InMemoryJobStore, with wire-friendly arguments and results.

	runner/modal_app.py exposes each of these as a Modal method; tests call them directly.
	"""

	def __init__(self, store, blobs: Blobs | None = None):
		self.store = store
		self.blobs = blobs

	# for C's API
	def submit(self, request: dict, visitor: str, idempotency_key: str) -> dict:
		request = externalise_files(request, self.blobs)  # the saved job state stays small
		s = self.store.submit(request, visitor=visitor, idempotency_key=idempotency_key)
		return {"job_id": s.job_id, "created": s.created, "position": s.position}

	def status(self, job_id: str) -> dict:
		return self.store.info(job_id)

	def position(self, job_id: str) -> int | None:
		return self.store.position(job_id)

	def events(self, job_id: str, after: int = -1) -> list[dict]:
		return [e.model_dump(mode="json", exclude_none=True) for e in self.store.events(job_id, after)]

	def cancel(self, job_id: str) -> None:
		self.store.cancel(job_id)

	# for workers
	def lease(self, worker: str) -> dict | None:
		lease = self.store.lease(worker)
		return asdict(lease) if lease else None

	def heartbeat(self, lease: dict) -> float:
		return self.store.heartbeat(Lease(**lease))

	def complete(self, lease: dict, record_id: str, spent_usd: float | None = None) -> None:
		self.store.complete(Lease(**lease), record_id=record_id, spent_usd=spent_usd)

	def fail(
		self,
		lease: dict,
		error: str,
		retryable: bool,
		error_code: str | None = None,
		spent_usd: float | None = None,
	) -> None:
		self.store.fail(
			Lease(**lease), error=error, retryable=retryable, error_code=error_code, spent_usd=spent_usd
		)

	def emit(self, job_id: str, stage: str, fields: dict, token: int | None = None) -> dict:
		ev = self.store.emit(job_id, stage=stage, token=token, **fields)
		return ev.model_dump(mode="json", exclude_none=True)

	def reap(self) -> list[str]:
		return self.store.reap()

	def expire(self, older_than_s: float) -> int:
		return len(self.store.expire(older_than_s))

	def demand(self) -> dict:
		"""Queued and running counts, and how long the oldest queued job has waited (the lateness
		signal to alarm on: a backlog shows there before anything else looks wrong)."""
		with self.store._lock:
			now = self.store._clock()
			ages = [
				now - self.store._jobs[j].submitted_at for j in self.store._order if j in self.store._jobs
			]
			return {
				"queued": len(self.store._order),
				"running": len(self.store._running()),
				"oldest_queued_s": round(max(ages), 1) if ages else 0.0,
			}
