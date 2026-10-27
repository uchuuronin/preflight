"""Write-once result store on a directory (a Modal Volume mount, or a local disk).

Layout: <root>/<kind>/<gpu>/<key[:2]>/<key>.json. Each result is written once
through a temp file and an atomic rename, so a preempted sweep leaves no partial
files and resumes by skipping keys that already exist (gap 13).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
META_FIELDS = ("status", "error_code", "gpu_seconds", "prior_gpu_seconds")


class ResultStore:
	def __init__(self, root: str | os.PathLike[str]):
		self.root = Path(root)

	def path(self, kind: str, gpu: str, key: str) -> Path:
		return self.root / kind / gpu / key[:2] / f"{key}.json"

	def meta_path(self, kind: str, gpu: str, key: str) -> Path:
		"""A few-field summary beside each record, so ledgers don't parse op tables and graphs."""
		return self.root / kind / gpu / key[:2] / f"{key}.meta"

	@staticmethod
	def _load(p: Path) -> dict[str, Any] | None:
		try:
			return json.loads(p.read_text())
		except (OSError, json.JSONDecodeError) as exc:
			log.warning("skipping unreadable corpus file %s: %s", p, exc)
			return None

	def read(self, kind: str, gpu: str, key: str) -> dict[str, Any] | None:
		p = self.path(kind, gpu, key)
		if not p.exists():
			return None
		return self._load(p)

	def write_once(
		self, kind: str, gpu: str, key: str, record: dict[str, Any], replace_errors: bool = False
	) -> bool:
		p = self.path(kind, gpu, key)
		if p.exists():
			existing = self.read(kind, gpu, key) or {"status": "error"}
			if not (replace_errors and existing.get("status") in ("error", "timeout")):
				return False
			# spend on the failed attempt stays counted, so retries can't hide money
			prior = float(existing.get("gpu_seconds", 0.0)) + float(existing.get("prior_gpu_seconds", 0.0))
			record = {**record, "prior_gpu_seconds": prior}
		p.parent.mkdir(parents=True, exist_ok=True)
		self._atomic_write(p, record)
		self._atomic_write(self.meta_path(kind, gpu, key), {k: record.get(k) for k in META_FIELDS})
		return True

	@staticmethod
	def _atomic_write(p: Path, data: dict[str, Any]) -> None:
		tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
		tmp.write_text(json.dumps(data, sort_keys=True))
		os.replace(tmp, p)

	def iter_records(
		self, kind: str | None = None, gpu: str | None = None
	) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
		kinds = (
			[kind]
			if kind
			else [d.name for d in self.root.iterdir() if d.is_dir()]
			if self.root.exists()
			else []
		)
		for k in kinds:
			kdir = self.root / k
			if not kdir.exists():
				continue
			gdirs = [kdir / gpu] if gpu else [d for d in kdir.iterdir() if d.is_dir()]
			for gdir in gdirs:
				for f in gdir.glob("*/*.json"):
					rec = self._load(f)
					if rec is not None:
						yield k, gdir.name, f.stem, rec

	def iter_meta(self, kind: str, gpu: str | None = None) -> Iterator[tuple[str, str, dict[str, Any]]]:
		"""(gpu, key, summary) per config, from the .meta files (falling back to the record)."""
		kdir = self.root / kind
		if not kdir.exists():
			return
		gdirs = [kdir / gpu] if gpu else [d for d in kdir.iterdir() if d.is_dir()]
		for gdir in gdirs:
			for rec_file in gdir.glob("*/*.json"):
				key = rec_file.stem
				meta = (
					self._load(rec_file.with_suffix(".meta"))
					if rec_file.with_suffix(".meta").exists()
					else None
				)
				if meta is None:
					rec = self._load(rec_file)
					if rec is None:
						continue
					meta = {k: rec.get(k) for k in META_FIELDS}
				yield gdir.name, key, meta

	def done_keys(self, kind: str, gpu: str, retry_errors: bool = False) -> set[str]:
		"""Keys not to run again. Errors and timeouts count as done unless retried."""
		return {
			key
			for _, key, meta in self.iter_meta(kind, gpu)
			if not (retry_errors and meta.get("status") in ("error", "timeout"))
		}

	def gpu_seconds(self, kind: str, gpu: str) -> list[float]:
		"""Billed seconds of each config's latest attempt on this GPU, for the running cost estimate."""
		return [float(meta.get("gpu_seconds") or 0.0) for _, _, meta in self.iter_meta(kind, gpu)]

	def spent_usd(
		self, usd_per_hour: dict[str, float], kind: str | None = None, gpu: str | None = None
	) -> float:
		"""Dollars spent, failed attempts included (prior_gpu_seconds)."""
		total = 0.0
		kinds = (
			[kind]
			if kind
			else ([d.name for d in self.root.iterdir() if d.is_dir()] if self.root.exists() else [])
		)
		for k in kinds:
			for g, _, meta in self.iter_meta(k, gpu):
				secs = float(meta.get("gpu_seconds") or 0.0) + float(meta.get("prior_gpu_seconds") or 0.0)
				total += secs / 3600 * usd_per_hour[g]
		return total
