"""Job events: what GET /v1/jobs/{id}/events streams (server-sent events).

A's runner emits them; C's API relays them. One event per SSE message, with
`seq` as the SSE id so a reconnecting client resumes with Last-Event-ID.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from .base import Strict
from .record import Finding

Stage = Literal[
	"queued",
	"admitted",
	"fetching",
	"mapping",
	"skeleton",
	"harnessing",
	"probing",
	"triage",
	"predicting",
	"done",
	"failed",
	"cancelled",
]


class JobEvent(Strict):
	job_id: str
	record_id: str | None = None
	seq: int = Field(ge=0)
	at: datetime
	stage: Stage
	progress: float | None = Field(default=None, ge=0, le=1, description="Within the current stage")
	queue_position: int | None = Field(default=None, ge=0)
	message: str | None = Field(default=None, description="i18n key or plain log line")
	level: Literal["debug", "info", "warn", "error"] = "info"
	probe_id: str | None = None
	finding: Finding | None = None
	record_rev: int | None = Field(
		default=None, description="Bumped when the stored record changed; refetch it"
	)
	retryable: bool | None = Field(default=None, description="On failed: whether resubmitting may succeed")


class JobRequest(Strict):
	"""Body of POST /v1/jobs (validated by C, executed by A)."""

	source: dict = Field(description="A Source: sample, repo, upload, notebook or template")
	lens: str | None = Field(default=None, description="Optional hint; the script map decides")
	objective: dict | None = None
	constraints: dict | None = None
	gpu: str | None = Field(default=None, description="Requested GPU class; admission control may change it")
	description: str | None = Field(default=None, description="The visitor's use case, for the guide")
