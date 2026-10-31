"""Job events: what GET /v1/jobs/{id}/events streams (server-sent events).

A's runner emits them; C's API relays them. One event per SSE message, with
`seq` as the SSE id so a reconnecting client resumes with Last-Event-ID.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .base import KnobValue, Strict
from .record import Constraints, ErrorCode, Finding

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


JobMode = Literal["plan", "probe", "verify", "sample_more"]


class JobEvent(Strict):
	job_id: str
	record_id: str | None = None
	share_token: str | None = Field(
		default=None,
		description="Where the record is read (GET /v1/records/{share_token}) once a skeleton exists",
	)
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
	error_code: ErrorCode | None = Field(
		default=None,
		description="On failed: a stable code in Probe.error_code's families; the UI shows t('error.<code>')",
	)


class JobRequest(Strict):
	"""Body of POST /v1/jobs (validated by C, executed by A)."""

	source: dict = Field(description="A Source: sample, repo, upload, notebook or template")
	lens: str | None = Field(default=None, description="Optional hint; the script map decides")
	objective: dict | None = None
	constraints: Constraints | None = None
	gpu: str | None = Field(
		default=None,
		min_length=1,
		max_length=64,
		description="Requested GPU class; admission control may change it",
	)
	description: str | None = Field(default=None, description="The visitor's use case, for the guide")
	mode: JobMode = Field(
		default="plan",
		description="plan: map and probe a script (default). probe: one probe at `knobs`. verify: a short "
		"fixed-seed run of `knobs` against the record's current settings. sample_more: `tasks` more "
		"agent tasks. All but plan take a record source.",
	)
	knobs: dict[str, KnobValue] | None = Field(
		default=None, max_length=64, description="probe and verify: the full setting to run at"
	)
	tasks: int | None = Field(default=None, ge=1, le=50, description="sample_more: how many more tasks")
	purpose: str | None = Field(
		default=None,
		max_length=64,
		pattern=r"^[a-z][a-z0-9_.]*$",
		description="Who asked, for the logbook, e.g. guide.run_probe",
	)
	patch: str | None = Field(
		default=None,
		min_length=1,
		max_length=262_144,
		description="0.5.0: a unified diff applied to the record's code before a probe or verify job "
		"(C's fix loop); code fixes can't be passed as knob overrides",
	)

	@model_validator(mode="after")
	def _check(self):
		on_record = isinstance(self.source, dict) and self.source.get("kind") == "record"
		if self.mode != "plan" and not on_record:
			raise ValueError(f"mode {self.mode} needs a record source (kind=record, record_ref)")
		if on_record and not isinstance(self.source.get("record_ref"), str):
			raise ValueError("a record source needs record_ref")
		if self.mode == "probe" and not self.knobs:
			raise ValueError("mode probe needs knobs")
		if self.knobs is not None and self.mode not in ("probe", "verify"):
			raise ValueError("knobs go with mode probe or verify")
		if self.patch is not None and self.mode not in ("probe", "verify"):
			raise ValueError("patch goes with mode probe or verify")
		if (self.tasks is not None) != (self.mode == "sample_more"):
			raise ValueError("tasks goes with mode sample_more, and sample_more needs it")
		return self
