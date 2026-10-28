"""Contract v0.1 — the single source of truth for every shape Preflight passes around.

Edit only through Chiya. After the freeze (Tue Oct 27, noon ET) changes are
additive and bump the minor version; run `uv run python -m preflight.schema.generate`
after any change so schema/*.schema.json and schema/ts/ stay in step.
"""

import gzip
import json
import zlib

from .base import LENSES, BlobRef, KnobValue, Lens, Strict
from .encoding import MAX_GRID_CELLS, MAX_RECORD_GZIP_BYTES
from .events import JobEvent, JobRequest
from .ids import new_share_token, new_ulid, new_write_token
from .migrations import UnsupportedSchemaVersion, upgrade
from .prediction import Estimate, Grid, GridAxis, KnobImpact, Objective, Prediction, Predictions, Target
from .record import (
	GPU,
	SCHEMA_VERSION,
	AgentSample,
	AgentTask,
	BillingMode,
	Campaign,
	CampaignCell,
	CheckpointStats,
	Constraints,
	CostBasis,
	CostLine,
	CostLineKind,
	Costs,
	DataStats,
	Domain,
	Env,
	EvalStats,
	Evidence,
	Finding,
	FlightRecord,
	GenerationStats,
	Graph,
	GraphEdge,
	GraphNode,
	Job,
	JobCost,
	KernelStat,
	KernelTable,
	Knob,
	LengthStats,
	Memory,
	ModelCall,
	ModelCard,
	Patch,
	Power,
	Probe,
	ProbeCost,
	Range,
	RLSignals,
	Source,
	Stability,
	StepSample,
	StepTime,
	Throughput,
	new_record,
)
from .script_map import ScriptKnob, ScriptMap

MAX_RECORD_RAW_BYTES = 20 * 1024 * 1024  # a record is under 1 MB gzipped; 20 MB raw is generous


def load_record(data: str | bytes | dict) -> FlightRecord:
	"""Parse a stored record (JSON text, gzipped bytes or a dict), upgrading old versions."""
	if isinstance(data, bytes):
		if data[:2] == b"\x1f\x8b":
			d = zlib.decompressobj(16 + zlib.MAX_WBITS)
			out = d.decompress(data, MAX_RECORD_RAW_BYTES + 1)
			if len(out) > MAX_RECORD_RAW_BYTES or d.unconsumed_tail:
				raise ValueError(
					f"record is too large once decompressed (over {MAX_RECORD_RAW_BYTES // 2**20} MB)"
				)
			data = out
		if len(data) > MAX_RECORD_RAW_BYTES:
			raise ValueError(f"record is too large (over {MAX_RECORD_RAW_BYTES // 2**20} MB)")
		data = data.decode("utf-8")
	elif isinstance(data, str) and len(data) > MAX_RECORD_RAW_BYTES:
		raise ValueError(f"record is too large (over {MAX_RECORD_RAW_BYTES // 2**20} MB)")
	raw = json.loads(data) if isinstance(data, str) else data
	return FlightRecord.model_validate(upgrade(raw))


def dump_record(rec: FlightRecord, *, gzipped: bool = False) -> bytes:
	"""Serialise for storage. Raises if the gzipped record breaks the 1 MB budget."""
	raw = rec.model_dump_json(exclude_none=True).encode("utf-8")
	packed = gzip.compress(raw, mtime=0)
	if len(packed) > MAX_RECORD_GZIP_BYTES:
		raise ValueError(f"record is {len(packed)} bytes gzipped; budget is {MAX_RECORD_GZIP_BYTES}")
	return packed if gzipped else raw


__all__ = [
	"GPU",
	"LENSES",
	"MAX_GRID_CELLS",
	"MAX_RECORD_GZIP_BYTES",
	"SCHEMA_VERSION",
	"AgentSample",
	"AgentTask",
	"BillingMode",
	"BlobRef",
	"Campaign",
	"CampaignCell",
	"CheckpointStats",
	"Constraints",
	"CostBasis",
	"CostLine",
	"CostLineKind",
	"Costs",
	"DataStats",
	"Domain",
	"Env",
	"Estimate",
	"EvalStats",
	"Evidence",
	"Finding",
	"FlightRecord",
	"GenerationStats",
	"Graph",
	"GraphEdge",
	"GraphNode",
	"Grid",
	"GridAxis",
	"Job",
	"JobCost",
	"JobEvent",
	"JobRequest",
	"KernelStat",
	"KernelTable",
	"Knob",
	"KnobImpact",
	"KnobValue",
	"LengthStats",
	"Lens",
	"Memory",
	"ModelCall",
	"ModelCard",
	"Objective",
	"Patch",
	"Power",
	"Prediction",
	"Predictions",
	"Probe",
	"ProbeCost",
	"RLSignals",
	"Range",
	"ScriptKnob",
	"ScriptMap",
	"Source",
	"Stability",
	"StepSample",
	"StepTime",
	"Strict",
	"Target",
	"Throughput",
	"UnsupportedSchemaVersion",
	"dump_record",
	"load_record",
	"new_record",
	"new_share_token",
	"new_ulid",
	"new_write_token",
	"upgrade",
]
