"""Contract v0.1: the flight record and the prediction-API types."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from preflight.schema import (
	SCHEMA_VERSION,
	Estimate,
	FlightRecord,
	Grid,
	GridAxis,
	Job,
	Knob,
	new_record,
	new_share_token,
	new_ulid,
)
from preflight.schema.encoding import MAX_GRID_CELLS, decode_f16, decode_mask, encode_f16, encode_mask
from preflight.schema.migrations import UnsupportedSchemaVersion, upgrade

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "schema" / "examples"
GOLDEN = ["finetune", "pretrain", "agent"]


def test_schema_version_is_semver_0_x():
	# 0.2.0 (Oct 27): run cost fields; 0.3.0 (Oct 28): at_risk_gpu_hours. Both additive over the 0.1.0 freeze
	assert SCHEMA_VERSION == "0.3.0"


def test_ulid_is_26_crockford_chars_and_sortable():
	a, b = new_ulid(), new_ulid()
	assert len(a) == 26
	assert set(a) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
	assert a[:10] <= b[:10]


def test_share_token_does_not_leak_creation_time():
	token = new_share_token()
	assert len(token) >= 22
	assert token != new_share_token()
	# not a ULID: no shared time prefix with a ULID made at the same moment
	assert not token.startswith(new_ulid()[:6])


def test_new_record_is_valid_and_minimal():
	rec = new_record(job=Job(lens="finetune", source={"kind": "sample", "uri": "sample://lora-hf"}))
	assert rec.schema_version == SCHEMA_VERSION
	assert rec.share_token != rec.record_id
	FlightRecord.model_validate_json(rec.model_dump_json())


def test_unknown_fields_are_rejected():
	rec = new_record(job=Job(lens="pretrain", source={"kind": "upload"}))
	data = json.loads(rec.model_dump_json())
	data["jobb"] = {}
	with pytest.raises(ValidationError):
		FlightRecord.model_validate(data)


def test_unknown_lens_is_rejected():
	with pytest.raises(ValidationError):
		Job(lens="quantum", source={"kind": "upload"})


@pytest.mark.parametrize("name", GOLDEN)
def test_golden_record_validates(name):
	raw = (EXAMPLES / f"{name}.json").read_text()
	rec = FlightRecord.model_validate_json(raw)
	assert rec.job.lens == name


@pytest.mark.parametrize("name", GOLDEN)
def test_golden_record_round_trips_unchanged(name):
	raw = json.loads((EXAMPLES / f"{name}.json").read_text())
	rec = FlightRecord.model_validate(raw)
	again = json.loads(rec.model_dump_json(exclude_none=True))
	assert again == raw


@pytest.mark.parametrize("name", GOLDEN)
def test_golden_record_under_1mb_gzipped(name):
	raw = (EXAMPLES / f"{name}.json").read_bytes()
	assert len(gzip.compress(raw)) < 1_000_000


@pytest.mark.parametrize("name", GOLDEN)
def test_golden_knob_ids_unique_and_probe_knobs_declared(name):
	rec = FlightRecord.model_validate_json((EXAMPLES / f"{name}.json").read_text())
	ids = [k.id for k in rec.knobs]
	assert len(ids) == len(set(ids))
	for probe in rec.probes:
		assert set(probe.knobs) <= set(ids)


@pytest.mark.parametrize("name", GOLDEN)
def test_golden_graph_edges_reference_nodes(name):
	rec = FlightRecord.model_validate_json((EXAMPLES / f"{name}.json").read_text())
	nodes = {n.id for n in rec.graph.nodes}
	for e in rec.graph.edges:
		assert e.source in nodes and e.target in nodes
	for n in rec.graph.nodes:
		assert n.parent is None or n.parent in nodes


def test_graph_rejects_dangling_edge():
	with pytest.raises(ValidationError):
		FlightRecord.model_validate(
			{
				**json.loads(
					new_record(job=Job(lens="finetune", source={"kind": "upload"})).model_dump_json()
				),
				"graph": {
					"nodes": [{"id": "a", "kind": "module", "label": "a"}],
					"edges": [{"source": "a", "target": "b"}],
				},
			}
		)


def test_probe_with_undeclared_knob_is_rejected():
	base = json.loads((EXAMPLES / "finetune.json").read_text())
	base["probes"][0]["knobs"]["not_a_knob"] = 3
	with pytest.raises(ValidationError):
		FlightRecord.model_validate(base)


def test_f16_round_trip_keeps_values_within_half_precision():
	values = np.array([0.0, 1.5, 1234.5, 65504.0, -3.25], dtype=np.float64)
	out = decode_f16(encode_f16(values), len(values))
	np.testing.assert_allclose(out, values, rtol=1e-3)


def test_f16_encodes_overflow_as_inf_not_garbage():
	out = decode_f16(encode_f16(np.array([1e6])), 1)
	assert np.isinf(out[0])


def test_mask_round_trip():
	mask = np.array([True, False, False, True, True, False, True, False, True])
	assert (decode_mask(encode_mask(mask), len(mask)) == mask).all()


def _grid(n_a: int, n_b: int) -> Grid:
	cells = n_a * n_b
	z = encode_f16(np.zeros(cells))
	return Grid(
		id="g",
		target_id="memory.peak",
		unit="bytes",
		axes=[
			GridAxis(knob_id="batch_size", values=list(range(n_a))),
			GridAxis(knob_id="seq_len", values=list(range(n_b))),
		],
		point=z,
		lo=z,
		hi=z,
		abstain=encode_mask(np.zeros(cells, dtype=bool)),
		model_id="analytical-memory@0.1",
	)


def test_grid_rejects_values_that_overflowed_float16():
	inf = encode_f16(np.full(4, 1e9))
	with pytest.raises(ValidationError, match="finite"):
		Grid(
			id="g",
			target_id="memory.peak",
			unit="bytes",
			axes=[GridAxis(knob_id="batch_size", values=[1, 2, 4, 8])],
			point=inf,
			lo=inf,
			hi=inf,
			abstain=encode_mask(np.zeros(4, dtype=bool)),
			model_id="m",
		)


def test_scaled_encoding_holds_bytes_and_long_runs():
	from preflight.schema.encoding import choose_scale

	values = np.array([3.2e9, 12.7e9, 81.0e9, 0.0])
	scale = choose_scale(values)
	assert scale > 1 and math.log2(scale).is_integer()
	out = decode_f16(encode_f16(values, scale=scale)) * scale
	np.testing.assert_allclose(out, values, rtol=1e-3)


def test_grid_values_applies_scale():
	from preflight.schema.encoding import choose_scale

	vals = np.array([3.2e9, 12.7e9, 81.0e9])
	s = choose_scale(vals)
	enc = encode_f16(vals, scale=s)
	g = Grid(
		id="g",
		target_id="memory.peak",
		unit="bytes",
		axes=[GridAxis(knob_id="batch_size", values=[1, 2, 4])],
		point=enc,
		lo=enc,
		hi=enc,
		abstain=encode_mask(np.zeros(3, dtype=bool)),
		model_id="m",
		scale=s,
	)
	np.testing.assert_allclose(g.values("point"), vals, rtol=1e-3)


def test_grid_at_cell_budget_is_accepted():
	g = _grid(128, 128)
	assert g.cells == MAX_GRID_CELLS


def test_grid_over_cell_budget_without_sidecar_is_rejected():
	with pytest.raises(ValidationError):
		_grid(129, 128)


def test_grid_payload_length_must_match_axes():
	with pytest.raises(ValidationError):
		Grid(
			id="g",
			target_id="t",
			unit="s",
			axes=[GridAxis(knob_id="k", values=[1, 2, 3])],
			point=encode_f16(np.zeros(2)),
			lo=encode_f16(np.zeros(3)),
			hi=encode_f16(np.zeros(3)),
			abstain=encode_mask(np.zeros(3, dtype=bool)),
			model_id="m",
		)


def test_estimate_abstain_needs_reason():
	with pytest.raises(ValidationError):
		Estimate(point=None, lo=None, hi=None, unit="bytes", status="abstain", model_id="m", source="grid")
	Estimate(
		point=None,
		lo=None,
		hi=None,
		unit="bytes",
		status="abstain",
		model_id="m",
		source="grid",
		reason="out of probed range",
	)


def test_estimate_interval_must_bracket_point():
	with pytest.raises(ValidationError):
		Estimate(point=10, lo=11, hi=12, unit="s", status="predicted", model_id="m", source="exact")


def test_knob_current_value_must_sit_in_domain():
	with pytest.raises(ValidationError):
		Knob(
			id="batch_size",
			label_key="knob.batch_size",
			unit="samples",
			kind="int",
			domain={"min": 1, "max": 64},
			current=128,
		)


def test_upgrade_current_version_is_identity():
	raw = json.loads((EXAMPLES / "finetune.json").read_text())
	assert upgrade(raw) == raw


def test_upgrade_rejects_newer_major():
	raw = json.loads((EXAMPLES / "finetune.json").read_text())
	raw["schema_version"] = "1.0.0"
	with pytest.raises(UnsupportedSchemaVersion):
		upgrade(raw)


def test_load_record_upgrades_then_validates():
	from preflight.schema import load_record

	rec = load_record((EXAMPLES / "agent.json").read_text())
	assert isinstance(rec, FlightRecord)


def test_generated_json_schema_is_up_to_date():
	from preflight.schema.generate import json_schemas

	for name, schema in json_schemas().items():
		on_disk = json.loads((ROOT / "schema" / f"{name}.schema.json").read_text())
		assert on_disk == schema, (
			f"schema/{name}.schema.json is stale: run `uv run python -m preflight.schema.generate`"
		)


def test_probe_carries_structured_error_eval_checkpoint_and_generation_stats():
	from preflight.schema import CheckpointStats, EvalStats, GenerationStats, Probe

	p = Probe(
		id="p1",
		knobs={},
		status="oom",
		steps=3,
		error="CUDA out of memory during eval",
		error_code="oom.eval",
		eval=EvalStats(
			batches=2,
			batch_size=32,
			seconds_per_batch=0.8,
			peak_allocated_bytes=38_000_000_000,
			grad_enabled=True,
		),
		checkpoint=CheckpointStats(saves=1, seconds_mean=12.5, bytes_mean=3_100_000_000),
		generation=GenerationStats(
			calls=4,
			seconds_total=9.0,
			new_tokens_mean=380,
			new_tokens_p95=512,
			new_tokens_max=512,
			truncation_rate=0.92,
			max_new_tokens=512,
		),
	)
	assert Probe.model_validate_json(p.model_dump_json()) == p


def test_error_code_must_be_a_known_family():
	from preflight.schema import Probe

	with pytest.raises(ValidationError):
		Probe(id="p", knobs={}, status="error", steps=0, error_code="Not A Code")


def test_scale_ignores_inf_and_handles_tiny_values():
	from preflight.schema.encoding import choose_scale

	assert choose_scale(np.array([8e10, np.inf])) > 1
	tiny = np.array([1e-8, 3e-8, 5e-9])
	s = choose_scale(tiny)
	assert s < 1
	np.testing.assert_allclose(decode_f16(encode_f16(tiny, scale=s)) * s, tiny, rtol=1e-3)


def test_load_record_refuses_a_gzip_bomb():
	from preflight.schema import load_record

	bomb = gzip.compress(b" " * (64 * 1024 * 1024))
	with pytest.raises(ValueError, match="too large"):
		load_record(bomb)
