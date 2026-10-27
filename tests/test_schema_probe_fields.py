"""v0.1 additions for B: Probe.kernels, the model-shape fields on ModelInfo, bounded GPU names."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from preflight.schema import Constraints, FlightRecord, KernelStat, KernelTable
from preflight.schema.events import JobRequest
from preflight.schema.script_map import ModelInfo

ROOT = Path(__file__).resolve().parents[1]


def golden(name):
	return FlightRecord.model_validate(json.loads((ROOT / f"schema/examples/{name}.json").read_text()))


@pytest.mark.parametrize("name", ["finetune", "pretrain"])
def test_golden_current_settings_probe_carries_a_kernel_table(name):
	rec = golden(name)
	p1 = rec.probes[0]
	assert p1.kernels is not None and p1.kernels.rows
	assert sum(r.device_s for r in p1.kernels.rows) < p1.step_time.p50_s  # kernels fit inside the step
	assert all(rec.probes[i].kernels is None for i in range(1, len(rec.probes)))  # only p1 is profiled
	for p in rec.probes:
		assert {"params_total", "params_trainable", "tokens_per_step", "samples_per_step",
			"micro_steps_per_step"} <= set(p.extras)  # fmt: skip


def test_golden_finetune_model_has_its_shape():
	m = golden("finetune").job.script_map.model
	assert (m.gated_mlp, m.tie_embeddings, m.norm, m.attn_impl, m.head_dim) == (
		True,
		True,
		"rmsnorm",
		"sdpa",
		128,
	)


def test_kernel_rows_reject_nan_and_negative_time():
	ok = {"name": "aten::mm", "kernel_class": "gemm", "calls": 1, "device_s": 0.1, "dtype": "bf16"}
	KernelStat(**ok)
	for bad in ({"device_s": float("nan")}, {"device_s": -1.0}, {"flops": float("inf")}, {"name": "x" * 201}):
		with pytest.raises(ValidationError):
			KernelStat(**{**ok, **bad})
	with pytest.raises(ValidationError):
		KernelTable(rows=[], steps=3, backend="tpu", allow_tf32=False)


def test_gpu_names_from_visitors_are_bounded():
	JobRequest(source={"kind": "sample"}, gpu="H100-SXM-80GB")
	with pytest.raises(ValidationError):
		JobRequest(source={"kind": "sample"}, gpu="1" * 65)
	with pytest.raises(ValidationError):
		Constraints(gpu_types=["L4", "1" * 65])
	with pytest.raises(ValidationError):
		Constraints(gpu_types=[""])
	with pytest.raises(ValidationError):
		JobRequest(source={"kind": "sample"}, gpu="")
	# constraints are typed at the edge, so a long name is a 422 there, not an error deep in the pipeline
	with pytest.raises(ValidationError):
		JobRequest(source={"kind": "sample"}, constraints={"gpu_types": ["1" * 65]})
	assert JobRequest(source={"kind": "sample"}, constraints={"gpu_types": ["L4"]}).constraints.gpu_types == [
		"L4"
	]


def test_model_norm_is_one_of_two():
	ModelInfo(name="m", norm="rmsnorm", shape_source="hf_config")
	with pytest.raises(ValidationError):
		ModelInfo(name="m", norm="batchnorm")
