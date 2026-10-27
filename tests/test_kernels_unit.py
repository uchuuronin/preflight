"""Kernel-table arithmetic on synthetic profiler events: GPU attribution, attention layouts, matmul bytes."""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")
from torch.autograd import DeviceType

from preflight.recorder.kernels import _attention_flops, _bytes, kernel_table

pytestmark = pytest.mark.torch


def ev(key, *, dev=DeviceType.CPU, self_dev=0.0, self_cpu=0.0, count=1, shapes=None, flops=0, ann=False):
	return NS(key=key, device_type=dev, self_device_time_total=self_dev, self_cpu_time_total=self_cpu,
		count=count, input_shapes=shapes or [], flops=flops, is_user_annotation=ann)  # fmt: skip


class Prof:
	def __init__(self, events):
		self.events = events

	def key_averages(self, group_by_input_shape=False):
		return self.events


def table(events, **kw):
	return kernel_table(Prof(events), steps=1, device="cuda", dtype="bf16", allow_tf32=False,
		classify=lambda n: "gemm" if "mm" in n else "other", **kw)  # fmt: skip


def test_gpu_annotation_spans_are_not_device_time():
	t = table([
		ev("aten::mm", self_dev=900.0, shapes=[[64, 64], [64, 64]]),
		ev("ampere_bf16_gemm", dev=DeviceType.CUDA, self_dev=900.0),
		# ProfilerStep#3 and Optimizer.step mirrored on the GPU timeline: they span the gemm
		ev("ProfilerStep#3", dev=DeviceType.CUDA, self_dev=1000.0, ann=True),
		ev("Optimizer.step#AdamW.step", dev=DeviceType.CUDA, self_dev=950.0, ann=True),
		ev("my_visitor_block", self_dev=0.0, ann=True),
	])  # fmt: skip
	names = [r["name"] for r in t["rows"]]
	assert names == ["aten::mm"]  # no unattributed row: every real kernel microsecond is claimed


def test_unclaimed_kernels_still_get_their_row():
	t = table([ev("aten::mm", self_dev=500.0), ev("custom_ext_kernel", dev=DeviceType.CUDA, self_dev=1000.0)])
	assert t["rows"][-1]["name"] == "(unattributed device time)"
	assert t["rows"][-1]["device_s"] == pytest.approx(500e-6)


def test_attention_flops_by_layout():
	b, h, lq, ls, d = 2, 8, 128, 128, 64
	dense = 4.0 * b * h * lq * ls * d
	bhld = [[b, h, lq, d], [b, h, ls, d], [b, h, ls, d]]
	blhd = [[b, lq, h, d], [b, ls, h, d], [b, ls, h, d]]
	assert _attention_flops("aten::scaled_dot_product_attention", bhld) == dense
	assert _attention_flops("aten::_scaled_dot_product_flash_attention", bhld) == dense
	assert _attention_flops("aten::_flash_attention_forward", blhd) == dense
	assert _attention_flops("aten::_efficient_attention_forward", [*blhd, [], []]) == dense
	# backward ops take grad_out first
	assert _attention_flops("aten::_flash_attention_backward", [[b, lq, h, d], *blhd]) == 2.5 * dense
	assert (
		_attention_flops("aten::_scaled_dot_product_flash_attention_backward", [[b, h, lq, d], *bhld])
		== 2.5 * dense
	)
	# varlen packing (3-d q): no guess
	assert _attention_flops("aten::_flash_attention_forward", [[512, h, d], [512, h, d]]) is None
	assert _attention_flops("aten::mm", bhld) is None


def test_matmul_bytes_count_the_right_operands():
	# linear: input [8, 32, 64], weight [128, 64], bias [128] -> output [8, 32, 128]
	read = 8 * 32 * 64 + 128 * 64 + 128
	assert _bytes("aten::linear", [[8, 32, 64], [128, 64], [128]], 2) == (read + 8 * 32 * 128) * 2
	# addmm: bias, a [16, 64], b [64, 32]
	assert _bytes("aten::addmm", [[32], [16, 64], [64, 32]], 2) == (32 + 16 * 64 + 64 * 32 + 16 * 32) * 2
	# _scaled_mm: a, b, scales (0-d), bias last
	assert _bytes("aten::_scaled_mm", [[16, 64], [64, 32], [], [], [32]], 1) == (
		16 * 64 + 64 * 32 + 32 + 16 * 32
	)
