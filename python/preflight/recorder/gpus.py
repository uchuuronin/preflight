"""Peak dense FLOP/s for MFU, by GPU and compute precision.

Minimal on purpose: B's engine owns the full NVIDIA and AMD spec table and
supersedes this once it lands (TODO(contract) in docs/requests.md). Values are
vendor datasheet dense (non-sparse) peaks.
"""

# name fragment (lowercase) -> {precision: FLOP/s}
PEAK_FLOPS: list[tuple[str, dict[str, float]]] = [
	("h200", {"bf16": 989e12, "fp16": 989e12, "fp32": 67e12, "tf32": 494e12}),
	("h100 pcie", {"bf16": 756e12, "fp16": 756e12, "fp32": 51e12, "tf32": 378e12}),
	("h100", {"bf16": 989e12, "fp16": 989e12, "fp32": 67e12, "tf32": 494e12}),
	("a100", {"bf16": 312e12, "fp16": 312e12, "fp32": 19.5e12, "tf32": 156e12}),
	("l40s", {"bf16": 362e12, "fp16": 362e12, "fp32": 91.6e12, "tf32": 183e12}),
	("l4", {"bf16": 121e12, "fp16": 121e12, "fp32": 30.3e12, "tf32": 60e12}),
	("a10g", {"bf16": 70e12, "fp16": 70e12, "fp32": 31.2e12, "tf32": 35e12}),
	("a10", {"bf16": 125e12, "fp16": 125e12, "fp32": 31.2e12, "tf32": 62.5e12}),
	("t4", {"bf16": 65e12, "fp16": 65e12, "fp32": 8.1e12}),
	("rtx 4090", {"bf16": 165e12, "fp16": 165e12, "fp32": 82.6e12, "tf32": 82.6e12}),
	("mi300x", {"bf16": 1307e12, "fp16": 1307e12, "fp32": 163.4e12, "tf32": 653.7e12}),
	("mi250", {"bf16": 362e12, "fp16": 362e12, "fp32": 47.9e12, "tf32": 181e12}),
]


def peak_flops(gpu_name: str | None, precision: str) -> float | None:
	if not gpu_name:
		return None
	name = gpu_name.lower()
	for frag, table in PEAK_FLOPS:
		if frag in name:
			return table.get(precision) or table.get("fp32")
	return None
