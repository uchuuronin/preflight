"""Modal smoke test: what a GPU reports to PyTorch, plus one timed matmul.

Run through GitHub Actions (see docs/setup/cloud.md):
	gh workflow run remote-run.yml -f name=smoke-l4 -f cmd="modal run scripts/cloud/smoke_modal.py --gpu L4"

Writes out/smoke_modal.json. Useful beyond the smoke test: `total_memory` is the capacity PyTorch
reports (B's open question on L4 usable memory), and the matmul gives an achieved bf16 TFLOP/s.
"""

import json
from pathlib import Path

import modal

app = modal.App("preflight-smoke")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy")


@app.function(image=image, gpu="L4", timeout=600)
def props() -> dict:
	import time

	import torch

	p = torch.cuda.get_device_properties(0)
	free, total = torch.cuda.mem_get_info()
	n = 8192
	a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
	b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
	for _ in range(3):
		a @ b
	torch.cuda.synchronize()
	start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	reps = 20
	start.record()
	for _ in range(reps):
		a @ b
	end.record()
	torch.cuda.synchronize()
	sec = start.elapsed_time(end) / 1000 / reps
	return {
		"name": p.name,
		"total_memory_bytes": p.total_memory,
		"mem_get_info_free_bytes": free,
		"mem_get_info_total_bytes": total,
		"multi_processor_count": p.multi_processor_count,
		"capability": f"{p.major}.{p.minor}",
		"torch": torch.__version__,
		"cuda": torch.version.cuda,
		"bf16_matmul_tflops": round(2 * n**3 / sec / 1e12, 1),
		"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
	}


@app.local_entrypoint()
def main(gpu: str = "L4"):
	out = props.with_options(gpu=gpu).remote()
	Path("out").mkdir(exist_ok=True)
	Path("out/smoke_modal.json").write_text(json.dumps(out, indent="\t") + "\n")
	print(json.dumps(out, indent="\t"))
