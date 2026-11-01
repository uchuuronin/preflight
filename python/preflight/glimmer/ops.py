"""Runs for the Glimmer endpoint, kept out of the deployed app (as runner/ops_app.py is): `modal run` starts a temporary
copy of whatever app a file defines, and the endpoint's app needs its key secret, which these runs don't.

    modal run --env main python/preflight/glimmer/ops.py::startup_check   # CPU: downloads, revisions, both images
    modal run --env main python/preflight/glimmer/ops.py::quantize        # CPU: the FP8 copy, once
    modal run --env main python/preflight/glimmer/ops.py::smoke --gpu a100 --calls 5
    modal run --env main python/preflight/glimmer/ops.py::smoke --gpu l4 --calls 5

Each writes out/glimmer_<name>.json. Agents start them only through remote-run.yml.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

from preflight.glimmer.common import (
	BF16_DIR,
	FP8_DIR,
	GGUF_DIR,
	GGUF_FILE,
	GGUF_REPO,
	GPUS,
	QUANT_IGNORE,
	REPO,
	USD_PER_H,
	WEIGHTS,
	_save_sources,
	_sources,
	fetch_image,
	llama_image,
	quant_image,
	run_smoke,
	vllm_image,
	weights,
)

app = modal.App("preflight-glimmer-ops")


@app.function(image=fetch_image, volumes={WEIGHTS: weights}, timeout=60 * 60, cpu=4, memory=8192)
def fetch() -> dict:
	"""Both of Meta's downloads into the volume, once; returns what each resolved to."""
	from huggingface_hub import HfApi, hf_hub_download, snapshot_download

	api = HfApi()
	src = _sources()
	for key, repo, kw, dest in (
		("bf16", REPO, {}, BF16_DIR),
		("gguf", GGUF_REPO, {"filename": GGUF_FILE}, GGUF_DIR),
	):
		if key in src:
			continue
		info = api.model_info(repo, files_metadata=True)
		t0 = time.monotonic()
		if key == "bf16":
			snapshot_download(repo, revision=info.sha, local_dir=dest)
			files = {s.rfilename: s.size for s in info.siblings}
		else:
			hf_hub_download(repo, revision=info.sha, local_dir=dest, **kw)
			files = {s.rfilename: s.size for s in info.siblings if s.rfilename == GGUF_FILE}
		src[key] = {
			"repo": repo,
			"revision": info.sha,
			"files": files,
			"bytes": sum(v or 0 for v in files.values()),
			"license": getattr(info.card_data, "license", None),
			"gated": info.gated,
			"downloaded_s": round(time.monotonic() - t0, 1),
			"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		}
		_save_sources(src)
	cfg = json.loads(Path(BF16_DIR, "config.json").read_text())
	idx = Path(BF16_DIR, "model.safetensors.index.json")
	names = sorted(json.loads(idx.read_text())["weight_map"]) if idx.exists() else []
	prefixes = sorted({".".join(n.split(".")[:3]) for n in names})
	return {"sources": src, "config": cfg, "weight_prefixes": prefixes[:80]}


@app.function(image=vllm_image, timeout=10 * 60)
def vllm_version() -> dict:
	"""The serving image without a GPU: vLLM's version and whether it knows Glimmer's parsers."""
	out = subprocess.run(["vllm", "serve", "--help=all"], capture_output=True, text=True, timeout=300)
	text = out.stdout + out.stderr
	import vllm

	return {
		"vllm": vllm.__version__,
		"muse_glimmer_parsers": "muse_glimmer" in text,
		"help_ok": out.returncode == 0,
	}


@app.function(image=llama_image, timeout=10 * 60)
def llama_version() -> dict:
	out = subprocess.run(["/app/llama-server", "--version"], capture_output=True, text=True, timeout=120)
	return {"llama_server": (out.stdout + out.stderr).strip()[-400:]}


@app.function(image=quant_image, volumes={WEIGHTS: weights}, timeout=60 * 60, cpu=8, memory=96 * 1024)
def make_fp8() -> dict:
	"""The FP8 copy of Meta's bf16 checkpoint: FP8_DYNAMIC (per-channel weights, per-token activations), no
	calibration data, so it's a pure function of the official weights. vLLM serves it on Ampere as FP8 weights
	with bf16 compute (Marlin)."""
	import transformers
	from llmcompressor import oneshot
	from llmcompressor.modifiers.quantization import QuantizationModifier

	src = _sources()
	if "bf16" not in src:
		raise RuntimeError("run startup_check first: the bf16 checkpoint isn't in the volume")
	if src.get("fp8", {}).get("from_revision") == src["bf16"]["revision"]:
		return {"skipped": "the FP8 copy of this revision is already there", "fp8": src["fp8"]}
	t0 = time.monotonic()
	cfg = transformers.AutoConfig.from_pretrained(BF16_DIR)
	loaders = [transformers.AutoModelForImageTextToText, transformers.AutoModelForCausalLM]
	model, err = None, []
	for loader in loaders:
		try:
			model = loader.from_pretrained(BF16_DIR, dtype="bfloat16", low_cpu_mem_usage=True)
			break
		except (ValueError, KeyError) as exc:  # the class this architecture doesn't register
			err.append(f"{loader.__name__}: {exc}")
	if model is None:
		raise RuntimeError("; ".join(err))
	recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=QUANT_IGNORE)
	oneshot(model=model, recipe=recipe)
	model.save_pretrained(FP8_DIR, save_compressed=True)
	# the tokenizer, chat template, generation and processor configs: every small file the save didn't write
	for p in Path(BF16_DIR).iterdir():
		dest = Path(FP8_DIR, p.name)
		if (
			p.is_file()
			and not dest.exists()
			and p.suffix not in (".safetensors", ".bin")
			and "index" not in p.name
		):
			dest.write_bytes(p.read_bytes())
	size = sum(f.stat().st_size for f in Path(FP8_DIR).glob("*.safetensors"))
	src["fp8"] = {
		"from_repo": REPO,
		"from_revision": src["bf16"]["revision"],
		"scheme": "FP8_DYNAMIC (llmcompressor 0.15.0), ignore " + ", ".join(QUANT_IGNORE),
		"architectures": getattr(cfg, "architectures", None),
		"bytes": size,
		"made_s": round(time.monotonic() - t0, 1),
		"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
	}
	_save_sources(src)
	return {"fp8": src["fp8"]}


@app.function(image=vllm_image, gpu=GPUS["a100"], volumes={WEIGHTS: weights}, timeout=25 * 60)
def smoke_a100(calls: int, concurrent: int) -> dict:
	return run_smoke("vllm", calls, concurrent)


@app.function(image=llama_image, gpu=GPUS["l4"], volumes={WEIGHTS: weights}, timeout=20 * 60)
def smoke_l4(calls: int, concurrent: int) -> dict:
	return run_smoke("llama", calls, concurrent)


# ---------------------------------------------------------------- local entrypoints (write out/)


def _write(name: str, data: dict) -> None:
	Path("out").mkdir(exist_ok=True)
	Path(f"out/glimmer_{name}.json").write_text(json.dumps(data, indent=1, default=str))
	print(json.dumps(data, indent=1, default=str)[:6000])


@app.local_entrypoint()
def startup_check():
	"""CPU only: both downloads (once), their revisions and licence, the config, and the two serving images."""
	t0 = time.monotonic()
	got = fetch.remote()
	got["vllm"] = vllm_version.remote()
	got["llama_cpp"] = llama_version.remote()
	got["wall_s"] = round(time.monotonic() - t0, 1)
	_write("startup_check", got)


@app.local_entrypoint()
def quantize():
	t0 = time.monotonic()
	got = make_fp8.remote()
	got["wall_s"] = round(time.monotonic() - t0, 1)
	_write("quantize", got)


@app.local_entrypoint()
def smoke(gpu: str = "a100", calls: int = 5, concurrent: int = 4):
	"""Cold start (container asked for to server ready), then `calls` triages one at a time and `concurrent` at
	once. Spend is the container's whole life on the GPU at Modal's rate."""
	fn = {"a100": smoke_a100, "l4": smoke_l4}[gpu]
	t0 = time.monotonic()
	got = fn.remote(calls, concurrent)
	wall = time.monotonic() - t0
	g = GPUS[gpu]
	got.update(
		{
			"gpu": g,
			"client_wall_s": round(wall, 1),
			"cold_start_s": round(wall - got["container_s"] + got["server_ready_s"], 1),
			"usd_upper": round(wall / 3600 * USD_PER_H[g], 3),
			"usd_container": round(got["container_s"] / 3600 * USD_PER_H[g], 3),
		}
	)
	_write(f"smoke_{gpu}", got)
