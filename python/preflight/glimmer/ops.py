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
	"""The serving image without a GPU: vLLM's version, whether it knows Glimmer's parsers, and the CUDA that torch
	was built for against the base image's (review, Nov 1: a mismatch shows here, not on a paid GPU)."""
	out = subprocess.run(["vllm", "serve", "--help=all"], capture_output=True, text=True, timeout=300)
	text = out.stdout + out.stderr
	nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=60)
	import torch
	import vllm

	return {
		"vllm": vllm.__version__,
		"torch": torch.__version__,
		"torch_cuda": torch.version.cuda,
		"nvcc": (nvcc.stdout or nvcc.stderr).strip().splitlines()[-1:],
		"muse_glimmer_parsers": "muse_glimmer" in text,
		"help_ok": out.returncode == 0,
	}


@app.function(image=llama_image, timeout=10 * 60)
def llama_version() -> dict:
	out = subprocess.run(["/app/llama-server", "--version"], capture_output=True, text=True, timeout=120)
	return {"llama_server": (out.stdout + out.stderr).strip()[-400:]}


@app.function(
	image=quant_image, volumes={WEIGHTS: weights}, timeout=40 * 60, cpu=8, memory=96 * 1024
)  # worst case about $0.77
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


# worst cases: A100 18 min $0.63, L4 35 min $0.47, quantize's $0.77: $1.87 of GPU and CPU, so Modal's memory and
# CPU charges on the GPU containers fit under Chiya's $2 (Nov 1)
@app.function(image=vllm_image, gpu=GPUS["a100"], volumes={WEIGHTS: weights}, timeout=18 * 60)
def smoke_a100(calls: int, concurrent: int) -> dict:
	return run_smoke("vllm", calls, concurrent, ready_s=12 * 60)


@app.function(image=llama_image, gpu=GPUS["l4"], volumes={WEIGHTS: weights}, timeout=35 * 60)
def smoke_l4(calls: int, concurrent: int) -> dict:
	return run_smoke("llama", calls, concurrent, ready_s=10 * 60)


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
	for key, fn in (("vllm", vllm_version), ("llama_cpp", llama_version)):
		try:
			got[key] = fn.remote()
		except Exception as exc:  # the downloads already landed: keep what's known
			got[key] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
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


@app.local_entrypoint()
def deploy_check():
	"""After `modal deploy`: the endpoint's URL, that it refuses a call without a proxy token (Modal's proxy answers
	401 before the app's code runs), and the containers the deployed app has running (0 when scaled to zero).
	Starts no GPU. Writes what it found even when a step fails."""
	import urllib.error
	import urllib.request

	from preflight.glimmer.common import APP_NAME

	found: dict = {}
	try:
		cls = modal.Cls.from_name(APP_NAME, "Glimmer", environment_name="main")
		found["url"] = cls().serve.get_web_url()
	except Exception as exc:
		found["url_error"] = f"{type(exc).__name__}: {exc}"[:400]
	if found.get("url"):
		try:
			with urllib.request.urlopen(f"{found['url']}/v1/models", timeout=30) as r:
				found["without_a_token"] = {"status": r.status, "refused": False}
		except urllib.error.HTTPError as exc:
			found["without_a_token"] = {"status": exc.code, "refused": exc.code in (401, 403)}
		except OSError as exc:
			found["without_a_token"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
	out = subprocess.run(
		["modal", "container", "list", "--env", "main", "--json"], capture_output=True, text=True, timeout=120
	)
	try:
		rows = json.loads(out.stdout or "[]")
		mine = [
			c for c in rows if isinstance(c, dict) and c.get("app_name") == APP_NAME
		]  # review: exact match
		found["containers_running"] = len(mine)
		found["containers"] = mine
	except ValueError:
		found["containers_error"] = (out.stderr or out.stdout)[-400:]
	_write("deploy_check", found)
