"""Shared by the Glimmer endpoint (app.py) and its runs (ops.py): sources, images, the serve commands and the
triage call the smokes make. See app.py for what runs where and why."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "preflight-glimmer"
MODEL_ID = "muse-glimmer"  # the served model name callers send
REPO = "meta-models/Muse-Glimmer-30B"
GGUF_REPO = "meta-models/Muse-Glimmer-30B-GGUF"
GGUF_FILE = "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"  # Meta's page: "start here", fits 24 GB
WEIGHTS = "/weights"
BF16_DIR = f"{WEIGHTS}/bf16"
FP8_DIR = f"{WEIGHTS}/fp8-dynamic"
GGUF_DIR = f"{WEIGHTS}/gguf"
SOURCES = f"{WEIGHTS}/sources.json"  # repo, file, resolved revision and sizes of each download

# cited: recipes.vllm.ai/meta-models/Muse-Glimmer-30B lists vLLM 0.28.0 and this image; the parsers are required
VLLM_IMAGE = "vllm/vllm-openai:v0.28.0"
# cited: Meta's GGUF page needs llama.cpp at or after commit 62bf73d (PR #26841); the build used is recorded per run
LLAMA_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
PINS = {
	"huggingface_hub": "huggingface_hub[hf_xet]==2.4.0",
	"llmcompressor": "llmcompressor==0.15.0",
	"openai": "openai==2.8.1",  # the llama.cpp image has none; vLLM's image brings its own
}
MAX_MODEL_LEN = (
	32768  # guess: room for a triage (about 8K in, 1K out) with KV to spare on 40 GB; measured per run
)
IDLE_S = 120  # Chiya, Nov 1: scale to zero after 2 minutes idle
PORT = 8000
# cited: the vLLM recipe's published sampling (greedy isn't recommended for Glimmer)
SAMPLING = {"temperature": 1.0, "top_p": 0.95}
GPUS = {"a100": "A100-40GB", "l4": "L4"}
# provider: modal.com/pricing (engine/prices, Nov 1), for the spend line each smoke prints
USD_PER_H = {"A100-40GB": 2.10, "L4": 0.80}

weights = modal.Volume.from_name("preflight-glimmer-weights", create_if_missing=True)

fetch_image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(PINS["huggingface_hub"])
	.add_local_python_source("preflight")
)
quant_image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(PINS["llmcompressor"], PINS["huggingface_hub"])
	.add_local_python_source("preflight")
)
vllm_image = (
	modal.Image.from_registry(VLLM_IMAGE)
	.entrypoint([])
	.env({"HF_HUB_OFFLINE": "1"})  # the weights come from the volume, never the hub at serve time
	.add_local_python_source("preflight")
)
llama_image = (
	modal.Image.from_registry(LLAMA_IMAGE, add_python="3.11")
	.entrypoint([])
	.pip_install(PINS["openai"])
	.add_local_python_source("preflight")
)


def _sources() -> dict:
	p = Path(SOURCES)
	return json.loads(p.read_text()) if p.exists() else {}


def _save_sources(s: dict) -> None:
	Path(SOURCES).write_text(json.dumps(s, indent=1))
	weights.commit()


# ignore: the vision tower and its projector stay bf16 (text-only triage never runs them), and so does lm_head
QUANT_IGNORE = ["lm_head", "re:.*vision.*", "re:.*visual.*", "re:.*mm_proj.*", "re:.*multi_modal_projector.*"]


# ---------------------------------------------------------------- the triage call both smokes make


def triage_messages() -> list[dict]:
	"""One triage of a probe log, shaped like Preflight's (about 8K tokens in, a short JSON answer out). The log is
	synthetic and fixed, so every run sends the same prompt."""
	lines = [
		"[preflight] probe p1: finetune_lora_hf, batch 8, seq_len 2048, bf16, gradient checkpointing off",
		"[preflight] requirements: already installed for this job",
	]
	loss = 2.91
	for step in range(1, 401):
		loss = max(0.6, loss * 0.9965 + (0.013 if step % 7 == 0 else -0.004))
		lines.append(
			f"step {step:4d} | loss {loss:.4f} | lr {min(2e-4, 2e-4 * step / 50):.2e} | grad_norm "
			f"{0.8 + (step % 13) * 0.07:.3f} | tokens/s {5120 + (step % 17) * 11} | mem 18.{step % 9} GiB"
		)
		if step % 97 == 0:
			lines.append(
				f"[eval] step {step}: eval_loss {loss + 0.11:.4f}, eval batch 32, no torch.no_grad()"
			)
	lines += [
		"Traceback (most recent call last):",
		'  File "train.py", line 212, in evaluate',
		"    out = model(**batch)",
		"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.12 GiB. GPU 0 has a total capacity of "
		"22.05 GiB of which 1.87 GiB is free. Of the allocated memory 19.21 GiB is allocated by PyTorch.",
	]
	system = (
		"You triage failed ML training probes for Preflight. Read the log and answer with JSON only: "
		'{"category": one of ["oom", "nan", "data", "install", "timeout", "other"], "line": the log line that '
		'shows it, copied exactly, "why": one sentence, "fix": one sentence}.'
	)
	return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


def _one_call(client, model: str, max_tokens: int) -> dict:
	t0 = time.monotonic()
	first, text, usage, finish = None, [], None, None
	stream = client.chat.completions.create(
		model=model,
		messages=triage_messages(),
		max_tokens=max_tokens,
		stream=True,
		stream_options={"include_usage": True},
		extra_body={"chat_template_kwargs": {"reasoning_strength": "low"}, "top_k": 64},
		**SAMPLING,
	)
	for chunk in stream:
		if chunk.usage:
			usage = chunk.usage
		for ch in chunk.choices or []:
			d = ch.delta
			thought = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None) or ""
			piece = (getattr(d, "content", None) or "") + thought
			if piece and first is None:
				first = time.monotonic()
			if getattr(d, "content", None):
				text.append(d.content)
			finish = ch.finish_reason or finish
	wall = time.monotonic() - t0
	out_toks = usage.completion_tokens if usage else None
	answer = "".join(text).strip()
	try:
		parsed = json.loads(answer.removeprefix("```json").removesuffix("```").strip())
	except ValueError:
		parsed = None
	decode_s = wall - (first - t0) if first else None
	return {
		"wall_s": round(wall, 2),
		"ttft_s": round(first - t0, 2) if first else None,
		"prompt_tokens": usage.prompt_tokens if usage else None,
		"completion_tokens": out_toks,
		"decode_tokens_per_s": round(out_toks / decode_s, 1) if out_toks and decode_s else None,
		"finish": finish,
		"answer_json": parsed is not None,
		"category": parsed.get("category") if isinstance(parsed, dict) else None,
	}


def _calls(base_url: str, model: str, calls: int, concurrent: int, max_tokens: int = 1536) -> dict:
	import concurrent.futures as cf

	from openai import OpenAI

	client = OpenAI(base_url=base_url, api_key=os.environ.get("GLIMMER_API_KEY", "local"), timeout=600)
	serial = [_one_call(client, model, max_tokens) for _ in range(calls)]
	t0 = time.monotonic()
	with cf.ThreadPoolExecutor(concurrent) as ex:
		together = list(ex.map(lambda _: _one_call(client, model, max_tokens), range(concurrent)))
	both = time.monotonic() - t0
	toks = sum(c["completion_tokens"] or 0 for c in together)
	return {
		"serial": serial,
		"concurrent": {
			"n": concurrent,
			"wall_s": round(both, 2),
			"tokens_per_s": round(toks / both, 1),
			"calls": together,
		},
	}


def _wait_ready(url: str, proc: subprocess.Popen, limit_s: float) -> float:
	import urllib.request

	t0 = time.monotonic()
	while time.monotonic() - t0 < limit_s:
		if proc.poll() is not None:
			raise RuntimeError(f"the server exited with {proc.returncode} before it was ready")
		try:
			with urllib.request.urlopen(url, timeout=5) as r:
				if r.status == 200:
					return time.monotonic() - t0
		except OSError:
			pass
		time.sleep(1)
	raise TimeoutError(f"not ready after {limit_s:.0f} s")


def _gpu_used() -> str:
	out = subprocess.run(
		["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
		capture_output=True,
		text=True,
	)
	return out.stdout.strip()


def _tail(path: str, keep: tuple[str, ...]) -> list[str]:
	try:
		lines = Path(path).read_text(errors="replace").splitlines()
	except OSError:
		return []
	return [ln[-300:] for ln in lines if any(k in ln for k in keep)][-40:]


def vllm_command(model_dir: str, api_key: bool) -> list[str]:
	cmd = [
		"vllm", "serve", model_dir,
		"--served-model-name", MODEL_ID,
		"--host", "0.0.0.0", "--port", str(PORT),
		"--max-model-len", str(MAX_MODEL_LEN),
		"--gpu-memory-utilization", "0.92",
		"--generation-config", "auto",
		"--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer",
		"--limit-mm-per-prompt", '{"image": 0, "video": 0}',  # text-only triage: no vision memory reserved
	]  # fmt: skip
	if api_key:
		cmd += ["--api-key", os.environ["GLIMMER_API_KEY"]]
	return cmd


def llama_command() -> list[str]:
	return [
		"/app/llama-server", "-m", f"{GGUF_DIR}/{GGUF_FILE}", "-a", MODEL_ID,
		"-ngl", "99", "-c", str(MAX_MODEL_LEN), "-np", "1", "-fa", "on", "--jinja",
		"--temp", "1.0", "--top-p", "0.95", "--top-k", "64",
		"--host", "127.0.0.1", "--port", str(PORT),
	]  # fmt: skip


def run_smoke(kind: str, calls: int, concurrent: int) -> dict:
	started = time.monotonic()
	weights.reload()
	src = _sources()
	log = "/tmp/server.log"
	if kind == "vllm":
		if "fp8" not in src:
			raise RuntimeError("run quantize first: the FP8 copy isn't in the volume")
		cmd, health = vllm_command(FP8_DIR, api_key=False), f"http://127.0.0.1:{PORT}/health"
		keep = ("KV cache", "Loading weights took", "Model loading took", "model weights took", "CUDA graph",
			"Maximum concurrency", "Error", "error", "torch.compile")  # fmt: skip
	else:
		cmd, health = llama_command(), f"http://127.0.0.1:{PORT}/health"
		keep = ("build", "model size", "offloaded", "KV", "CUDA", "error", "loaded")
	with open(log, "w") as fh:
		proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
	try:
		load_s = _wait_ready(health, proc, 20 * 60)
		mem = _gpu_used()
		res = _calls(f"http://127.0.0.1:{PORT}/v1", MODEL_ID, calls, concurrent)
	finally:
		proc.terminate()
		try:
			proc.wait(30)
		except subprocess.TimeoutExpired:
			proc.kill()
	return {
		"kind": kind,
		"command": cmd,
		"sources": src,
		"server_ready_s": round(load_s, 1),
		"gpu_memory_after_load": mem,
		"server_log": _tail(log, keep),
		"container_s": round(time.monotonic() - started, 1),
		**res,
	}
