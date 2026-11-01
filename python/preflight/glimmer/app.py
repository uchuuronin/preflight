"""Muse Glimmer on Modal: the hosted endpoint (8-bit, A100-40GB) and the own-GPU proof (4-bit, L4).

Chiya, Nov 1: Glimmer is Apache 2.0 and published by Meta, so it's hosted. Sources, both Meta's own (`meta-models`
on Hugging Face; the revision each download resolved to is recorded in the weights volume and in README.md):

- `meta-models/Muse-Glimmer-30B`: the bf16 checkpoint. Meta publishes no 8-bit one, so `quantize` makes an FP8
  copy of it once (llm-compressor FP8_DYNAMIC, weights only, data-free), and the endpoint serves that copy with
  vLLM. Quantising at load would hold the bf16 model (about 60 GB) on the GPU first, which a 40 GB card can't.
- `meta-models/Muse-Glimmer-30B-GGUF`, file Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf: Meta's 4-bit build for a
  24 GB card, served by llama.cpp, as Meta's page recommends for local use.

Runs (agents start these only through remote-run.yml; every one has its own time limit):

    modal run --env main python/preflight/glimmer/ops.py::startup_check   # CPU: downloads, revisions, parsers
    modal run --env main python/preflight/glimmer/ops.py::quantize        # CPU: the FP8 copy, once
    modal run --env main python/preflight/glimmer/ops.py::smoke --gpu a100 --calls 5
    modal run --env main python/preflight/glimmer/ops.py::smoke --gpu l4 --calls 5
    modal deploy --env main python/preflight/glimmer/app.py               # the endpoint, scales to zero

Each `modal run` writes out/glimmer_<name>.json. The deployed endpoint is vLLM's OpenAI-compatible server behind
Modal's web URL, at most one container, scaled to zero after 2 minutes idle, and it asks for the key in the
`preflight-glimmer` secret (GLIMMER_API_KEY): an open URL would let anyone start an A100 on our bill.
"""

from __future__ import annotations

import subprocess

import modal

from preflight.glimmer.common import (
	APP_NAME,
	FP8_DIR,
	GPUS,
	IDLE_S,
	PORT,
	WEIGHTS,
	vllm_command,
	vllm_image,
	weights,
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------- the hosted endpoint (deploy)


@app.cls(
	image=vllm_image,
	gpu=GPUS["a100"],
	volumes={WEIGHTS: weights},
	secrets=[modal.Secret.from_name("preflight-glimmer", required_keys=["GLIMMER_API_KEY"])],
	scaledown_window=IDLE_S,
	min_containers=0,
	max_containers=1,
	timeout=60 * 60,
)
@modal.concurrent(max_inputs=16)
class Glimmer:
	@modal.web_server(port=PORT, startup_timeout=20 * 60)
	def serve(self):
		subprocess.Popen(vllm_command(FP8_DIR, api_key=True))
