# Glimmer private mode: where it runs, and what it costs (B with A; due Sun Nov 1)

Contract: private mode is "deferred until it has a host (not on Meta's API or Baseten); A and B propose a host, likely a Modal endpoint, with its cost". The plan's goal is triage that runs where the code is, so code never leaves the user's machine. This covers two places Glimmer can run, prices both, and settles the host with A's proposal. Still open: whether the weights can be downloaded and under what licence, and the quantised checkpoint to use.

## What Glimmer needs

Glimmer is a 30B dense model with 128K context, text and tools, which fits one 24–32 GB GPU when quantised (Judging playbook). At 4 bits, with scales, the weights are about 17 GB, at roughly 0.55 bytes per parameter (guess). The KV cache needs the model's real shape. Until A reads it from the checkpoint, the numbers below assume a Qwen2.5-32B-like shape: 64 layers, 8 KV heads, head dim 128, fp8 KV cache.

## Two places to run it

1. **The user's own GPU (the real private mode).** `preflight triage --private` starts vLLM (or llama.cpp on a Mac) on the user's machine and points triage at it.
   - Nothing leaves the machine, and it costs us nothing.
   - It needs a 24 GB card or more. On a 24 GB card the KV cache holds about 21K tokens, so a triage of one 8K-token log fits, but the full 128K context doesn't.
   - The demo can show it but can't host it.
2. **A Modal endpoint (the hosted demo of private mode).** vLLM on one GPU, scaled to zero after 2 minutes idle, behind the same per-visitor quota and daily spend cap as the probes. Visitors' code goes to our Modal container, not to a model provider. That's private from Meta and Baseten, not from us, and the page must say so.

## Cost per triage on Modal

From B's serving model (`engine/serving.py`), **uncalibrated**: its efficiencies are guesses until A's vLLM probes land, so the model's own uncalibrated band is ×0.61 to ×1.64. One triage is an 8K-token prompt and an 800-token answer. Simulated:

| GPU | Modal $/h | KV tokens | One triage, alone | $ per triage |
| --- | --- | --- | --- | --- |
| L4 | 0.80 | 21K | 74 s | $0.016 |
| A10 | 1.10 | 21K | 42 s | $0.013 |
| L40S | 1.95 | 184K | 27 s | $0.015 |
| A100-40GB | 2.10 | 149K | 18 s | $0.010 |
| A100-80GB | 2.50 | 442K | 15 s | $0.010 |

None of them keeps up with one triage every 5 seconds (160 output tokens per second offered). Under that load the A100s drain at about 120 tokens per second, roughly one triage every 7 seconds. So one container serves a demo's trickle, and a rush queues: the per-visitor cap matters more than the GPU choice.

**Cold starts dominate a demo with sparse traffic.** Loading 17 GB from a Modal volume and starting vLLM takes on the order of 90 s (guess; A's first run will measure it). That is about $0.05 on an A100-40GB, more than five triages. Scaling to zero after 2 minutes idle keeps a quiet day near $0, and a busy hour under $2.50.

## Proposal (settled with A's reply, Oct 31)

A proposed a Modal L4 at 4 bits behind vLLM's OpenAI server, scaled to zero (`docs/requests/from-a.md`, Oct 31): it's the GPU a student has, so private mode and the hosted demo run on the same thing, and it plugs into the agent proxy as an OpenAI-compatible upstream with no new code. B's table agrees on cost (an L4 triage is about $0.016, an A100-40GB one $0.010, both simulated and uncalibrated) and differs on speed: about 74 s an answer on the L4 against 18 s on the A100-40GB, and the L4's KV cache holds about 21K tokens, not 128K. Both are right about different things, so:

- **Private mode targets the L4-class card.** The CLI runs Glimmer at 4 bits on the user's own 24 GB GPU, with the same prompt and tools as the hosted path. Its limits go on the page: about 21K tokens of context and about a minute an answer.
- **The hosted demo runs on an A100-40GB**, with an L40S fallback. A stranger shouldn't wait over a minute, and long logs fit.
  - Scale to zero after 2 minutes.
  - At most 1 container, and at most 3 private triages per visitor.
  - A $3 daily cap, shared with the probe budget's cap.
  - The page says plainly that visitors' code reaches our Modal container, not Meta or Baseten.
- **One smoke run measures both, and only once the weights' terms are confirmed** (Chiya, Nov 1: no test run until the download terms and the licence are confirmed in writing, with the link here): a cold start plus 5 triages on an L4 and the same on an A100-40GB. That's about $0.20 by the table above, capped at $0.50. It calibrates the serving model on Glimmer and measures the cold start, which is a guess today (about 90 s).

**Settled (Chiya, Nov 1, 11:18 AM ET): hosted.** Glimmer is Apache 2.0 and published by Meta, so the licence hold is dropped.
- **Licence:** Apache 2.0 ([Meta's model page](https://dev.meta.ai/docs/muse-glimmer); [Meta's Hugging Face release post](https://huggingface.co/blog/muse-glimmer)).
- **Download:** Meta's own repos on Hugging Face, org `meta-models`: `meta-models/Muse-Glimmer-30B` (bf16, about 59.6 GB) and `meta-models/Muse-Glimmer-30B-GGUF`. Revisions served (startup check, Nov 1, GitHub run 41327585942): `meta-models/Muse-Glimmer-30B` at `cec94fcfd6512f2cfe708138c32533b346ea4f69` (59.58 GB, `MuseGlimmerForConditionalGeneration`: 52 layers, 32 query and 2 KV heads of 128, 2,048-token sliding window, 131,072 context) and `meta-models/Muse-Glimmer-30B-GGUF` at `b44f1072cf75e918a1b82a3a17b45c4389b076e7` (the Q4_K_M file, 16.76 GB). Both report Apache 2.0 and are not gated. llama.cpp: build 12030, commit a91eab67d.
- **Endpoint (8-bit, A100-40GB):** Meta publishes no 8-bit checkpoint; the FP8 builds on the Hub are third-party. So B makes one FP8 copy of Meta's bf16 checkpoint (llm-compressor FP8_DYNAMIC, data-free, so it depends on nothing but Meta's weights), and vLLM 0.28.0 serves it ([vLLM's Glimmer recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B) for the version and parsers). Quantising at load isn't possible on 40 GB: vLLM loads the bf16 model first.
- **Own GPU (4-bit, L4):** Meta's `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (16.8 GB, "fits 24 GB VRAM") on llama.cpp, which is how Meta's page runs it locally. A smoke run proves it; it isn't hosted.
- **Access:** Modal proxy auth on the endpoint, so a request without a proxy token never starts the A100.
- **Code:** `python/preflight/glimmer/` (`app.py` the endpoint, `ops.py` the runs).

## Measured (Nov 1 smoke runs)

One triage is the fixed prompt in `glimmer/common.py`: 17,138 tokens in (the earlier 8K guess was characters over 4, not tokens) and about 300 out, reasoning low. Every answer in both runs was valid JSON with the right category (oom), 14 of 14 (5 alone and 4 at once on the A100, 5 on the L4).

| | A100-40GB, FP8, vLLM 0.28.0 (the endpoint) | L4, Q4_K_M, llama.cpp b12030 (own-GPU proof) |
| --- | --- | --- |
| GPU memory used after load | 36.0 of 40 GiB (36,821 MiB; 32K context) | 15.8 of 22.5 GiB (16,202 MiB; 32K context) |
| Cold start, asked for to ready | 286 s (load 34 GB from the volume, CUDA graphs) | 18 s for the server, but the L4 took 19 min to schedule |
| Time to first token | 4.6 s (about 3,700 prompt tokens/s) | 22.3 s (about 770 prompt tokens/s) |
| Decoding, one call | 38 tokens/s | 14.4 tokens/s |
| One triage, alone | 12.8 s (median of 5) | 43.7 s (median of 5) |
| Four at once | 27.7 s for 4 (6.9 s each, 47 output tokens/s together) | not measured (one slot) |
| Cost per triage, warm | $0.0075 alone, $0.004 with 4 in flight ($2.10/h) | $0.0097 ($0.80/h) |
| A call after scaling to zero adds | $0.237 (286 s cold start plus the 120 s idle tail) | |

Against the earlier simulated table (A100-40GB: 18 s and $0.010 a triage, about 90 s cold start): the warm triage is faster than simulated, and the cold start is about three times the guess. On a quiet demo the cold start dominates, as predicted: a lone triage after the endpoint has scaled to zero costs about $0.24, almost all of it the start. Price rows: `engine/prices` `SELF_HOSTED_RATES` (GPU-seconds per call, from these runs).

## Open questions for A

- Which quantised checkpoint (AWQ, GPTQ or GGUF) and which tokenizer?
- Is triage's prompt shape as assumed: about 8K tokens in, 800 out, with tools?
- Does Glimmer's tool calling work through vLLM's OpenAI-compatible server?
