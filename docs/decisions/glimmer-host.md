# Glimmer private mode: where it runs, and what it costs (B's draft, with A; due Sun Nov 1)

Contract: private mode is "deferred until it has a host (not on Meta's API or Baseten); A and B propose a host, likely a Modal endpoint, with its cost". The plan's goal is triage that runs where the code is, so code never leaves the user's machine. This draft covers two places Glimmer can run and prices both. **A to add:** how triage calls it, whether the weights can be downloaded and under what licence, and the quantised checkpoint to use.

## What Glimmer needs

Glimmer is a 30B dense model with 128K context, text and tools, which fits one 24–32 GB GPU when quantised (Judging playbook). At 4 bits, with scales, the weights are about 17 GB, at roughly 0.55 bytes per parameter (guess). The KV cache needs the model's real shape. Until A reads it from the checkpoint, the numbers below assume a Qwen2.5-32B-like shape: 64 layers, 8 KV heads, head dim 128, fp8 KV cache.

## Two places to run it

1. **The user's own GPU (the real private mode).** `preflight triage --private` starts vLLM (or llama.cpp on a Mac) on the user's machine and points triage at it.
   - Nothing leaves the machine, and it costs us nothing.
   - It needs a 24 GB card or more. On a 24 GB card the KV cache holds about 21K tokens, so a triage of one 8K-token log fits, but the full 128K context doesn't.
   - The demo can show it but can't host it.
2. **A Modal endpoint (the hosted demo of private mode).** vLLM on one GPU, scaled to zero after 2 minutes idle, behind the same per-visitor quota and daily spend cap as the probes. Visitors' code goes to our Modal container, not to a model provider. That's private from Meta and Baseten, not from us, and the page must say so.

## Cost per triage on Modal

From B's serving model (`engine/serving.py`), **uncalibrated**: its efficiencies are guesses until A's vLLM probes land, so read these as ±35%. One triage is an 8K-token prompt and an 800-token answer. Simulated:

| GPU | Modal $/h | KV tokens | One triage, alone | $ per triage | Output tok/s at 0.2 triages/s |
| --- | --- | --- | --- | --- | --- |
| L4 | 0.80 | 21K | 74 s | $0.016 | 20 |
| A10 | 1.10 | 21K | 42 s | $0.013 | 34 |
| L40S | 1.95 | 184K | 27 s | $0.015 | 111 |
| A100-40GB | 2.10 | 149K | 18 s | $0.010 | 122 |
| A100-80GB | 2.50 | 442K | 15 s | $0.010 | 126 |

**Cold starts dominate a demo with sparse traffic.** Loading 17 GB from a Modal volume and starting vLLM takes on the order of 90 s (guess; A's first run will measure it). That is about $0.05 on an A100-40GB, more than five triages. Scaling to zero after 2 minutes idle keeps a quiet day near $0, and a busy hour under $2.50.

## Proposal

- **Hosted demo:** an A100-40GB on Modal, with an L40S fallback. It's the cheapest per triage, about 18 s per answer, and has room for 128K-token logs.
  - Scale to zero after 2 minutes.
  - At most 1 container, and at most 3 private triages per visitor.
  - A $3 daily cap, shared with the probe budget's cap.
- **Real private mode:** the CLI on the user's GPU, with the same prompt and tools, so the two answer alike.
- **Before either ships:** one paid smoke run, with Chiya's yes: a cold start plus 10 triages on an A100-40GB, under $0.50. It calibrates the serving model and measures the cold start.

## Open questions for A

- Can the weights be downloaded, and under what licence?
- Which quantised checkpoint (AWQ, GPTQ or GGUF) and which tokenizer?
- Is triage's prompt shape as assumed: about 8K tokens in, 800 out, with tools?
- Does Glimmer's tool calling work through vLLM's OpenAI-compatible server?
