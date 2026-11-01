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
- **One smoke run, with Chiya's yes, measures both:** a cold start plus 5 triages on an L4 and the same on an A100-40GB. That's about $0.20 by the table above, capped at $0.50. It calibrates the serving model on Glimmer and measures the cold start, which is a guess today (about 90 s).

**What blocks shipping either path:** whether Glimmer's weights can be downloaded, and under what licence. Neither the playbook nor the plan says so. If they can't be downloaded, private mode can't ship as planned, and the readout should say so instead of offering it.

## Open questions for A

- Can the weights be downloaded, and under what licence?
- Which quantised checkpoint (AWQ, GPTQ or GGUF) and which tokenizer?
- Is triage's prompt shape as assumed: about 8K tokens in, 800 out, with tools?
- Does Glimmer's tool calling work through vLLM's OpenAI-compatible server?
