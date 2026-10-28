# Requests from the PM chat and Chiya (decisions)

Only the PM chat edits this file. Each line is something this agent needs from another (or a reply to one). The receiving agent answers in its own file with a REPLY line naming the request; the asker moves the line to Done.

`- TODO(contract) [PM -> X] what is needed, and why. (date)`

## Open

- MEASURED [PM -> all] (Oct 28, run 40614430138, runs/20261028-012026-40614430138-pm-smoke-models on remote-runs) Model APIs, 20 tiny requests each, all HTTP 200:
  - Baseten `meta-models/muse-spark-open-1026` (chat completions): latency p50 0.56 s, p95 0.68 s, max 0.78 s.
  - Meta `muse-spark-1.3`, `/v1/responses`: p50 1.78 s, p95 3.58 s, max 5.81 s, with about 516 reasoning tokens even for a one-word answer; `/v1/chat/completions` also works (3 of 3).
  - Reasoning tokens are counted inside output tokens on both providers (Baseten completion 33 incl. 22 reasoning; Meta output 526 incl. 516). This answers B's question to A on `AgentTask.output_tokens`.
  - Meta's model list: muse-spark-1.1, 1.2, 1.3 (each with a `-contributor` variant), muse-wonder-1.0, muse-image-1.0, muse-voice-transcribe-1.0, sam-3.1. No Muse Code and no Muse Glimmer model id: Muse Code is the terminal agent (installed with Meta's script), and Glimmer needs its own host.
- TODO(contract) [PM -> C] Set the guide's Meta timeout and token caps from measured guide turns, not the smoke test's tiny prompt; the floor is about 500 reasoning tokens a call (about $0.002 at A's price table). (Oct 28)
- TODO(contract) [PM -> A, B] Where does Muse Glimmer run for the private mode? It isn't on the Meta API or Baseten lists; the Modal credits cover serving Meta open-weight models on Modal Endpoints. Propose a host and its cost before G2. (Oct 28)
- TODO(contract) [PM -> C] Muse Code isn't an API model: the fix loop runs the Muse Code terminal agent inside a sandbox with `META_API_KEY`. Confirm how it's installed and authenticated non-interactively there. (Oct 28)
- DECIDED [Chiya -> all] (Oct 27) Agent sample size: 30 tasks with the 80% runaway bound; the guide offers 20 more through `sample_more` (needs the visitor's click) when the cost range straddles the budget.
- DECIDED [Chiya -> all] (Oct 27) Below the smallest measured batch: an extrapolated estimate with a wider range, labelled, plus a "run a probe here" offer; far outside the measured range still abstains.
- DECIDED [Chiya -> all] (Oct 27) Licence Apache-2.0; A adds the LICENSE file, C adds a CI step failing on GPL, AGPL or SSPL dependencies (gap 22).
- DECIDED [Chiya -> all] (Oct 27) Modal: $100 hackathon credits plus the $30 monthly free tier, split in docs/setup/cloud.md. Baseten $100 and Meta $150 credits are live.
- TODO(contract) [PM -> A] Read the Meta base URL from `META_API_BASE` (C's name, now canonical) in `harness/spark.py`; keep `META_BASE_URL` as a fallback for one release. (Oct 27)
- TODO(contract) [PM -> B] Re-plan the NVIDIA corpus slice to $45 of Modal (T4, L4, A10, L40S; a few A100 rows only for the cross-GPU holdout): grouped probe-like configs and real HF LoRA and 4-bit jobs first, no H100 rows. The earlier $150 to $245 plan doesn't fit. (Oct 27)
- TODO(contract) [PM -> all] Cloud runs go through `remote-run.yml` and `amd-run.yml` only (docs/setup/cloud.md). Record each run's GPU, minutes and dollars in your status file. (Oct 27)
- DECIDED [Chiya -> all] Total cost readout (Oct 26): B owns one dated, sourced USD price table; the user picks the provider; spot = spot rate plus expected lost work from B's checkpoint-interval model. On the user's own cluster: GPU-hours first, "market equivalent: $X on-demand" in grey. The total is GPU plus LLM/API tokens for the job; Preflight's own probe cost shows separately ("this plan cost $0.42 to make"). Failure risk stays out of the total as "+ $X at risk" = P(failure) x cost up to the crash. D3 adds regional rows and converts currency at display time with a dated exchange rate. (Oct 27)

## Done


