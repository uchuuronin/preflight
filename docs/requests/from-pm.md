# Requests from the PM chat and Chiya (decisions)

Only the PM chat edits this file. Each line is something this agent needs from another (or a reply to one). The receiving agent answers in its own file with a REPLY line naming the request; the asker moves the line to Done.

`- TODO(contract) [PM -> X] what is needed, and why. (date)`

## Open

- DECIDED [Chiya -> all] (Oct 27) Agent sample size: 30 tasks with the 80% runaway bound; the guide offers 20 more through `sample_more` (needs the visitor's click) when the cost range straddles the budget.
- DECIDED [Chiya -> all] (Oct 27) Below the smallest measured batch: an extrapolated estimate with a wider range, labelled, plus a "run a probe here" offer; far outside the measured range still abstains.
- DECIDED [Chiya -> all] (Oct 27) Licence Apache-2.0; A adds the LICENSE file, C adds a CI step failing on GPL, AGPL or SSPL dependencies (gap 22).
- DECIDED [Chiya -> all] (Oct 27) Modal: $100 hackathon credits plus the $30 monthly free tier, split in docs/setup/cloud.md. Baseten $100 and Meta $150 credits are live.
- TODO(contract) [PM -> A] Read the Meta base URL from `META_API_BASE` (C's name, now canonical) in `harness/spark.py`; keep `META_BASE_URL` as a fallback for one release. (Oct 27)
- TODO(contract) [PM -> B] Re-plan the NVIDIA corpus slice to $45 of Modal (T4, L4, A10, L40S; a few A100 rows only for the cross-GPU holdout): grouped probe-like configs and real HF LoRA and 4-bit jobs first, no H100 rows. The earlier $150 to $245 plan doesn't fit. (Oct 27)
- TODO(contract) [PM -> all] Cloud runs go through `remote-run.yml` and `amd-run.yml` only (docs/setup/cloud.md). Record each run's GPU, minutes and dollars in your status file. (Oct 27)
- DECIDED [Chiya -> all] Total cost readout (Oct 26): B owns one dated, sourced USD price table; the user picks the provider; spot = spot rate plus expected lost work from B's checkpoint-interval model. On the user's own cluster: GPU-hours first, "market equivalent: $X on-demand" in grey. The total is GPU plus LLM/API tokens for the job; Preflight's own probe cost shows separately ("this plan cost $0.42 to make"). Failure risk stays out of the total as "+ $X at risk" = P(failure) x cost up to the crash. D3 adds regional rows and converts currency at display time with a dated exchange rate. (Oct 27)

## Done


