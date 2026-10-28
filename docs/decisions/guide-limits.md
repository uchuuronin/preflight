# The live guide's limits, from real Muse Spark tours (C, Oct 27)

**Question.** The guide's timeout, output cap, round cap and spend caps were guesses. What do
whole tours on Muse Spark actually take?

**Method.** `api/scripts/guide_tour.py` runs the real `POST /v1/guide/turns` route (typed input,
signed state, ledger, guards, number check) inside a Modal container, the place the production
API calls Muse Spark from, through `remote-run.yml` so the key stays in Modal's `preflight-meta`
secret. It plays the visitor's browser with an on-grid Python stand-in for the guide's browser
tools (`api/scripts/tour_play.py`), applies every move the guide proposes, approves the first
spending tool once, and records every round: model latency seen from the API, input, output
and cached tokens, cost, tools called, what the visitor would read. Model `muse-spark-1.3`,
`reasoning.effort: low`, base `https://api.meta.ai/v1`, Responses format. Samples: lora-7b,
lora-qwen, pretrain, agent; 3 tours each.

Dollar figures use the guide's price table (`muse_spark.PRICE_*`: $1.25 in, $4.25 out, $0.15
cached per million tokens), cited from a third-party listing and **not yet confirmed** against
Meta's Usage page.

## Run 1 (old limits: 900 output tokens, no re-rank guard), 12 tours, 91 rounds

| | p50 | p90 | p95 | max |
|---|---|---|---|---|
| Round latency (s) | 1.67 | 2.82 | 3.29 | 3.99 |
| Output tokens | 379 | | 863 | 900 (cap) |
| Input tokens | 2,856 | | | 5,389 |

- 4 of 91 rounds hit the 900-token output cap; one came back with no text at all. Reasoning
  tokens count as output (the plain smoke call with a 16-token cap returned only reasoning).
- 2 of 12 tours ended at the 8-round cap: the model called `sensitivity` again and again on
  the same setting.
- 50 of 89 replies used an em dash, against the instructions.
- Per tour: mean $0.033, max $0.115; mean 7.6 rounds, max 21.

## Changes made from run 1

| Setting | Was | Now | Source |
|---|---|---|---|
| `muse_spark.TIMEOUT_S` | 30 s | 10 s | derived: max 4.0 s at 900 output tokens, scaled to the new cap (about 7 s), plus 40% |
| `MAX_OUTPUT_TOKENS` | 900 | 1,600 | derived: about 1.85x the measured p95 (863), so reasoning has room |
| `MAX_ROUNDS_PER_TURN` | 8 | 8 | measured: finished tours used at most 4 rounds between visitor actions |
| `Limits.usd_per_run` | 0.40 | 0.40 | derived: three of the costliest tours plus one worst-case reservation |
| `Limits.turns_per_visitor_day` | 40 | 64 | derived: three of the longest tours (21 rounds) |
| `Limits.turns_per_ip_day` | 160 | 256 | guess: four visitors behind one address |

Behaviour: the server refuses a second `sensitivity` on the same setting in one turn (with a hint
to say "nothing safe" and stop) and lets the model answer within the same request; dashes are
replaced in replies; the instructions ask for rounded numbers and one ranking per setting.

## Run 2 (new limits), 12 tours, 48 rounds

Latency p50 2.36 s, p95 4.41 s, max 5.10 s; output p95 977, max 1,037 (no round near the cap);
no loops, no dashes; mean $0.012 and max $0.054 per tour. But 2 of 3 lora-7b tours and 1
lora-qwen tour ended with a sentence announcing a move ("I'll tour the cost-saving knobs...")
and no tool call: a stall. The instruction "write nothing while you are still calling tools"
was ignored 30 times out of 48 and may have caused the stalls.

## Run 3: four versions of the instructions

See the table below (filled from `c-guide-prompts`).
