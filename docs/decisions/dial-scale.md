# Dial scales: one fixed scale per record and readout (C, Oct 27)

**Question.** The dials used a running maximum: the full scale grew to whatever the visitor had
flown through, so the same cost sat at a different needle angle after a detour. The review asks
for a fixed scale per record. Which fixed scale?

**Method.** `web/scripts/mc_dial_scale.py` (numpy, `uv run --with numpy python
web/scripts/mc_dial_scale.py`). For the dev record and the finetune and pretrain examples, for
every dial target with a grid (time.total, cost.total, throughput.tokens_per_s, memory.peak):
2,000 random walks of 40 single-knob steps from the record's own setting (what a visitor does),
reading each visited cell's upper bound. Seed 7.

| Fixed scale | Visits off the scale | Visits reading as empty (needle < 5%) | Steps that visibly move the needle (>= 2% of the arc) |
|---|---|---|---|
| A: linear to the largest hi in the grid | 0% | 26% | 54% |
| B: linear to the 95th percentile of hi | 2.5% (worst 7.6%) | 8.6% | 62% |
| C: linear to 2x the starting hi | 8.9% (worst 26%) | 1.6% | 65% |
| D: linear to 1.25x the largest hi one step away | 11% (worst 28%) | 0.7% | 66% |
| **E: log scale over the grid's range, at most 1,000x wide** | **0%** | **0%** | **68%** |

**Chosen: E.** The only scale with nothing off the dial and nothing reading as empty, and it
moves the needle on the most steps. A linear scale to the grid's maximum (A) leaves a quarter of
visits as an empty-looking dial, because time and cost span two orders of magnitude across a
grid.

**Exceptions.** A limit the visitor set keeps a linear scale to that limit (budget: "of US$60
budget"; wall-time cap or deadline), since "how much of my limit" is the question. Probabilities
stay linear 0 to 1. A record with estimates but no grid gets a log scale from lo / 4 to hi x 4
(`ESTIMATE_ONLY_SPAN`, a guess until B ships grids for that lens). Every dial's tooltip names
its scale ("Scale runs from 24 to 9.5k").

**Memory stack.** The bar answers "does it fit", so it stays linear, fixed per record at 1.25x
the largest capacity any GPU the record can choose (`STACK_HEADROOM`, a guess). Past that the
bar clips and says the peak runs off the scale.

**Caveats.** Three records, two of them hand-made examples. The script reruns on the first real
records from A's job store; if E's empty-or-off share rises above A's, revisit.
