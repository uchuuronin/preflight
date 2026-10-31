# First calibration table: corpus phase 1 on L4 (Oct 30)

**Data:** 426 grouped kernel rows on an L4, exported by `b-corpus-export` (remote-runs, GitHub run 41152543575).
- 10 rows are phase 0's smoke configs; 416 are phase 1's.
- All 426 came from the grouped configs. 416 ran, and 10 ran out of memory.
- There are 84 architecture groups: 81 complete groups of 5, two of 9 (two sampled groups landed on the same architecture), and one partial group of 3.
- This is more than the 340 configs phase 1's log printed, because the last wave's containers kept writing rows after the workflow cut the log off.

**Method:** `python -m preflight.corpus.calibrate`.
- Each group's three probes are fitted, and its two queries are scored, separately per GPU.
- Split conformal on |log(actual / predicted)|, grouped by architecture.
- A GPU and job type with fewer than 9 runs merges into its job type across GPUs.
- The A100 holdout has no rows yet.

**Result:**

```
memory.peak: leave-one-architecture-out coverage 148 of 163 (90.8%, Wilson 95% 85.4% to 94.3%), median half-width 0.148
time.step: leave-one-architecture-out coverage 148 of 163 (90.8%, Wilson 95% 85.4% to 94.3%), median half-width 0.282
P(OOM) reliability (predicted vs observed, Wilson 95% on each bin):
  [0.00, 0.05) n=162 predicted 0.01 observed 0.00 (Wilson 95% 0.00 to 0.02)
  [0.30, 0.70): 2 points, too few to draw
  [0.70, 1.00): 4 points, too few to draw
  reliable enough to show as a probability: False
```

| Cell (target, L4, lens) | n | Architecture groups | 90% half-width (log) | About |
|---|---|---|---|---|
| memory.peak, finetune (corpus LoRA) | 33 | 18 | 0.324 | +-38% |
| memory.peak, pretrain | 35 | 18 | 0.152 | +-16% |
| memory.peak, retrieval | 32 | 16 | 0.148 | +-16% |
| memory.peak, vision | 63 | 30 | 0.131 | +-14% |
| time.step, finetune | 33 | 18 | 0.424 | +-53% |
| time.step, pretrain | 35 | 18 | 0.307 | +-36% |
| time.step, retrieval | 32 | 16 | 0.248 | +-28% |
| time.step, vision | 63 | 30 | 0.282 | +-33% |

**Reading it:**
- **Coverage:** near 90% is what split conformal gives by construction. The informative numbers are the widths, which are measured on real runs.
- **P(OOM):** it stays in bins. Only 6 queries fell above 30%, too few to draw, so the reliability check can't pass yet.

**Not installed in the engine yet**, because the rule says calibration data must match what users send. These rows are the corpus's synthetic architectures. Users send real Hugging Face models, and the closest match is phase 0's 50 real HF LoRA and QLoRA jobs, which sit in the `hf` store and aren't exported yet. The table is kept as data in `python/preflight/corpus/data/calibration-l4-phase1.json`. Installing it means writing `python/preflight/engine/_calibration_table.py` with `calibrate --out`, and that is Chiya's call.

## Update, Oct 30 evening: HF jobs in, grouped by architecture, memory installed

**Split check (Chiya's question):** the coverage was always leave-one-group-out, never a row split. But a corpus group ID hashes everything except batch and sequence, so 7 of the 73 architectures spanned more than one group, differing only in dtype or attention. Regrouping strictly by architecture (`calibrate.arch_key`: the fields that shape the model, nothing about how it runs) gave the same 148 of 163 for both targets. The calibrator now groups by architecture.

**Real HF jobs:** 50 rows came back from `b-corpus-export-hf` (49 ran, 1 out of memory), covering 5 models, each as bf16 LoRA and 4-bit QLoRA. That gives 19 scored queries per target.
- **The mix:** by Chiya's rule, calibration must match what users send, so the corpus's synthetic LoRA runs became their own job type (`finetune_synthetic`). User fine-tunes are now calibrated on real HF models only.
- **Scoring:** each HF architecture was scored against a table built without it.

| HF fine-tunes, L4, leave-one-architecture-out | Inside the 90% band | Wilson 95% | Half-width (log) |
|---|---|---|---|
| memory.peak | 18 of 19 (94.7%) | 75.4% to 99.1% | 0.067 to 0.084 (about +-7%) |
| time.step | 18 of 19 (94.7%) | 75.4% to 99.1% | 0.558 to 0.575 (about +-75%) |

**All rows together:** 166 of 182 for both targets (91.2%, Wilson 86.2% to 94.5%).

**Installed: memory cells only** (`--targets memory.peak`). Step time keeps its simulated band and says so. The real HF band, +-0.56, is past the engine's +-50% abstain limit, so installing it would abstain step time on every L4 fine-tune. The cause is a model flaw, not noise:
- **Above the probes** (8x512), the line under-predicts by 7% to 44%. The small, latency-bound configs pull it flat, while scaling the largest probe in proportion lands within about 5% to 12%.
- **Below the probes** (2x256), small models sit on a latency floor and time doesn't fall.

The fix is a step-time model with a floor below the probes and proportional growth above. It can't be validated on these same 19 queries, so phase 2's A10 HF jobs are its held-out test.

**P(OOM):** still shown in bins. Only 7 points fall above 30%, too few to draw, so the reliability check can't pass yet.
