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
