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

## Update, after review: memory errs high, synthetic LoRA out of the table

**Memory queries are sized by seq_len again.**
- **Why:** a run's memory peak comes from the longest row in the whole dataset (or seq_len if rows reach it), and a few probe batches don't see that row. Sizing a query by the rows the probes saw could say "fits" for a run that runs out of memory later.
- **How it works now:** probes are fitted with the longest padded row they processed, warm-up included. Queries keep the setting, so on data shorter than the limit the engine over-predicts and never under-predicts.
- **Held out on A's L4 smoke record:** p3 is within 0.5%. p1 and p2 are 76% and 61% high, because their rows reached 517 of 1024.
- **What would make it tight and still safe:** the dataset's own longest tokenized row, which the probe could record.

**Step time:** a probe cut by its seq_len now gives a lower bound on row length at its batch, instead of being dropped. Held out on the same record: 2 of 3 within 15% (Wilson 95% 21% to 94%), 3 of 3 inside the 90% interval (44% to 100%).

**Installed table:** memory cells only, with the corpus's synthetic LoRA left out entirely (66 scores), so it can't reach a card or a pooled cell. Leave-one-architecture-out coverage is 136 of 149 (91.3%, Wilson 95% 85.6% to 94.8%).

| Cell | n | 90% half-width (log) |
|---|---|---|
| finetune (real HF) | 19 | 0.067 |
| pretrain | 35 | 0.152 |
| retrieval | 32 | 0.148 |
| vision | 63 | 0.131 |
| all job types, pooled | 149 | 0.127 |

## Latency-floor step time: fitted, then failed its held-out test (Oct 31)

`steptime.EXTRAPOLATION = "anchored"` scales the nearest probe's measured phases by a power of the work ratio outside the probed sizes. The powers were fitted on the 19 L4 HF queries only: about 1.1 above the probes, and 0.5 with a 0-to-1 band below. `scripts/eval_steptime_extrapolation.py` scores it against the fitted line.

| Queries | Line (shipped) | Anchored |
|---|---|---|
| Design set, L4 HF, above the probes | 6 of 9 within 15% | 9 of 9 |
| Design set, L4 HF, below | 6 of 10 | 1 of 10 (10 of 10 inside the band) |
| Held out, grouped corpus on L4, above | 62 of 77 (Wilson 95% 70% to 88%) | 30 of 77 (29% to 50%) |
| Held out, grouped corpus on L4, below | 57 of 82 (59% to 78%) | 18 of 82 (14% to 32%) |

It fits the jobs it was tuned on and is worse everywhere else, so it stays off and the line remains the model. Whether the HF pattern holds on other HF jobs is the A10 HF test, which needs those rows exported. The HF step-time band stays uninstalled meanwhile.

## Phase 2 rows in (Oct 31): four GPUs, memory reinstalled, A100 holdout still to run

Rows: `remote-runs` exports `b-corpus-export-2` and `b-corpus-export-hf-2`. Grouped: L4 577, A10 577, L40S 195, T4 75. HF: 50 each on L4 and A10. The 214 synthetic LoRA scores stay out of the table, as before. That leaves 487 scores per target. The two A100 rows are withheld from everything.

**Memory, installed** (`engine/_calibration_table.py`, 17 cells). Leave-one-architecture-out coverage is 439 of 487 (90.1%, Wilson 95% 87.2% to 92.5%), median half-width 0.134 (measured):

| Lens | Inside | Median half-width |
| --- | --- | --- |
| finetune (real HF) | 36 of 38 | 0.084 |
| pretrain | 105 of 117 | 0.134 |
| retrieval | 105 of 115 | 0.148 |
| vision | 193 of 217 | 0.127 |

L4 and A10 now have their own cells for every lens. L40S has its own for pretrain, retrieval and vision. T4 has its own vision cell (n = 10, half-width 0.565, roughly ×0.57 to ×1.76); its pretrain and retrieval cells are under 9 points, so those use the pooled cells.

**Step time, not installed: a decision for Chiya.** Coverage is also 439 of 487 (the same count by coincidence; the per-lens counts differ). The bands are wider: median half-width 0.236. By lens:

| Lens | Inside | Median half-width |
| --- | --- | --- |
| finetune (real HF) | 35 of 38 | 0.558 |
| pretrain | 106 of 117 | 0.267 |
| retrieval | 106 of 115 | 0.210 |
| vision | 192 of 217 | 0.172 |

The HF finetune band stays off, per the earlier decision. The other three are honest at 90%, but a ±0.17 to ±0.27 log band means roughly ×0.8 to ×1.3.

**P(OOM) now passes the reliability rule** (at least 2 bins with at least 10 points, and at least 10 real OOMs: 18 in the scored pairs, from 29 OOM rows). By bin:

| Bin | Points | Predicted | Observed (Wilson 95%) |
| --- | --- | --- | --- |
| [0, 0.05) | 587 | 0.01 | 0.00 (0.00 to 0.01) |
| [0.70, 1] | 14 | 0.97 | 0.93 (0.69 to 0.99) |
| [0.05, 0.30) | 5 | | too few to draw |
| [0.30, 0.70) | 6 | | too few to draw |

So the table sets `oom_reliable`. The middle bins are still thin: the probability is checked only near 0 and near 1.

**A100 holdout: not scored.** The rerun asked Modal for an A100-80GB and got an A100-PCIe-80GB. Its rows were filed under the board it ran on, so the sweep's ledger saw no progress and stopped after 2 configs, as its guard should (measured). Two fixes:
- The ledger now counts rows on a sibling board as done (`sweep.LANDS_ON`).
- By default, the calibrator holds out every A100 board, MIG slices included (`calibrate.held_out`). Passing `--holdout-gpus` overrides that. The step-time eval script drops those boards too, so the holdout never feeds a choice between models.

Two rows are not a test, and the holdout is scored once, so it waits for the rerun.

**Sensitivity tie.** With an A10 finetune cell, a bigger GPU now reaches the pooled cell, which has 38 runs. Its P(OOM) floor is 0.013, against 0.025 for 19 runs, so `gpu_type` outranked `eval_batch_size` for "make it fit" on resolution alone. For ranking, P(OOM) is now clamped to [0.05, 0.95], the floor and ceiling of the smallest allowed cell (derived: 0.5 / (9 + 1)). Other probabilities, such as a sampled runaway rate, are ranked as measured. Clamped, the two levers tie, and the tie goes to the one with the lower peak memory: `eval_batch_size`. A lever that only moves P(OOM) below 0.05 (the pretrain fixture, at about 1e-63) is no longer listed.

**Anchored step time on the A10 HF jobs (Oct 31): fails again, stays off.** This is the anchored model's held-out test on the same job type it was fitted on (the L4 HF jobs), but on another GPU. It scores the 19 A10 queries (measured):

| Queries | Model | Within 15% (Wilson 95%) | Median abs log error |
| --- | --- | --- | --- |
| above the largest probe (9) | line | 7 of 9 (45% to 94%) | 0.020 |
| above the largest probe (9) | anchored | 7 of 9 (45% to 94%) | 0.096 |
| below the smallest probe (10) | line | 8 of 10 (49% to 94%) | 0.064 |
| below the smallest probe (10) | anchored | 1 of 10 (2% to 40%) | 0.323 |

The fitted line does as well or better on both sides, so `EXTRAPOLATION` stays `"line"`. On every held-out group in phase 2 (A100 boards excluded), the line puts 461 of 575 queries within 15% (80%).

## Chiya's calls, Oct 31: step-time bands for three lenses, P(OOM) a number only where verified

**Step time, installed for vision, retrieval and pretrain** (`calibrate --lenses "time.step=pretrain+retrieval+vision"`). Leave-one-architecture-out coverage over those three lenses is 404 of 449 (90.0%, Wilson 95% 86.9% to 92.4%), median half-width 0.226 (measured). Fine-tune keeps its simulated band, and so does every other lens: the table has no all-lens step-time cell to borrow. The same-GPU model card adds "Calibrated on corpus models, not users' scripts; checked on unseen scripts Nov 2."

**P(OOM): a number only where the reliability diagram verifies it.** Below 5% and above 95%, it is shown as a number. In between, it is reported as the range 5–95% and worded "uncertain (5–95%)". It switches to a number by itself once every middle bin (5–30%, 30–70%, 70–95%) holds at least 10 points whose mean prediction sits inside the Wilson interval of what happened. The table now stores these bins (`oom_bins`), and `engine.oom.shown_as_number` reads them. Today's bins (measured):

| Bin | Points | Predicted | Observed (Wilson 95%) | Agrees |
| --- | --- | --- | --- | --- |
| 0–5% | 587 | 0.010 | 0.003 (0.001 to 0.012) | yes |
| 5–30% | 5 | 0.11 | 0.20 (0.04 to 0.62) | too few |
| 30–70% | 6 | 0.37 | 0.33 (0.10 to 0.70) | too few |
| 70–95% | 4 | 0.91 | 1.00 (0.51 to 1.00) | too few |
| 95–100% | 10 | 0.990 | 0.90 (0.60 to 0.98) | **no** |

**For Chiya: the top bin doesn't verify yet.** 9 of its 10 jobs ran out of memory, against a mean prediction of 0.99. The one miss had a predicted P of 0.9955. As you set it, the rule shows numbers above 95% anyway. With ten points, one miss is enough to fail a 0.99 prediction (10 of 10 would pass), so this bin needs more OOM runs to settle. The other choice is to cap what's shown there at "above 95%".

## A100 holdout, scored once (Oct 31)

`b-corpus-a100-3` ran the other 93 configs on the fixed code. Modal put the work on both boards: 42 rows on A100-PCIe-80GB and 53 on A100-SXM-80GB, with no OOMs, as expected on 80 GB. One group was split across the boards and couldn't be scored. The 16 synthetic LoRA scores stay out, as they do everywhere. That leaves 28 scored queries per target.

The method was fixed before looking: the calibrator as committed, the A100 never calibrated on, each query scored against cells built without its own architecture, and the installed lenses only. Results (measured, scored once, never tuned on):

| Target | A100-PCIe-80GB | A100-SXM-80GB | Both boards (Wilson 95%) |
| --- | --- | --- | --- |
| Memory peak | 8 of 12 | 14 of 16 | **22 of 28, 79% (60% to 90%)** |
| Step time, same GPU | 11 of 12 | 15 of 16 | **26 of 28, 93% (77% to 98%)** |

**Step time holds up on an unseen GPU. Memory under-covers:** the 95% interval tops out just under 90%. Of the six misses, two are underestimates, the dangerous direction: one actual peak is 40% above the prediction (a PCIe vision group), the other 16% above (a PCIe pretrain group). The other four are overestimates, two of them extreme: SXM vision groups at 3.4× and 6.9× below the prediction. These are diagnoses, not fixes. Nothing is tuned on these rows. The card reports the per-board figures (`per_gpu_coverage`) beside the leave-one-architecture-out coverage. The fix has to be checked on data other than the A100: the AMD rows, and the unseen scripts on Nov 2.

## Cross-GPU step time on the corpus: K1 alone, the GBDT residual, and K1's band (Oct 31)

`scripts/fit_steptime_gbdt.py` builds every case where two non-A100 GPUs ran the same config: 2,286 cases over 566 configs, on A10, L4, L40S and T4. Each case replays the source GPU's op table onto the target with K1 and compares the result with the target's measured step. Scoring is leave-one-GPU-out: for each GPU, train on the pairs that don't touch it, and test on the pairs that target it.

**The corpus op table counts device time three times.** The worker keeps profiler ranges, aten ops and the CUDA kernels under them, each with its own self device time. Together they sum to 2.6× the step (median, L4 rows), which made K1 look 2–3× off. The script uses the aten ops alone, which carry the FLOPs and byte estimates and cover about 0.9 of the step. The worker's table should keep one view; that is a corpus fix for the next sweep.

**GBDT residual: fails held out, not installed.** Its hyperparameters were fixed before any test GPU was scored (200 trees, depth 3, rate 0.05). Within 20% of the measured step (measured):

| Target GPU | K1 | K1 + GBDT |
| --- | --- | --- |
| A10 (830) | 522, 63% | 428, 52% |
| L4 (830) | 503, 61% | 395, 48% |
| L40S (435) | 245, 56% | 212, 49% |
| T4 (191) | 112, 59% | 129, 68% |
| All, Wilson 95% | **1,382 of 2,286, 60% (58% to 62%)** | 1,164, 51% (49% to 53%) |

The residual only helps on T4. Overall it is worse, so `engine/steptime_gbdt.py` stays unused: `available()` is False until a generated file shows it beating K1 held out. By the plan's rule, the GNN would have to beat the GBDT; the more useful bar now is beating K1.

**K1's band is overconfident on the corpus: a decision for Chiya.** The band is σ = 0.12, sized by simulation (S4c), and its 90% band covers only 1,352 of 2,286 cases (59%, Wilson 57% to 61%). A split-conformal half-width fitted on the other GPUs' pairs covers 2,130 of 2,286 held out (93%, Wilson 92% to 94%). The pooled half-width is 0.45, roughly ×0.64 to ×1.57.

| Held-out GPU | Half-width from the others | Covered |
| --- | --- | --- |
| A10 | 0.59 | 98% |
| L4 | 0.49 | 94% |
| L40S | 0.44 | 87% |
| T4 | 0.43 | 84% |

Those Wilson intervals treat the 2,286 pairs as independent. They aren't: they come from 566 configs and 95 architectures. A bootstrap clustered by architecture gives the honest intervals: the conformal band covers 93% (95% interval 90% to 96%), and the simulated band 59% (53% to 67%). The proposal is to install it as the cross-GPU band, behind `engine.steptime_cross_gpu` (already off). These are the corpus's aten ops, not A's kernel tables, so it also gets checked on the AMD rows and the Nov 2 scripts.

## GNN cost model against the GBDT, on held-out architectures (Oct 31)

`scripts/fit_cost_gnn.py` predicts log(step) with no probe at all, from the module graph, the config and the GPU: what the GPU dial needs before anything has run. It uses 1,203 non-A100 rows over 79 architectures, in five folds by architecture, so no architecture is ever trained and tested on. Everything was fixed before any fold was scored: the hyperparameters, the 0.5/0.5 ensemble weight, and the shorter training needed to finish in time. Within 20% of the measured step (measured):

| Model | Within 20% (Wilson 95%) | Median abs log error |
| --- | --- | --- |
| GBDT on config, GPU and graph counts | 491 of 1,203, 41% (38% to 44%) | 0.251 |
| GNN (GraphSAGE over the module graph) | 422 of 1,203, 35% (32% to 38%) | 0.310 |
| Mean of the two | 549 of 1,203, 46% (43% to 48%) | 0.229 |

The GNN alone doesn't beat the GBDT. The plan's validation row is "GNN + GBDT ensemble vs GBDT alone", though, and the ensemble does beat the GBDT, by 58 rows. These Wilson intervals treat rows as independent and they overlap, so the difference needs an architecture-clustered check before the ensemble is used.

Either way, a no-probe cost model is a prior, not a prediction. At 41–46% within 20%, it is well behind K1 with a probe (60%) and the same-GPU line (80%).

**Installed (Chiya, Oct 31): the cross-GPU band, `time.step.cross`.** `calibrate --cross` builds it from the 2,286 corpus pairs, and the engine uses it in K1's replay. K1's band is never narrower than this: pooled half-width 0.448 (2,286 pairs, 95 architectures), with per-GPU cells for the four corpus GPUs. The card reports leave-one-GPU-out coverage, 2,130 of 2,286.

**Consequence for the GPU dial: every GPU move now abstains.** The plan abstains when a band is wider than ±50%. K1's honest band is ±45% before the probe's own error is added, and ±50% after. So moving to another GPU now says "run a probe here" instead of showing a range. That is the truth about the corpus, but it changes what visitors see. Chiya's options:
- keep it;
- show cross-GPU time as a wide range labelled "rough, no probe on this GPU", past the ±50% rule;
- or wait for a better replay. The corpus fed K1 aten-op tables, while users' probes carry kernel tables, so the AMD rows and the Nov 2 scripts may tell a different story.

The flag `engine.steptime_cross_gpu` stays off either way.

## The A100 memory underestimates: what non-A100 data says (Oct 31)

Chiya asked for the cause of the two A100 underestimates (40% and 16% low), found on non-A100 data only. `scripts/diagnose_memory_misses.py` splits every held-out memory miss on L4, A10, L40S and T4 by what could grow with the card. Measured, on 487 queries:

- **The misses don't grow with the card.** The same config misses by the same amount on 24 GB and 48 GB cards: a CNN query is 2.0× under on both the L4 and the L40S, and an encoder query 1.32× under on A10, L4 and L40S. The median miss is 0.000 on every GPU.
- **Not the allocator's reserved-to-allocated gap.** The miss on allocated memory alone is the same size as on reserved (+0.81 against +0.71 for the worst). The correlation between the miss and how much the gap moved is 0.19.
- **Not fixed workspaces or the CUDA context.** The underestimates have a smaller fixed share of reserved memory than the queries overall (0.22 against 0.52).
- **It's the config, and where the query sits.** Underestimates by more than 10% hit 14% of queries below the smallest probe, against 5% above it. They come from CNNs (17%), GPTs (13%) and encoders (11%); MLPs and the HF LoRA jobs have none. Below the probes, the line through them drops faster than memory does: the real curve has a floor the line can't see.

**Fix (memory-analytical@0.3.1):** below the smallest probe, the top of the range is at least what the line gives at that probe, because memory doesn't shrink faster than the work does. On non-A100 rows, below-probe queries above the band's top fall from 11 of 236 to 2. The point estimate is unchanged. Above the probes, 4 of 251 still exceed the top. The worst is an encoder going to a longer sequence, which is config-specific.

**What this can't say:** whether the two A100 misses were below-probe queries. Telling would mean looking at holdout rows, which are never re-scored. Two checks are still to come:
- The AMD rows (MI300X, 192 GB, a different allocator) test whether misses grow with the card.
- The Nov 2 scripts test the fix on users' jobs.

Monotonicity also means a below-probe underestimate can't hide an OOM when the smallest probe fit. The OOM-relevant misses are the ones above the probes.
