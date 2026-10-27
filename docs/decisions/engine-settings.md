# Engine settings: memory fit, bands, OOM words, divergence labels, cross-GPU time, corpus budget (Oct 27)

> **Calibration: simulated. Don't quote these figures.** Every result below comes from a Monte Carlo, and in most of them the simulated truth has the same form as the fitted model. They show that each rule works on its own assumptions, not that the engine is accurate (Review, Oct 27). Model cards say `calibration: simulated` until split-conformal calibration on held-out real runs replaces them. None of these numbers goes in the README, the video or the UI.

Seven settings decide whether B's numbers can meet the plan's validation targets (peak memory within 10%, no missed OOMs, 90% intervals covering at least 88%, step time within 20% on an unmeasured GPU) and whether the corpus stays inside its budget. Each was chosen by simulating 3 to 9 candidates over 3,000 to 20,000 trials (20,000 for memory, 4,000 for divergence and cross-GPU, 3,000 sweeps for the budget) (`uv run python scripts/mc_engine_settings.py`). S6 re-checks A's agent-total rule with B's code.

## Assumptions

- **Memory.** These follow A's D2 where the two overlap:
  - allocator overhead of 2 to 12% with ±2% probe-to-probe jitter, rounded to 2 MiB;
  - eager attention adds a bs·s² term, in 30% of jobs;
  - queries stay within 2× of the current bs·s.

  B's own error comes on top. The analytical model gets the weights right within 5%, the activation coefficient off by 30%, the logits term off by 15%, and the attention kernel wrong 10% of the time. The current run sits at 30 to 110% of the GPU, so near-OOM jobs are common.
- **Divergence.** Simulated loss curves:

  | Curve | Share | Counts as diverged |
  | --- | --- | --- |
  | stable power-law decay | 35% | no |
  | spike that recovers | 15% | no |
  | plateau | 10% | no |
  | NaN | 12% | yes |
  | exponential blow-up | 18% | yes |
  | slow upward drift | 10% | yes |
- **Cross-GPU.** 40 kernels per step with log-normal FLOPs and arithmetic intensity. Each kernel's efficiency drifts between GPUs (12% within a vendor, 25% across vendors), plus unscaled host time. The truth model shares K1's roofline structure, so treat K1's lead as an upper bound until the held-out GPU test on real kernels.
- **Corpus.** Real configs take 30 to 90 s on average, log-normal with σ 0.6. Modal bills 5 to 35% more than the measured GPU time, for cold starts, image pulls, idle scale-down and retries. Waves hold 32 configs.

## S1. How probes calibrate the memory model

| Candidate | Within 10% | Within 10% near the cap | Missed OOM | False alarm |
| --- | --- | --- | --- | --- |
| E1 empirical S + β·bs·s + γ·bs·s² (A's D2) | 84.8% | 78.8% | 1.58% | 1.02% |
| E2 analytical + constant offset | 74.1% | 83.9% | 1.65% | 1.48% |
| **E3 analytical + residual a + b·bs·s, least squares over the probes** | **88.9%** | **93.6%** | **0.98%** | **0.94%** |
| E4 scaled analytical S + α·(dynamic part) | 88.5% | 87.4% | 1.47% | 0.77% |
| E5 mean of E1 and E4 | 87.8% | 83.9% | 1.44% | 0.85% |

**Chosen: E3.** The analytical terms carry the shape, including the s² term, and the probes correct the level and slope. It wins most clearly near the GPU's capacity. That is also where a probe tends to OOM, which leaves the purely empirical E1 one point short. 0.7% of jobs OOM on every probe; the engine abstains on those and says so.

## S2. The 90% band and the fits / tight / fails words

| Band rule (E3) | Coverage | Near the cap | Median width (hi/lo) |
| --- | --- | --- | --- |
| R1 fixed σ 0.03 | 71.7% | 76.8% | 1.104 |
| R2 fixed σ 0.05 | 85.9% | 90.5% | 1.179 |
| R3 calibrated constant, half-width 0.107 in log space | 90.0% | 94.3% | 1.238 |
| R4 R3 + disagreement with E4 | 94.6% | 96.8% | 1.238 |
| **R5 R3 as a floor, widened to 1.5 × the leave-one-probe-out error** | **94.8%** | **96.4%** | **1.238** |

**Chosen: R5.** It has the same median width as R3. It only widens when the three probes disagree with each other, which is exactly when a fit deserves less trust. With no probe yet, the analytical-only half-width is 0.241, measured in the same simulation (57% of queries within 10%). The conformal layer replaces both constants with values calibrated on the real corpus at G1.

**With one or two probes** (S2c: the current probe OOMed, or one failed), coverage on in-range queries, overall / near the cap:

| Half-width | Current only | Current + half batch | Current + half seq | Half batch + half seq |
| --- | --- | --- | --- | --- |
| 0.107 (R3) | 80.5% / 87.7% | **91.5% / 93.7%** | **90.8% / 93.9%** | **91.0% / 92.2%** |
| 0.134 | 85.1% / 91.1% | 93.8% / 95.6% | 93.2% / 95.5% | 93.4% / 94.7% |
| **0.160** | **88.1% / 93.0%** | 94.9% / 96.4% | 94.4% / 96.4% | 94.7% / 95.5% |
| 0.214 | 92.1% / 95.3% | 96.2% / 97.2% | 95.7% / 97.0% | 96.2% / 97.1% |
| 0.241 (prior) | 93.4% / 96.1% | 96.6% / 97.5% | 96.1% / 97.3% | 96.7% / 97.3% |

**Chosen: 0.107 with two probes and 0.160 with one**, the narrowest widths that meet the plan's 88% coverage target (0.160 gives 88.1% with one probe; 0.134 gives 85.1%). Queries outside half the smallest to twice the largest probed work units abstain (the contract's rule), so no band is stretched past the probes.

| fits below / fails at or above | P(fits given it OOMs) | P(fails given it fits) | Share tight |
| --- | --- | --- | --- |
| 1% / 50% | 0.15% | 1.09% | 10.4% |
| 2% / 30% | 0.19% | 2.87% | 7.0% |
| 5% / 50% | 0.27% | 1.09% | 7.3% |
| 10% / 50% | 0.62% | 1.09% | 5.7% |
| 5% / 80% | 0.27% | 0.44% | 9.2% |
| **1% / 80%** | **0.15%** | **0.44%** | **12.4%** |

**Chosen: fits below 1%, fails at 80% or more, tight between.** It ties for the fewest missed OOMs and the fewest false alarms. The price is that one setting in eight reads "tight", which near the cliff is the honest answer, and the UI offers a probe there.

## S2d. Memory below the smallest probe (Chiya's call, Oct 27)

Chiya decided that sizes below the smallest probe get an estimate labelled "extrapolated", with a wider range and a "run a probe here" offer, and that far outside still abstains. Five ways to widen the band, scored on S1's 20,000 jobs, with queries 2^-k below the smallest probe (batch lowered first, then sequence). The engine's existing rule abstains once the band passes +-50%. Each cell gives coverage of the estimates given, then the share abstained:

| Rule | 0.5 halvings | 1 | 2 | 4 | 6 | 8 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 constant x1.5 | 100% / 6% | 99.9% / 6% | 96.8% / 6% | 89.8% / 7% | 88.2% / 7% | 87.4% / 8% | 87.6% / 8% |
| G2 constant x2 | 100% / 9% | 100% / 9% | 98.9% / 9% | 93.8% / 9% | 92.0% / 10% | 91.6% / 10% | 91.5% / 10% |
| **G3 x(1 + 0.25 per halving)** | **99.8% / 5%** | **99.5% / 5%** | **96.7% / 6%** | **93.8% / 9%** | **94.5% / 13%** | **95.8% / 18%** | **96.7% / 22%** |
| G4 x(1 + 0.5 per halving) | 99.9% / 5% | 99.9% / 6% | 98.9% / 9% | 97.2% / 17% | all abstain | all abstain | all abstain |
| G5 + 0.05 per halving | 99.9% / 4% | 99.8% / 5% | 97.8% / 5% | 95.9% / 9% | all abstain | all abstain | all abstain |

**Chosen: G3.** It keeps at least 93.8% coverage at every depth with the fewest abstentions near the probes. Its abstentions then grow with distance and fall on the jobs whose probes fit worst, so "far outside abstains" happens where the estimate stops being trustworthy rather than at an arbitrary line. Memory at small sizes is mostly fixed terms the probes do measure, which is why it can go this far down. Above twice the largest probe still abstains, as before.

## S3. Labelling a corpus run as diverged

Curves: stable decay 32%, spike that recovers 14%, plateau that never learns 9% (all not diverged); NaN 11%, exponential blow-up 16%, slow upward drift 9%, and spike then collapse to the trivial baseline 9% (all diverged). Half the collapses happen in the first 30 steps, before the run has learned anything; a corpus smoke test on CPU turned this case up (lr 3.0: loss 6.9, then 271, then flat at ln 256 = 5.55, uniform over bytes), and the first version of this study didn't model it.

| Rule | Accuracy | Fires on recovered spikes | Fires on plateaus | Misses drift | Misses collapse | Median lag (steps) |
| --- | --- | --- | --- | --- | --- | --- |
| D1 non-finite only | 65.3% | 0% | 0% | 100% | 100% | 0 |
| D2 raw loss > 1.5 × running min | 85.4% | 100% | 0% | 6.7% | 0% | 3 |
| D3 EMA above best + max(0.5 × best, 0.2) (B's first) | 80.1% | 97.7% | 49.0% | 19.8% | 0% | 8 |
| D4 D3 held 20 steps | 83.4% | 69.8% | 49.0% | 25.3% | 0% | 27 |
| D5 D3 held 50 steps | 91.8% | 2.9% | 49.0% | 29.9% | 5.7% | 57 |
| D6 rolling median(25) > 1.25 × its running min + 0.1 | 88.5% | 76.4% | 0% | 5.9% | 3.7% | 15 |
| D7 D6 held 30 steps | 92.8% | 30.9% | 0% | 10.3% | 21.4% | 15 |
| D8 hindsight: the run ends well above its best, onset from D6 | 96.9% | 0% | 0% | 8.0% | 26.9% | 15 |
| **D9 D8, or a spike past 2× the starting loss with no learning below 90% of it** | **99.2%** | **0%** | **0%** | **8.0%** | **0.3%** | **14** |

**Chosen: D9 for corpus labels**, with one refinement the simulation can't see: the no-learning line is 90% of the trivial baseline, min(starting loss, ln vocab), because a large init can start a run above uniform prediction (the smoke-test run started at 6.9 against ln 256 = 5.55). In the simulation every run starts at the baseline, so the numbers above are unchanged. Corpus runs are complete when they're labelled, so hindsight is allowed: a spike that recovers isn't divergence, a run that ends far above its best is, and so is a run that blew up early and never learned. The onset step is what the classifier learns to warn about early. Live warnings during a probe still use rules that can't see the future (D6 as an early hint) until the classifier replaces them at G2. B's first rule fired on half of all plateaus through its "final above initial" clause, which D9 drops.

## S4. Step time on a GPU the job wasn't probed on

| Candidate | Within 20% | Within 35% | Median error |
| --- | --- | --- | --- |
| **K1 each kernel keeps its measured roofline efficiency** | **98.2%** | **100.0%** | **4.2%** |
| K2 whole step × peak-FLOPs ratio | 8.1% | 22.7% | 50.6% |
| K3 whole step × bandwidth ratio | 92.1% | 99.4% | 7.6% |
| K4 each phase by arithmetic intensity | 97.7% | 99.9% | 4.4% |
| K5 assume the same MFU | 8.1% | 22.7% | 50.6% |

**Chosen: K1, with K4 as the fallback** when a kernel has no FLOP or byte count. A's kernel table (`Probe.kernels`, top 64 ops plus a roll-up row per class) counts FLOPs for matmuls and attention and estimates bytes for the rest; rows with neither are scaled by their class, which the classifier reads from the op name (CUDA, HIP, Triton and aten; on a real profiled step no op is left unclassed). Replay scales each row by the ratio of its rooflines on the two GPUs. That equals "keep the measured efficiency" whenever the counts are consistent, and stays exact on the probed GPU when an estimated byte count overshoots what the GPU could move. A kernel table from a CPU probe isn't scaled to a GPU: the engine abstains. The whole-step FLOPs ratio and "same MFU", the most common rules of thumb, are off by half because most steps are not compute-bound throughout.

## S4b. Step time below the smallest probe

The same decision for the same-GPU step-time model. Small batches under-use the GPU, so time stops falling linearly with the batch. Simulated over 20,000 jobs with a saturating utilisation curve, A's probes at x and x/2, 3% timing noise and the model's own fit:

| Rule | 0.5 halvings | 1 | 2 | 2.5 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| T0 no widening | 98.6% | 90.1% | 63.4% | 56.0% | 51.5% | 46.7% |
| T1 memory's rule, x(1 + 0.25 per halving) | 99.1% | 94.6% | 76.8% | 70.5% | 66.8% | 63.4% |
| T3 x(1 + 1 per halving) | 99.8% | 98.6% | 95.8% | abstains | abstains | abstains |
| **T4 + 0.1 (log) per halving** | **99.6%** | **97.8%** | **92.7%** | **88.9%** | abstains | abstains |
| T5 upper end only, + 0.1 per halving | 99.6% | 94.3% | 74.1% | 68.4% | abstains | abstains |

**Chosen: T4.** It holds 88.9% or better until the band passes +-50%, about 2.7 halvings (6.5x) below the smallest probe, and abstains from there. Memory's rule is too narrow for time. A one-sided band does worse because the fitted line can overshoot as well as undershoot below the probes. Reproduce with `uv run python scripts/mc_engine_settings.py s2d` and `s4b`.

## S4c. How wide K1's band is, by vendor

S4 scored K1's point error. The band needs the error's 90th percentile, and S4's per-kernel efficiency drift (0.12 same vendor, 0.25 cross vendor) is an assumption no AMD data backs yet. So the q90 from S4 (0.101 same vendor, 0.179 cross vendor, in log units) was re-scored with the true drift at 1x, 1.5x and 2x that assumption. Coverage of the 90% band:

| Margin on S4's q90 | Same vendor (drift x1 / x1.5 / x2) | Cross vendor (drift x1 / x1.5 / x2) |
| --- | --- | --- |
| x1 | 90.3% / 76.5% / 65.1% | 89.7% / 74.6% / 61.5% |
| x1.25 | 95.4% / 84.7% / 74.4% | 94.3% / 83.7% / 70.0% |
| x1.5 | 97.3% / 90.5% / 81.9% | 96.8% / 88.8% / 78.0% |
| **x2** | **99.4% / 96.2% / 91.2%** | **99.1% / 94.9% / 89.9%** |

**Chosen: margin x2.** `SIGMA_SAME_VENDOR` stays 0.12, now backed by S4c rather than set by hand, and `SIGMA_CROSS_VENDOR` drops from a hand-set 0.25 to 0.22. Combined with the same-GPU sigma, 0.25 put every NVIDIA-to-AMD band past the +-50% abstain line, so the engine never gave AMD a number. At 0.22 it gives one, with the wider range the plan promises until AMD is measured. The conformal layer recalibrates both from the corpus. Reproduce with `uv run python scripts/mc_engine_settings.py s4c`.

## S5. Keeping the corpus sweep inside its budget

| Candidate | P(overspend) | Mean overspend | Budget left unused | Configs run |
| --- | --- | --- | --- | --- |
| B1 overhead 1.00, fixed 45 s per config | 92.3% | $0.47 | 0.0% | 840 |
| B2 overhead 1.25, fixed 45 s (B's first) | 83.0% | $0.36 | 0.0% | 830 |
| B3 overhead 1.25, running p75 of real configs | 22.7% | $0.01 | 0.1% | 828 |
| B4 overhead 1.35, running p90, one wave held in reserve | 0.0% | $0.00 | 8.8% | 763 |
| **B5 overhead 1.25, running p90, one wave held in reserve** | **0.0%** | **$0.00** | **8.1%** | **769** |

**Chosen: B5.** A fixed per-config estimate overspends in 83 to 92% of sweeps because real configs run longer than 45 s, even though each overspend is small since the sweep re-plans every wave. Learning the per-config time from finished configs, plus holding one wave in reserve, never overspends and still uses 92% of the budget.

## S6. Agent and API run totals from a sample of tasks (re-check of A's D3)

A's D3 rerun chose the rule; B implemented it in `engine/agent_cost.py` and re-scored it on A's generator (same seed and laws, 10,000 simulated runs of 500 tasks): a bootstrap of the sample mean, with the upper end raised by (one-sided Clopper-Pearson bound on the runaway rate minus the observed rate) times the runaway cost, whether or not a runaway was sampled.

| Sample | Runaway bound | 90% interval covers (95% CI) | Width / total | Runaway-rate interval covers |
| --- | --- | --- | --- | --- |
| 30 tasks | 75% | 88.7% (88.1-89.3) | 1.07 | 98.3% |
| **30 tasks** | **80%** | **90.3% (89.8-90.9)** | **1.16** | 98.3% |
| 30 tasks | 85% | 92.3% (91.7-92.8) | 1.26 | 98.3% |
| 30 tasks | 90% | 94.0% (93.6-94.5) | 1.42 | 98.3% |
| 50 tasks | 75% | 87.9% (87.3-88.5) | 0.91 | 97.9% |
| **50 tasks** | **80%** | **89.5% (88.8-90.1)** | **0.96** | 97.9% |
| 50 tasks | 85% | 91.0% (90.5-91.6) | 1.03 | 97.9% |
| 50 tasks | 90% | 92.7% (92.2-93.2) | 1.12 | 97.9% |

**Chosen: an 80% bound, for either sample size.** It reproduces A's 89.1% at 50 tasks, and with this rule 30 tasks also clears the 88% target, at an interval 21% wider than 50 tasks gives. The sample size stays Chiya's call (A's D3). The runaway-rate readout (`risk.runaway`, a two-sided Clopper-Pearson interval on the share of tasks above 8x the median) covers the run's true share 98% of the time: conservative, as exact binomial intervals are. When most tasks cost nothing (cached), the median is 0 and A's rule would call every paid task a runaway; B takes 8x the median of the paid tasks instead. A's generator has no free tasks, so this case is covered by a unit test, not by S6. Reproduce with `uv run python scripts/mc_engine_settings.py s6`.

## S6b. 30 tasks, then a 20-task top-up when the range straddles the budget (Chiya's call, Oct 27)

Chiya chose 30 tasks with the 80% bound as the default. When the cost range straddles the budget, the guide offers 20 more tasks with the user's click (`agent_cost.budget_check`, `combine_samples`). Topping up only after seeing a straddling range is a data-driven choice, so the final coverage was checked over 10,000 runs from A's generator, with budgets at 0.5x to 2x the run's true total:

| Quantity | Result |
| --- | --- |
| Coverage, 30 tasks alone | 89.8% |
| Runs topped up | 71.6% |
| Topped-up runs still straddling after 50 tasks | 83.0% |
| Final coverage, all runs | 88.2% (+-0.6) |
| Final coverage, topped-up runs | 92.4% |
| Under/over calls that were right, among runs with a clear call | 93.9% |

The rule meets the 88% target. A range that misses tends not to straddle the budget either, so runs that skip the top-up cover less (about 78%), and the topped-up ones make up for it. Under and over calls are right 94% of the time. The top-up settles only about one straddle in six when budgets sit this close to the true cost. So the guide should promise a narrower range, not a settled answer. Reproduce with `uv run python scripts/mc_engine_settings.py s6b`.

