# Engine settings: memory fit, bands, OOM words, divergence labels, cross-GPU time, corpus budget (Oct 27)

Seven settings decide whether B's numbers can meet the plan's validation targets (peak memory within 10%, no missed OOMs, 90% intervals covering at least 88%, step time within 20% on an unmeasured GPU) and whether the corpus stays inside its budget. Each was chosen by simulating 3 to 9 candidates over 3,000 to 20,000 trials (20,000 for memory, 4,000 for divergence and cross-GPU, 3,000 sweeps for the budget) (`uv run python scripts/mc_engine_settings.py`).

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

**Chosen: K1, with K4 as the fallback** when a kernel has no FLOP count. A's profiler window doesn't record FLOPs today, so kernels are classed by name in that case. The whole-step FLOPs ratio and "same MFU", the most common rules of thumb, are off by half because most steps are not compute-bound throughout.

## S5. Keeping the corpus sweep inside its budget

| Candidate | P(overspend) | Mean overspend | Budget left unused | Configs run |
| --- | --- | --- | --- | --- |
| B1 overhead 1.00, fixed 45 s per config | 92.3% | $0.47 | 0.0% | 840 |
| B2 overhead 1.25, fixed 45 s (B's first) | 83.0% | $0.36 | 0.0% | 830 |
| B3 overhead 1.25, running p75 of real configs | 22.7% | $0.01 | 0.1% | 828 |
| B4 overhead 1.35, running p90, one wave held in reserve | 0.0% | $0.00 | 8.8% | 763 |
| **B5 overhead 1.25, running p90, one wave held in reserve** | **0.0%** | **$0.00** | **8.1%** | **769** |

**Chosen: B5.** A fixed per-config estimate overspends in 83 to 92% of sweeps because real configs run longer than 45 s, even though each overspend is small since the sweep re-plans every wave. Learning the per-config time from finished configs, plus holding one wave in reserve, never overspends and still uses 92% of the budget.
