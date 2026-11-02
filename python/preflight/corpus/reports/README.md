# Divergence classifier: held-out fit reports

`scripts/fit_divergence.py` wrote both files. Folds and the bootstrap group runs by settings family, because the sweep is one architecture (a small GPT), so grouping by architecture would leave one group. Targets (plan): AUROC at least 0.85, and a warning at least 100 steps before the run diverges.

| Report | Runs | Diverged | AUROC (95%) | Median lead, steps (95%) | False alarms | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| [divergence_nov1.json](divergence_nov1.json) (final) | 720 (377 L4, 343 A10) | 44 | 0.66 (0.56 to 0.76) | 138 (3 to 269): the point estimate clears 100, the interval doesn't | 73 of 676 stable runs | Missed: not used; the warning comes from the threshold rules (D6) |
| [divergence_nov1_329.json](divergence_nov1_329.json) (first) | 329 (L4) | 21 | 0.58 (0.41 to 0.74) | 3 (0 to 243) | 38 of 308 stable runs | Missed; superseded by the final fit |

37 of the 44 divergences in the final fit are fp16 blow-ups, and 15 come in the first 30 steps, when the classifier has at most a few steps past its 25-step window to read.

## Reliability bins, final fit

Held-out runs, binned by the classifier's highest P over the run. A bin shows a number only with at least 10 runs and the shown P inside the Wilson interval of the bin's observed rate.

| P bin | Runs | Mean P | Observed diverged |
| --- | --- | --- | --- |
| 0 to 5% | 618 | 0.2% | 2.6% |
| 5 to 20% | 43 | 10.6% | 9.3% |
| 20 to 50% | 22 | 33.4% | 18.2% |
| 50 to 80% | 21 | 67.6% | 38.1% |
| 80 to 95% | 8 | 86.0% | 50.0% |
| 95 to 100% | 8 | 98.0% | 100% |

In each JSON file, `bins` rows are `[low, high, runs, mean P, observed rate]`.

# Loss forecast: held-out coverage (Nov 2)

[loss_coverage_nov2.json](loss_coverage_nov2.json), from `scripts/loss_coverage.py` on the divergence sweep's 676 runs that didn't diverge (`corpus/data/loss_curves_heldout_nov2.json.gz`); LC-PFN's quantiles came from `corpus/lcpfn_app.py::coverage` on Modal (CPU, run 41397869309, 1,352 forecasts, none failed). Each run is seen for its first 60 or 150 of 600 steps and forecast at step 600; the truth is the mean of its last 10 steps. Target: the 90% band covers at least 90%.

| Seen | Power law: covered of answered (Wilson 95%) | LC-PFN | Ensemble | Median abs. log error (power, LC-PFN, ensemble) |
| --- | --- | --- | --- | --- |
| 60 steps | 111 of 465, 24% (20% to 28%); 211 abstained | 627 of 676, 93% (90% to 95%) | 272 of 296, 92% (88% to 95%); 380 abstained | 0.17, 0.07, 0.08 |
| 150 steps | 147 of 636, 23% (20% to 27%); 40 abstained | 499 of 676, 74% (70% to 77%) | 490 of 547, 89.6% (86% to 92%); 129 abstained | 0.10, 0.04, 0.04 |

Verdict: not wired. The power law's band is far too narrow; LC-PFN covers at 60 steps but not at 150, and the ensemble covers at 60 steps only where it answers (296 of 676 curves) and falls just short at 150. The loss forecast stays out of `predict`, as before. One model family and one run length, so this says nothing yet about other jobs.
