# Divergence classifier: held-out fit reports

`scripts/fit_divergence.py` wrote both files. Folds and the bootstrap group runs by settings family, because the sweep is one architecture (a small GPT), so grouping by architecture would leave one group. Targets (plan): AUROC at least 0.85, and a warning at least 100 steps before the run diverges.

| Report | Runs | Diverged | AUROC (95%) | Median lead, steps (95%) | False alarms | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| [divergence_nov1.json](divergence_nov1.json) (final) | 720 (377 L4, 343 A10) | 44 | 0.66 (0.56 to 0.76) | 138 (3 to 269) | 73 of 676 stable runs | Missed: not used; the warning comes from the threshold rules (D6) |
| [divergence_nov1_329.json](divergence_nov1_329.json) (first) | 329 (L4) | 21 | 0.58 (0.41 to 0.74) | 3 (0 to 243) | 38 of 308 stable runs | Missed; superseded by the final fit |

37 of the 44 divergences in the final fit are fp16 blow-ups, and 15 come in the first 30 steps, before the classifier has a full window to read.

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
