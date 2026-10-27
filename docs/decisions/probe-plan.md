# Probe plan: how long, where, how many (Oct 27)

Three probe-plan settings decide whether A's measurements can meet the plan's
validation targets (step time within 15%, peak memory within 10%, no missed
OOMs, agent cost inside its 90% interval). Each was chosen by simulating 3–5
candidates over 20,000 trials (`uv run python scripts/mc_probe_plan.py`).

## Assumptions the simulation makes

- Step time: lognormal jitter with σ from 2% (compute-bound) to 10%
  (data-bound); a per-step stall chance up to 8% at 1.5–3× (GC, worker
  refill, logging); the first step 2–6× slower (compile, autotune, first
  allocations), the second up to 1.4×, the third sometimes 1.1×.
- Memory: weights and state 3.6–4.4 GB; linear activation and logits terms per
  token; half the scripts use eager attention (a bs·s² term), half SDPA;
  allocator overhead 2–12% over allocated, rounded to 2 MiB (PyTorch's caching
  allocator docs give about 9% for a fragmented case and 2 MiB / 20 MiB
  rounding). Queries stay within 2× of the largest probed bs·s, beyond which
  B abstains anyway.
- Agent cost: per-task cost lognormal with σ 0.5–1.2, plus 0–5% runaway tasks
  at 10–40×, over a 500-task run.

## D1. Steps per probe and the step-time estimator

| Candidate | Within 5% | Within 15% | Mean error | Steps |
| --- | --- | --- | --- | --- |
| warmup 2, timed 3, mean | 50.8% | 89.2% | 8.4% | 5 |
| warmup 2, timed 3, median | 53.4% | 97.8% | 5.7% | 5 |
| **warmup 2, timed 5, median** | **55.1%** | **99.3%** | **5.0%** | **7** |
| warmup 3, timed 8, 20% trimmed mean | 54.3% | 98.6% | 5.0% | 11 |
| warmup 3, timed 8, mean | 49.7% | 92.5% | 6.4% | 11 |

Chosen: warmup 2, timed 5, median (`probes/plan.py`). Two extra steps buy ten
points of the 15% target; more steps after that buy nothing. The record
carries `p50_s` and `mean_s`; **B should project run time from `p50_s`**.

## D2. Which points calibrate memory

| Candidate | Within 10% | Missed OOM | False alarm | Probes |
| --- | --- | --- | --- | --- |
| current + half batch | 65.9% | 4.44% | 0.95% | 2 |
| current + double batch (half on OOM) | 63.6% | 4.58% | 0.66% | 2.6 |
| current + half sequence | 58.9% | 4.38% | 1.03% | 2 |
| **current + half batch + half sequence** | **88.2%** | **1.01%** | **1.20%** | **3** |
| analytical terms only | 42.2% | 16.65% | 0.00% | 1 |

Chosen: three probes. Two points along one axis can't separate the bs·s terms
from eager attention's bs·s² term, which is exactly what drives OOM at long
sequences. The third probe costs about a minute of GPU (about $0.01 on an L4)
and cuts missed OOMs by three quarters; the rest comes from B's intervals.
With batch 1 the plan varies sequence twice (s/2, s/4) instead.

## D3. Agent sample size (open: Chiya's call)

| Candidate | 90% interval covers | Width / total | Share of run |
| --- | --- | --- | --- |
| 20 tasks, bootstrap | 59.5% | 0.58 | 4% |
| 30 tasks, bootstrap | 63.8% | 0.59 | 6% |
| 50 tasks, bootstrap | 70.8% | 0.65 | 10% |
| 30 tasks, bootstrap + rule-of-three tail | 89.1% | 1.34 | 6% |
| 50 tasks, bootstrap + rule-of-three tail | 85.6% | 0.95 | 10% |

A plain bootstrap under-covers at every size: runaway tasks are rare enough to
be missed. Adding a rule-of-three tail (if no runaway-sized task was seen, up
to 3/n of tasks could be) restores coverage. 30 tasks covers best but with a
wide interval; 50 tasks is 30% narrower at 3.5 points less coverage and costs
4% more of the run. The default is 30 until Chiya decides; the estimator
itself is B's.

## Also decided with these

- The current-settings probe pulls the first eval (eval_every → step 3) and
  one checkpoint (two steps before the end) inside the probe window, so an
  eval-time OOM or a slow save is measured. Eval loops are cut to 2 batches;
  save and eval time are subtracted from step time and reported on their own.
- Only the current-settings probe is profiled (wait 1, warmup 1, active 3, on
  extra steps after the timed ones).
