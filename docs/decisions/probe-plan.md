# Probe plan: how long, where, how many (Oct 27)

These probe-plan settings decide whether A's measurements can meet the plan's
validation targets (step time within 15%, peak memory within 10%, no missed
OOMs, agent cost inside its 90% interval). Each was chosen by simulating 3–5
candidates (`scripts/mc_probe_plan.py`, `mc_agent_sample.py`, `mc_kernels_gpu.py`).

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

## D3. Agent sample size and the runaway margin (rerun Oct 27, 10,000 trials)

The first pass added a rule-of-three margin only when no runaway-sized task
was in the sample. Rerunning 50 tasks with wider margins showed why that can't
reach 88%: coverage plateaus near 87% because the misses happen when a runaway
*was* sampled (12.4% of trials), where the margin never applies.

| Candidate | 90% interval covers (95% CI) | Width / total |
| --- | --- | --- |
| 30 tasks, tail 3/n | 89.4% (88.8–90.0) | 1.33 |
| 50 tasks, plain bootstrap | 70.9% (70.0–71.8) | 0.63 |
| 50 tasks, tail 3/n | 85.2% (84.5–85.9) | 0.95 |
| 50 tasks, tail 4/n | 86.1% (85.5–86.8) | 1.05 |
| 50 tasks, tail 5/n | 86.5% (85.8–87.2) | 1.12 |
| 50 tasks, tail 6/n | 86.8% (86.1–87.4) | 1.18 |

The fix: always add the gap between a one-sided Clopper–Pearson upper bound on
the runaway rate and the observed rate, priced at the observed runaway cost (20×
the median when none was seen). With no runaway seen this is the rule of three;
with some seen it widens the upper end the bootstrap under-reaches.

| 50 tasks, runaway bound at | Covers (95% CI) | Width / total |
| --- | --- | --- |
| 70% | 86.1% (85.4–86.7) | 0.86 |
| 75% | 87.7% (87.1–88.4) | 0.90 |
| **80%** | **89.1% (88.5–89.7)** | **0.96** |
| 85% | 90.7% (90.1–91.2) | 1.03 |
| 90% | 92.4% (91.8–92.9) | 1.12 |
| 95% | 94.5% (94.0–94.9) | 1.28 |

The narrowest setting that clears the 88% target is **50 tasks with an 80%
runaway bound**: 89.1% coverage at 0.96× the total, narrower than 30 tasks
(1.33×) and passing. An 85% bound reaches the nominal 90% at 1.03×. The
default stays at 30 until Chiya decides. Whichever is chosen is then checked
against the validation table's real comparison (a full benchmark run against
its 20–50-task sample) once Modal and the Meta API are live. The estimator is
B's; the request is in docs/requests.md. Reproduce with
`uv run python scripts/mc_agent_sample.py`.

## K. Kernel rows kept for B's cross-GPU model (Oct 27, 20,000 trials)

B scales each recorded kernel by its roofline on the target GPU and keeps its
measured efficiency; device time not in the table lands in B's host bucket,
which does not scale. Share of cross-GPU step-time estimates within 20%:

| Rows kept | Within 20% | Median error | Rows |
|---|---|---|---|
| top 32 | 80.1% | 9.6% | 32 |
| top 64 | 89.3% | 6.3% | 64 |
| top 128 | 96.6% | 3.2% | 128 |
| **top 64 + tail rolled up per class** | **100%** | **0.2%** | **78** |
| every kernel | 100% | 0.0% | 392 |

Chosen: top 64 by time, then one `(tail) <class>` row per kernel class, then
`(unattributed device time)` when kernels outside aten hold over 1% of device
time (`recorder/kernels.py`). On CPU probes, autograd nodes, view ops and host
syncs are left to the host bucket: they launch nothing on a GPU.

## G. Which GPU a probe runs on (Oct 27, 40,000 simulated jobs)

Jobs drawn from LoRA, full fine-tune, pretraining, inference and agent runs,
median 1.2B parameters with a heavy tail, a 30% error on the memory estimate
made before any probe, and 20% of jobs whose size nothing in the code gives
away (a model built in the script), which start on the cheapest GPU. Usable
memory is B's `usable_capacity_gib()` in GB, as the code uses it.

| Policy | Measured on first try | Measured at all | Mean GPU $ a job |
|---|---|---|---|
| always L4 | 72.8% | 72.8% | $0.030 |
| always A100-80GB | 94.2% | 94.2% | $0.099 |
| always H100 | 94.2% | 94.2% | $0.149 |
| **size-aware, 1.3× margin, one step up** | **89.1%** | **93.0%** | **$0.059** |
| size-aware, 1.15× margin, one step up | 88.1% | 93.0% | $0.057 |

Chosen: the cheapest GPU with 1.3× the estimated need, a step up after a GPU
OOM (not a host OOM). 1.15× saves $0.002 a job for a further point of
first-try failures, each a wasted container start the visitor waits through.

### L. The top of that ladder

H100 holds no more memory than A100-80GB (0.03 GiB), so it never wins a
memory choice and a step from one to the other can't clear an OOM; steps need
5% more memory. The question is what sits above 80 GB. Same policy, same jobs:

| Top | Measured | Mean GPU $ a job |
|---|---|---|
| A100-80GB | 93.2% | $0.055 |
| A100-80GB + H100 | 93.2% | $0.058 |
| + RTX PRO 6000 96 GB | 93.8% | $0.057 |
| **+ H200 141 GB** | **95.7%** | **$0.063** |
| + H200 + B200 180 GB | 96.4% | $0.067 |

Provisional: L4, L40S, A100-80GB, H200 (`probes/gpu_choice.py`), pending
Chiya's call on B200. T4 is cheaper than L4 but has no bf16 tensor cores, so
a bf16 probe there times an emulated path.

### U. Steps up when the size was unknown

On that ladder, after a GPU OOM:

| Policy | Measured, all jobs | Measured, unknown size | Mean GPU $ a job |
|---|---|---|---|
| one step up | 95.7% | 88.2% | $0.064 |
| up to two steps | 97.0% | 94.5% | $0.066 |
| straight to the largest | 97.6% | 97.8% | $0.071 |
| start unknown sizes on L40S, one step | 97.0% | 94.5% | $0.071 |
| **walk the whole ladder** | **97.6%** | **97.8%** | **$0.068** |
| whole ladder, and two steps when sized | 97.6% | 97.8% | $0.068 |

Chosen: walk the ladder when the size was unknown, one step when it was
estimated (a second step for sized jobs adds nothing). Each failed rung costs
about a minute, since an OOM fails at load or the first step. Reproduce K, G,
L and U with `uv run python scripts/mc_kernels_gpu.py`.

## Also decided with these

- The current-settings probe pulls the first eval (eval_every → step 3) and
  one checkpoint (two steps before the end) inside the probe window, so an
  eval-time OOM or a slow save is measured. Eval loops are cut to 2 batches;
  save and eval time are subtracted from step time and reported on their own.
- Only the current-settings probe is profiled (wait 1, warmup 1, active 3, on
  extra steps after the timed ones).
