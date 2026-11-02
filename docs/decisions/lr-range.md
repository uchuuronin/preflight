# The learning-rate range test: the GPU check (D, Nov 2)

Chiya's list item 3 (Nov 2, 11:15): a new probe type, Smith's range test (about 100 steps, the
learning rate rising exponentially from 1e-7 to 1, loss against rate); suggest the rate at the
steepest loss drop and flag where it diverges; store it in the probe's extras (no contract change),
behind `probes.lr_range` (off). Validate on the LoRA and nanoGPT zoo scripts on an L4 or A10, cap
US$2; compare the suggestion with each script's own rate, and run 50 steps at the suggested rate to
confirm it trains without diverging.

Built in `f909ac3` (`probes/lr_range.py`, the recorder's `lr_range`, `probes/local.py`
`with_lr_range`, the pipeline's `p1-lr` point). The check is `runner/lr_range_app.py`, run once as
`d-lr-range-gpu` (GitHub 41490183749, `ws-d` `f909ac3`, Nov 2, 17:20 UTC), on an A10 both times.
For each script: the range test, then 50 steps held at the suggested rate and 50 at the script's
own rate as the baseline (the recorder holds the rate, so the scripts' warm-ups and schedules
don't change it), the eval off.

## Results [measured, A10, `out/lr_range_*.json` on `remote-runs`]

| | LoRA fine-tune (`zoo/finetune_lora_hf`) | nanoGPT pretrain (`zoo/pretrain_nanogpt`) |
| --- | --- | --- |
| Script's own rate | 1e-4 | 6e-4 (its peak, after a warm-up) |
| Range test | 68 points kept, stopped at divergence (about step 69); the ramp checked (`lr_range.checked`) | 97 points kept, divergence at step 99; the ramp checked |
| Suggested (steepest fall) | 7.9e-5, 0.8 times its own | 2.5e-4, 0.4 times its own |
| Lowest loss at | 7.7e-4 | 3.4e-3 |
| Diverged at | 7.6e-3 | 1.0 (the ramp's last step) |
| 50 steps at the suggested rate | loss 1.287 to 1.216 (first 10 steps' mean to last 10's), no divergence | 8.74 to 6.01, no divergence |
| 50 steps at its own rate | 1.275 to 1.215 | 7.87 to 6.03 |
| GPU time (range, suggested, own) | 179 s, 100 s, 97 s | 124 s, 65 s, 64 s |

**Cost:** US$0.19 for the two containers' whole lives (633 s on an A10 at US$1.10 an hour)
[derived: measured time at list price], against the item's US$2.

## What it means

- On both scripts the suggestion lands just below the rate the script's author chose (0.8 and 0.4
  times), and 50 steps at it didn't diverge (one seed, a fixed rate from the start). That's the
  range test pointing at a rate on the falling part of the curve; 50 steps don't show it holds for
  a whole run.
- It isn't better than the scripts' own rates here. LoRA's two runs see the same batches and their
  losses nearly coincide step by step: at 0.8 times the author's rate, 50 steps can't tell them
  apart. nanoGPT's loss falls more slowly at first at the suggested rate and ends at the same level
  (6.01 against 6.03). Both zoo scripts already have well-chosen rates; the test's
  value is for a script whose rate is far off, which neither of these is. A planted bad rate would
  show that; not run.
- The divergence point is where to stop: LoRA blows up about 75 times above its own rate. nanoGPT
  met the divergence rule only at the ramp's last step (1.0), but its curve was already unstable
  from about 0.4 (losses of 13 to 35 between 0.38 and 0.61), so the reading of its divergence is
  late; a rule on the curve's rise, not just 4 times the best, would catch it earlier.
- Read as Smith and fastai do: a starting rate for the script's own schedule, not a measured
  optimum. Two scripts on one GPU is a small check.

## A's review (Nov 2) and what changed

A reviewed `f909ac3` (`from-a.md`, ws-a): ready to merge off, six things before the flag goes on.
The fixes then had their own read-only review, whose two blocking findings are fixed too.

1. **"Checked" proved nothing.** It compared the rate the optimizer held with the one the
   recorder had just written, so it always matched, even for an optimizer that ignores its rate.
   The table above says "the ramp checked" under that old test; it doesn't show the rate took
   effect. Now `follows_rate` (`probes/lr_range.py`) reads the step's effect: the least-squares
   slope of log(update size) on log(rate), up to the divergence. The update size is the summed
   change of up to four of the ramp's own optimizer's weights, none that start at zero (a ratio
   to the weight went flat on LoRA's B and zero-init projections, whose weight is the updates'
   own sum: the second review's blocking finding). At least 0.5 is `lr_range.checked`; below
   is `lr_range.not_set`; fewer than 8 measured steps or under two decades of rate is
   `lr_range.unverified`. Only `checked` keeps the rates; the others keep the losses alone. On
   CPU: Adam, AdamW with zero-init weights, a tensor rate and a fused Adam read as checked; a
   fixed-step optimizer and transformers' Adafactor with `relative_step` read as not set.
   Weights kept in bf16 with no fp32 copy round small updates to zero, so a ramp on them may read
   as unverified.
2. **Group ratios under a warm-up from 0:** the ratio is read from `initial_lr` when a scheduler
   set it, and a hand-written warm-up from exactly 0 is read again at its next step (when the script
   writes its rate before each step, as such warm-ups do) (a CPU test
   with two groups at 1e-3 and 2.5e-4 under a warm-up from 0 holds 0.25 every step; it read 1.0
   before).
3. **A second optimizer** holding a tensor the ramp's doesn't (Muon beside AdamW, a GAN's two,
   even with a shared encoder, a script's plain-tensor weights) is counted as it steps:
   `lr_range.other_optimizers`, and the rates are dropped (an attack's perturbation counts too,
   on the safe side). One over only the
   ramp's weights that steps instead of it means the script rebuilt its optimizer: `not_set`,
   and the probe stops.
4. **`p1-lr` reads as an ordinary ok probe** in the engine's filters: asked of B (the engine's
   owner) in `from-d.md`. Harmless while its measured fields are None, which they are.
5. **Contract drift:** the golden pretrain record (`scripts/make_golden.py`, A's, and the schema
   example and web fixture made from it) has `lr_range_divergence_lr` on an ordinary probe and a
   finding `lr.headroom`; the code writes `lr_range.diverged_lr` on `p1-lr`. Asked of A.
6. **Smaller:** a GradScaler that skips every step (fp16 overflow) no longer runs the probe to its
   timeout: a scaler halves its scale at each skip, so 8 halvings with no step (24 before the
   first, past its start-up skips) stop the probe (`lr_range.stalled`; the last rate applied is
   the divergence, when the rates stand). Long accumulation with a scaler (64 backwards a step) is
   not a stall. Many more backwards than any step took while another optimizer steps instead
   (a rebuilt or an unfrozen optimizer) also stops it, keeping the curve so far. The point's timeout is capped at what the job
   has left, in time and dollars. A `p1` answered from the cache now gets its range test too (it
   was skipped). The check app reads its single-rate runs' losses directly (they keep no rates,
   so the curve was empty: the second review's other blocking finding) and no longer prints
   whole loss lists.

Tests added for each, and for the cache key unchanged with no range test, a budget refusal, the
dollar cap and `nproc > 1` (that guard was already there).

**The GPU check re-run with the new test** (`d-lr-range-gpu-2`, GitHub 41528070349, Nov 2, 21:24
UTC, `ws-d` `7263013`, an A10 both times) [measured]: both scripts read **checked** now under the
real test (the updates grew with the ramp). Suggestions as before: LoRA 7.9e-5 (own 1e-4),
nanoGPT 2.5e-4 (own 6e-4). Divergence read at 1.2e-2 for LoRA (7.6e-3 before) and 0.72 for
nanoGPT (1.0 before): two runs each, one seed per run: the divergence point moved 1.6 times
(LoRA) and 1.4 times (nanoGPT) between them. 50 steps at the suggestion didn't diverge on either (LoRA 1.283 to 1.216, nanoGPT 8.74 to
6.01), nor at the scripts' own rates. The confirmation runs' losses now come back (they were empty
after the first round of fixes). Cost: US$0.18 for 596 s of A10 containers [derived: container
time at list price]; item 3's spend is US$0.37 of its US$2.

**A planted bad rate** (Chiya, Nov 2, 17:33; `d-lr-range-planted-1e-3`, GitHub 41538650072, an
A10, US$0.14 [derived]) [measured]: the LoRA script with `learning_rate` 1e-3, ten times its
author's. The reading is the same as on the unchanged script (suggested 7.9e-5, smoothed lowest loss at 7.7e-4 (the raw curve's lowest point is at 3.4e-4; it's noisy),
divergence 1.0e-2, checked); the script's 1e-3 sits past the smoothed curve's lowest point, 12.6 times the
suggestion. 100 steps held at each: neither diverged; at 1e-3 the loss stayed above the suggested
rate's in every 10-step block after the first, by 0.01 to 0.05 (one seed, the same batches). So it
catches the rate as too high and suggests a sane one; what the too-high rate cost in 100 steps was
a slower fall, not a blow-up. This run: US$0.14 of its US$0.50 cap; the range test's three GPU runs
total US$0.51. Video: shot 7b in `docs/demo/video-plan.md`.

## What's next

- A's re-check of the fixes.
- B: read `lr_range.suggested_lr`, `lr_range.diverged_lr` and the curve (`lr_range.lr.NNN`,
  `lr_range.loss.NNN`, `probes.lr_range.curve()`) into the guide; C: draw the curve with the
  suggested and diverged rates marked. Both asked in `from-d.md`.
- The flag stays off until A redeploys the runner with it and the PM turns it on. The probe runs
  in the sandbox like any other; its curve goes into the record, so whatever reads the record (the
  live guide on Muse Spark, Meta's API) reads the curve too. That's the same kind of data as the
  probes' measurements the data-use notice already covers, so C should check the notice's wording
  covers it before the flag goes on. It adds one probe's GPU time to the job (2 to 3 minutes
  here), inside the job's budget.
