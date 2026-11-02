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
| 50 steps at the suggested rate | trains: loss 1.287 to 1.216 (first 10 steps' mean to last 10's), no divergence | trains: 8.74 to 6.01, no divergence |
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

## What's next

- A's review (asked in `docs/requests/from-d.md`): it touches the recorder and the pipeline.
- B: read `lr_range.suggested_lr`, `lr_range.diverged_lr` and the curve (`lr_range.lr.NNN`,
  `lr_range.loss.NNN`, `probes.lr_range.curve()`) into the guide; C: draw the curve with the
  suggested and diverged rates marked. Both asked in `from-d.md`.
- The flag stays off until A redeploys the runner with it and the PM turns it on. The probe runs
  in the sandbox like any other; its curve goes into the record, so whatever reads the record (the
  live guide on Muse Spark, Meta's API) reads the curve too. That's the same kind of data as the
  probes' measurements the data-use notice already covers, so C should check the notice's wording
  covers it before the flag goes on. It adds one probe's GPU time to the job (2 to 3 minutes
  here), inside the job's budget.
