# The 3-minute video: shot list and voice-over (filmed Tue Nov 3, evening)

By D, Mon Nov 2, overnight (Chiya's list, item 3); statuses updated Mon Nov 2, 11:10 ET; shot 9b added 12:05 ET, its stopgap step replaced 16:30 ET; shot 7b added 17:45 ET. This is the plan of record for Tuesday. It
replaces the plan A and plan B shot lists in `storyboard.md` and `fallback-cut.md`, which stay as
the earlier drafts; the 45-second cut in `storyboard.md` is unchanged.

## Rules

- **Only features built Oct 26 to Nov 4, 2026,** and the video says so on screen ("Built Oct 26 to
  Nov 4").
- **Every Meta model is labelled on screen at the moment it works,** and only where a record shows
  it did that work: Muse Spark (Meta), Muse Spark Open (Meta's open model, on Baseten), Muse Glimmer
  (Meta's open weights, on our own GPU).
- **Live-site flows only** (`https://preflight-cqn.pages.dev`), with real records; never a sample
  that says "Illustrative" while the narration talks about measurements.
- **Spoken or captioned numbers come from the submission's numbers table**
  (`docs/submission/README.md`), each measured, or measured time or tokens priced at list rates and
  said "at list price". Predicted numbers may be on screen with the app's own label, never spoken.
- **A shot that isn't ready by Tue Nov 3, noon ET is cut, not faked.** Its time goes to the shot
  named in its row.

## Shots

Narration at about 2.5 words a second; every line is also a burned-in caption. About 2:55.

| # | Time | Screen and action | Voice-over | On-screen label | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | 0:00 | `zoo/finetune_lora_hf/train.py` in an editor; slow zoom to `EVAL_EVERY = 200` and the eval batch line | "This fine-tune looks fine. At step 200, its first evaluation, it runs out of memory." | "A real training script, with a planted bug" | **Works now** (an editor, no app) |
| 2 | 0:10 | Live site, home; the title fades in; click "Start a preflight" | "Preflight flies your training run before you pay for it." | "Built Oct 26 to Nov 4" | **Works now** |
| 3 | 0:18 | New flight: paste the repo link of `zoo/finetune_lora_hf`, pick "Finish soonest" and L4, scroll the data-use notice, tick "Got it", press "Check in" | "Point it at your code and say what matters. It tells you exactly what it sends." | | **Works now** (`runner.queue` on in production; the check-in flow passes D's stranger audit at both sizes and themes) |
| 4 | 0:32 | Boarding: the stages light in turn; hold on "Reading your script" with its tag, then "Checking the logs" with its tag | "Muse Spark reads the script and finds the training loop. A few real steps run on a real GPU, and Muse Spark Open reads every probe's log." | "Muse Spark (Meta)" while "Reading your script" lights; "Muse Spark Open (Meta's open model, on Baseten)" while "Checking the logs" lights | **Works now** for Muse Spark (tags come from the record's provenance since finding 12's fix). Spark Open's tag shows only if that check-in's probes were triaged: true on the live site, where `harness.spark_open_triage` is on. Film a fresh check-in and cut the GPU wait (D's A10 probe waited 84 s on Nov 2, `record_nov2_a10.json`; L4 waits have run from 8 to 59 min in A's and B's runs; since A's capacity fix, a job that can't get an L4 falls back to an A10, with the wait shown (Chiya, Nov 2)) |
| 5 | 0:47 | Cockpit of the Oct 28 record (`ICDYrfHQ2nBNbv3mlRHHMg`): the map draws, the memory ring fills past the line, the red finding "runs out of memory" | "In 189 seconds of probing, for about four cents of GPU at list price, it caught the crash before the real run." | "189 s of probing · $0.042 of GPU at list price · 20.4 of 22.06 GiB · one script" | **Needs by Tuesday:** that record opening on the live site (check it with the browser before filming). Once C confirms `workers/` is deployed (today, a preview first), open it from its share page, `/s/ICDYrfHQ2nBNbv3mlRHHMg`. If it doesn't open, use A's fresh production record from shot 4 and re-take every spoken number from it |
| 6 | 1:02 | Click the Eval loop node: its detail sheet shows the rule checks' findings and the triage finding with its quoted log line | "Muse Spark Open says what went wrong, and quotes the line it read; code checks the quote is really in the log." | "Muse Spark Open (Meta's open model, on Baseten)" over the app's own "Found by Muse Spark Open triage" | **Needs by Tuesday noon:** A's production LoRA triage re-run, with its finding quoting the out-of-memory line (A, by Tue Nov 3, noon ET; Chiya, Nov 2). If not, cut and give 12 s to shot 7 |
| 7 | 1:14 | The knob dock: drag "Eval batch" down; the ring drops under the line, the dials move; press "Possible outcomes" | "Move a setting and the whole run answers at once, with its range." | the app's "Predicted" on the dials | **Works now** (`c.hops` on in `flags.production.json`) |
| 7b | (1:24) | The LoRA script checked in with `learning_rate=1e-3` (ten times its author's 1e-4); its cockpit's range-test curve: loss against learning rate, the script's rate marked past the curve's lowest point, the suggested rate marked lower down | "Set the learning rate ten times too high and a hundred quick steps on a rising rate show it: past the bottom of the curve. Preflight suggests one where the loss falls fastest." | "Learning-rate range test · measured on an A10" | **Needs by Tuesday noon:** B's re-check of the range test's fixes, `probes.lr_range` on and the runner redeployed (A, the PM), and C's curve in the cockpit; the GPU check passed (section below). About 10 s; it takes 9b's slot if 9b is cut, otherwise it's not in the 3-minute cut |
| 8 | 1:26 | The guide panel: "Start the tour"; the guide proposes one knob move and moves it on the map; Accept | "Ask the guide, and Muse Spark moves the knobs that matter for your goal, one at a time, and you decide each move." | "Muse Spark (Meta)" | **Works now** (`c.guide.live` on). If the live guide is refused (its daily cap), the panel falls back to the rule-based tour: then drop the label and the line |
| 9 | 1:40 | Scroll to the exports: the Slurm file, the run command, the Weights & Biases helper | "Take the plan with you: a batch file, the command, and your settings and predictions in Weights and Biases." | | **Works now** (`c.exports` on in `flags.production.json`) |
| 9b | (1:50) | Terminal beside the cockpit: `preflight watch` running the real script; the cockpit's in-flight panel follows the live record; an alert line appears in both | "Run it for real and Preflight watches. The moment the run leaves its predicted band, it says so." | the app's "Predicted" band; no Meta model (no model call here) | **Needs B's GPU test.** `c.inflight` is on in production (PM, 4:25 PM). `watch` runs and streams on a CPU; no alert fired then because `preflight run` filled no predictions, which B has fixed (on `main`; steps below). Goes in only if ready by Tue Nov 3, noon ET, taking its 10 s from shot 13 if that's cut; otherwise not in the 3-minute cut |
| 10 | 1:50 | Hard cut to the agent sample measured on Oct 31 (PF331): the agent map, then the caching finding open | "Not every job is a training run. Here a coding agent on Muse Spark ran 130 programming tasks through Preflight's proxy, and not one input token came from the cache." | "Agent eval on Muse Spark (Meta) · measured" · "0 of 905,739 input tokens from cache, 130 tasks, two runs" | **Works now** (sample PF331 says Measured; its record holds 0 of 246,463 for 30 tasks; the 130-task total is A's status, so it's a caption, not the screen) |
| 11 | 2:05 | The GRPO sample measured on an L4 (PF510): the RL band | "In this GRPO run, 15 of 20 prompts had no reward spread at all, so they teach nothing." | "GRPO on an L4 · measured · 15 of 20 prompts with no reward spread" | **Works now** (sample PF510 says Measured; `engine.rl` on in `flags.production.json`) |
| 12 | 2:15 | New flight with "Private mode" ticked; then the record of a private check-in: its triage finding | "For code that mustn't leave, private mode triages on Muse Glimmer, Meta's open weights, on our own GPU." | "Muse Glimmer (Meta's open weights, on Preflight's own GPU)" | **Needs by Tuesday:** one private check-in whose record shows a Glimmer triage finding; C posts its record id after today's deploy (Chiya, Nov 2). Done already: `harness.glimmer_private` is on in `main` (the PM's 5:15 merge, with C's data-use line) and in A's runner, which holds Glimmer's secret; the "Private mode" tick reaches the live site with C's deploy today. Glimmer's cold start is 5 to 8 min (A's runs), so check in ahead and film the record. If not ready, replace it with a captioned card over the agent sample: "The same 26 tasks: Muse Spark 23, Muse Glimmer 15, Muse Spark Open 14 ended on a tool call (measured, an upper bound on passes)", labelled "Muse Glimmer (Meta's open weights, on our own GPU)", and the line "We ran the same agent on Meta's open weights too, on our own GPU." (A's `a-pareto-*` runs; the card is a measured result, not a live-site flow) |
| 13 | 2:27 | Checks page: scroll "Each check": the caught OOM and the misses | "Every prediction is checked against real runs, misses included." | "Held-out checks on real runs" | **Needs by Tuesday:** B regenerates the validation table once the unseen runs are scored (likely Mon Nov 2, night). If its checks still say "not measured", cut the shot and give its 10 s to shot 14 |
| 14 | 2:37 | Copy the cockpit's address, open it in a new tab: the same flight opens | "Every flight is a link anyone can open." | | **Works now** (the link is the page's address). The share page's preview card needs `workers/` in the deploy; not shown here |
| 15 | 2:45 | End card, about 10 s | "Preflight. Fly the run before you pay for it." | link · repository · "Built Oct 26 to Nov 4" | **Works now** |

## Shot 7b: a planted bad learning rate (Chiya, Nov 2, 17:33)

`d-lr-range-planted-1e-3` (GitHub 41538650072, Nov 2, 22:36 UTC, `ws-d` `9b6ddb1`, an A10, US$0.14
[derived: 469 s of container at list price]) [measured]: the LoRA zoo script with its
`learning_rate=1e-4` changed to `1e-3`, then the range test and 100 steps held at each rate.

- **It catches it.** The range test reads the curve the same way as on the unchanged script (the
  ramp sets the rate, not the script): suggested 7.9e-5, lowest loss at 7.7e-4, divergence at
  1.0e-2, and the ramp checked. The script's 1e-3 is **past the curve's lowest point**, 12.6 times
  the suggestion and 10 times below where it diverges.
- **The suggestion is sane:** 7.9e-5 is 0.8 times the author's own 1e-4.
- **What 1e-3 costs here, honestly:** not a blow-up. Over 100 steps on the same batches, the loss
  at 1e-3 sat above the suggested rate's in every 10-step block after the first, by 0.02 to 0.05
  (last 10 steps' mean 1.225 against 1.174); neither diverged. So the beat says "past the bottom
  of the curve", not "it would crash", and shows no loss numbers spoken.

What the screen needs: a hosted record of that check-in with `probes.lr_range` on (its probe
`p1-lr`) and C's drawing of the curve with the script's rate and the suggestion marked. Until both
exist, the beat isn't filmable.

## Shot 9b: the in-flight steps (Chiya's list, Nov 2, item 2)

**What D checked, on a CPU** (`scripts/d_watch_cpu_check.py`, run `d-watch-cpu-check`, GitHub
41485598315, a GitHub runner with 2 CPUs and no GPU, torch 2.15.0 (its CUDA build, running on the
CPU), Nov 2; no spend) [measured]:

- `preflight run zoo/notebook_tiny/tiny_train.py`, then `preflight watch` on its record: both exit
  0; the live record streams (200 steps, 10 windows of 20) and ends `ok`.
- With `max_steps` raised to 50,000: 28.6 s of run, 2,500 windows, about 0.6 ms a step.
- That record watched against the same script with `batch_size` 64 raised to 1024: 68.8 s, about
  2.4 times as long, **and no alert**.

**Why no alert (before B's fix):** the record had no `time.step`, `time.total` or `memory.peak` estimate, and
`watch` alerts only against those. That isn't the CPU: `preflight run` never fills predictions
(`cli/main.py` writes the measured record; only the hosted runner calls `with_predictions`), so no
record the CLI makes has bands, on any device (found by D's read-only review; asked of A in
`from-d.md`). On a laptop today the shot can show the live record following the run, not an alert.

**For the alert** (B's test first; Chiya films the same steps on a machine with an NVIDIA GPU, or
the shot is cut). From a clone of the repository:

1. `pip install -e ".[probe]"` (the repository isn't public until Nov 3 to 4; after that,
   `pip install "preflight[probe] @ git+https://github.com/uchuuronin/preflight"`).
2. `mkdir film && cp zoo/notebook_tiny/tiny_train.py film/`, and in `film/tiny_train.py` set
   `max_steps = 200000` (a guess for about a minute on a GPU; B's test sets the real number).
3. `preflight run film/tiny_train.py`: it prints the record's path, `records/<share>.json`.
4. Check the record has bands (`preflight run` fills them since B's fix, on `main` from the
   PM's 4:25 PM merge, Nov 2):
   `python -c "import sys; from preflight.schema import load_record; r = load_record(open(sys.argv[1]).read()); print({k: (e.status, e.lo, e.hi) for k, e in r.predictions.estimates.items() if k in ('time.step', 'memory.peak')})" records/<share>.json`.
   If neither `time.step` nor `memory.peak` has a range, stop: no alert can fire. On a CPU (D,
   Nov 2, 16:30 ET) `memory.peak` had one (604 to 978 MiB) and `time.step` abstained.
5. Make a change the prediction didn't see, big enough to move the GPU's memory and step time
   (a tiny model's batch alone may not: memory is reserved in 2 MiB blocks and launch time
   dominates): in `film/tiny_train.py`, `nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 1)` to
   `nn.Linear(32, 8192), nn.ReLU(), nn.Linear(8192, 1)` and `batch_size = 64` to
   `batch_size = 4096`. B's test confirms which change fires.
6. Open `https://preflight-cqn.pages.dev` in Chrome or Edge (they follow the file; other browsers
   read it once), go to the Logbook and open "Watch a real run" (needs `c.inflight` on in
   production).
7. In a terminal beside it: `preflight watch records/<share>.json film/tiny_train.py --window 200`.
   It prints the live record's path (`records/<new>.json`). In the browser, choose that file (not
   the `.live.jsonl`); the cockpit follows it, rereading every 3 s.
8. Expect `[preflight] step N: peak memory ... GiB, above the predicted ...` after the first
   window, and `... s a step, above the predicted ...` after two windows running; the in-flight
   panel shows the same alert. Film from step 7 to the alert.

**Needs, in order:** B's GPU test (do steps 1 to 8 give bands and an alert, which change fires,
how many steps make about a minute); `c.inflight` on in production (on since the PM's 4:25 PM
merge, Nov 2; live once C deploys); then Chiya's take. The stopgap step is gone: `preflight run`
fills predictions (B's fix, on `main`).

## Before filming (Tue Nov 3, evening)

- [ ] Each "needs" row above is ready, or cut as its row says.
- [ ] Every number spoken or captioned is in the submission's numbers table.
- [ ] Each Meta model named is labelled on screen while it works, and its record shows the call.
- [ ] No sample that says "Illustrative" is on screen while the narration talks about
  measurements.
- [ ] Captions burned in for every line, at least 2 s each, white on a dark band.
- [ ] The repository is public and the end card's link opens.

## What is left out, and why

- **The fix loop (Muse Code):** `c.fix_loop` is off in production, and gating a fix on the re-probe
  isn't wired end to end, so it's not a live-site flow today.
- **The campaign calendar:** on in production, but its sample is illustrative; three minutes don't
  hold it with the stories above. It's in the README.
- **The divergence warning:** its classifier misses (AUROC 0.66 against 0.85) and isn't used.
