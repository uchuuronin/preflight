# The 3-minute video: shot list and voice-over (filmed Tue Nov 3, evening)

By D, Mon Nov 2, overnight (Chiya's list, item 3). This is the plan of record for Tuesday. It
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
| 4 | 0:32 | Boarding: the stages light in turn; hold on "Reading your script" with its tag, then "Checking the logs" with its tag | "Muse Spark reads the script and finds the training loop. A few real steps run on a real GPU, and Muse Spark Open reads every probe's log." | "Muse Spark (Meta)" while "Reading your script" lights; "Muse Spark Open (Meta's open model, on Baseten)" while "Checking the logs" lights | **Works now** for Muse Spark (tags come from the record's provenance since finding 12's fix). Spark Open's tag shows only if that check-in's probes were triaged: true on the live site, where `harness.spark_open_triage` is on. Film a fresh check-in and cut the GPU wait (D's A10 probe waited 84 s on Nov 2, `record_nov2_a10.json`; L4 waits have run from 8 to 59 min in A's and B's runs) |
| 5 | 0:47 | Cockpit of the Oct 28 record (`ICDYrfHQ2nBNbv3mlRHHMg`): the map draws, the memory ring fills past the line, the red finding "runs out of memory" | "In 189 seconds of probing, for about four cents of GPU at list price, it caught the crash before the real run." | "189 s of probing · $0.042 of GPU at list price · 20.4 of 22.06 GiB · one script" | **Needs by Tuesday:** that share link opening on the live site (A's record; check it with the browser before filming). If it doesn't, use A's fresh production record from shot 4 and re-take every spoken number from it |
| 6 | 1:02 | Click the Eval loop node: its detail sheet shows the rule checks' findings and the triage finding with its quoted log line | "Muse Spark Open says what went wrong, and quotes the line it read; code checks the quote is really in the log." | "Muse Spark Open (Meta's open model, on Baseten)" over the app's own "Found by Muse Spark Open triage" | **Needs by Tuesday:** A's production LoRA triage re-run (since `b300684`) posted, with its finding quoting the out-of-memory line. If not, cut and give 12 s to shot 7 |
| 7 | 1:14 | The knob dock: drag "Eval batch" down; the ring drops under the line, the dials move; press "Possible outcomes" | "Move a setting and the whole run answers at once, with its range." | the app's "Predicted" on the dials | **Works now** (`c.hops` on in `flags.production.json`) |
| 8 | 1:26 | The guide panel: "Start the tour"; the guide proposes one knob move and moves it on the map; Accept | "Ask the guide, and Muse Spark moves the knobs that matter for your goal, one at a time, and you decide each move." | "Muse Spark (Meta)" | **Works now** (`c.guide.live` on). If the live guide is refused (its daily cap), the panel falls back to the rule-based tour: then drop the label and the line |
| 9 | 1:40 | Scroll to the exports: the Slurm file, the run command, the Weights & Biases helper | "Take the plan with you: a batch file, the command, and your settings and predictions in Weights and Biases." | | **Works now** (`c.exports` on in `flags.production.json`) |
| 10 | 1:50 | Hard cut to the agent sample measured on Oct 31 (PF331): the agent map, then the caching finding open | "Not every job is a training run. Here a coding agent on Muse Spark ran 130 programming tasks through Preflight's proxy, and not one input token came from the cache." | "Agent eval on Muse Spark (Meta) · measured" · "0 of 905,739 input tokens from cache, 130 tasks, two runs" | **Works now** (sample PF331 says Measured; its record holds 0 of 246,463 for 30 tasks; the 130-task total is A's status, so it's a caption, not the screen) |
| 11 | 2:05 | The GRPO sample measured on an L4 (PF510): the RL band | "In this GRPO run, 15 of 20 prompts had no reward spread at all, so they teach nothing." | "GRPO on an L4 · measured · 15 of 20 prompts with no reward spread" | **Works now** (sample PF510 says Measured; `engine.rl` on in `flags.production.json`) |
| 12 | 2:15 | New flight with "Private mode" ticked; then the record of a private check-in: its triage finding | "For code that mustn't leave, private mode triages on Muse Glimmer, Meta's open weights, on our own GPU." | "Muse Glimmer (Meta's open weights, on Preflight's own GPU)" | **Needs by Tuesday:** `harness.glimmer_private` on in production (the tick only shows with it), the runner holding Glimmer's secret, and one private check-in whose record shows a Glimmer triage finding. Glimmer's cold start is 5 to 8 min (A's runs), so check in ahead and film the record. If not ready, replace it with a captioned card over the agent sample: "The same 26 tasks: Muse Spark 23, Muse Glimmer 15, Muse Spark Open 14 ended on a tool call (measured, an upper bound on passes)", labelled "Muse Glimmer (Meta's open weights, on our own GPU)", and the line "We ran the same agent on Meta's open weights too, on our own GPU." (A's `a-pareto-*` runs; the card is a measured result, not a live-site flow) |
| 13 | 2:27 | Checks page: scroll "Each check": the caught OOM and the misses | "Every prediction is checked against real runs, misses included." | "Held-out checks on real runs" | **Needs by Tuesday:** B's scorer run on Mon Nov 2 and the page as deployed. If its checks still say "not measured", cut the shot and give its 10 s to shot 14 |
| 14 | 2:37 | Copy the cockpit's address, open it in a new tab: the same flight opens | "Every flight is a link anyone can open." | | **Works now** (the link is the page's address). The share page's preview card needs `workers/` in the deploy; not shown here |
| 15 | 2:45 | End card, about 10 s | "Preflight. Fly the run before you pay for it." | link · repository · "Built Oct 26 to Nov 4" | **Works now** |

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
