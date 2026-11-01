# Fallback cut of the 3-minute video (plan B)

Plan A is the plan's video (the GRPO pain, the guide's tour, the OOM frontier, Muse Code's
race, the agent Pareto plot, campaign mode, the validation table) with `storyboard.md` for its
fine-tune shots. Use this cut on Tue Nov 3 if the GRPO, Pareto, race or campaign beats aren't
ready, or aren't built for real, by then. It stands on two measured runs: the planted-bug LoRA
fine-tune and the agent eval on Muse Spark. Nothing in it needs a beat that isn't built today.

## Rules (the same as plan A's)

- **Only measured numbers, and measured time or tokens priced at list rates (next rule), are
  spoken or captioned,** each from the table below. Predicted or other derived numbers may show
  on screen where the app labels them ("Predicted", "derived"), but the narration never says
  them.
- **Dollar figures are spoken and captioned "at list price"** (Chiya, Nov 1): measured time or
  tokens priced at list rates, e.g. "$0.042 of GPU at list price". The README keeps tagging them
  derived; in the video the caption's "at list price" says the same thing.
- **"One script" or "one run" stays on screen whenever a number is.** These are examples, not
  rates.
- **Only our work, built Oct 26 to Nov 4,** and every Meta model named only where a record
  shows it did that work.
- **Nothing from Chiya's ongoing research.**
- **Record on the deployed app with the real records,** never a sample (samples say
  "Illustrative").
- **The Spark Open triage shot (11b) is a real production record or nothing** (Chiya, Nov 1).
  A posts its share link on Sun Nov 1. If it isn't there by Tue Nov 3 at noon ET, the shot is
  dropped, not faked.

## Numbers this cut may use

| Number | What it is | Source |
| --- | --- | --- |
| step 200 | where the real fine-tune crashes: its first evaluation | the script (`EVAL_EVERY = 200`), and measured: A's full run `a-gt-lora-l4-5` (Oct 31, `docs/status/a.md`) ran out of memory at step 200 |
| 189 s | probe time until the eval out-of-memory was caught (probe `p1-eval`); upload to record took 6 min 45 s in all | measured, A's status, record `ICDYrfHQ2nBNbv3mlRHHMg` |
| 20.4 of 22.06 GiB | memory in use at the probe's peak when the eval ran out, on the L4 | measured: 21.9 GB (20.4 GiB) at the peak (A's status; the record's `p1-eval` probe) and 23,689,363,456 bytes of L4 (the record's env), in GiB as the app shows memory |
| 130 tasks, in two runs | MBPP tasks run by the agent through Preflight's proxy | measured: `a-agent-sample-1` (30 tasks, Oct 31) and `a-agent-builds-1` (100 more, Nov 1), A's status |
| 30 tasks in 49 s | the sample the Oct 31 record holds, and its script wall time | measured, A's status (`a-agent-sample-1`) and the record's `agent_sample` |
| $0.042 | GPU cost of the fine-tune probe, captioned "$0.042 of GPU at list price" | measured probe time x the $0.80/h L4 list price, A's status |
| $0.43 | Muse Spark calls for the 30-task sample on screen, captioned "at list price" | the proxy's token counts at the price table's rate: $0.4289, A's status (`a-agent-sample-1`); the record's 30 per-task costs sum to $0.428888 |
| $1.67 | Muse Spark calls for all 130 tasks, captioned "at list price" | the same, $0.4289 + $1.2371 = $1.666, A's status (`a-agent-sample-1`, `a-agent-builds-1`) |
| 0 of 905,739 | input tokens served from cache over those 130 tasks | measured: 0 of 246,463 (the 30, shown in the Oct 31 record) and 0 of 659,276 (the 100), A's status |
| 0 of 246,463 | the same, for the 30 tasks the Oct 31 record holds; this is the one the app shows | measured, the record's `agent.no_prompt_caching` finding |

**Not used, and why:**
- **The retry loop as a measured stall.** It never fired in the 130 tasks: A measured 0
  runaways in 130 (one-sided 95% upper bound 2.3%), because no sampled build failed its tests.
  Plant 3 now fires on rows 26 to 29 (D, `6215d1b`), but A hasn't re-sampled yet. Until a record
  shows the stall, the cut shows the loop only as the app finds it: read from the code
  (finding `read.agent.tool_retry_loop`; its detail cites `run.py:121` of the script as the record
  read it, and `while True` is line 123 of today's file). If A's re-sample shows the stall before
  Tuesday, add its numbers here with their run ID and change shot 9's line.
- **The context overflow at turn 11.** It is derived from measured growth (2,745 tokens a turn),
  and no sampled task got past 4 turns. It may show on screen under the app's "derived" label,
  but it isn't spoken.
- **The full run's cost.** The app shows the record's prediction, $7.15 (5.90 to 13.71,
  "Predicted"); it may stay on screen with that label but isn't spoken or captioned. A's derived
  "about $6.4" (the 130 tasks' mean x 500) is in A's status file only and isn't used.
- The derived "about 6 minutes and $0.09" the fine-tune crash would have cost, any hours or
  monthly savings, and any accuracy rate.

## Shots

About 2:58. Narration at about 2.5 words a second; every line is also a caption.

| # | Time | Screen | What moves | Narration | On-screen label |
| --- | --- | --- | --- | --- | --- |
| 1 | 0:00 | `zoo/finetune_lora_hf/train.py` in an editor, cursor on `EVAL_EVERY = 200` | Slow zoom to the eval batch line | "This fine-tune looks fine. At step 200, its first evaluation, it runs out of memory." | "A real training script, with a planted bug" |
| 2 | 0:10 | Preflight home | Title fades in; cursor on "Start a preflight" | "Preflight flies the run before you pay for it." | "Built Oct 26 to Nov 4" |
| 3 | 0:18 | Check-in, then Boarding (plan A's shots 3 to 6, cut to 20 s) | Paste the repo link, pick "Finish soonest" and L4; the stages light in turn | "Point it at the code. It reads the script, then runs a few real steps on a real GPU." | "Muse Spark (Meta)" only while its stage lights, and only if the record shows the call |
| 4 | 0:38 | Cockpit of the Oct 28 fine-tune record | The map draws; the memory ring fills past the line; the red finding appears | "In 189 seconds of probing, for about four cents of GPU at list price, it caught the crash: the evaluation reaches 20.4 of 22.06 gibibytes and runs out." | "189 s of probing · $0.042 of GPU at list price · 20.4 of 22.06 GiB · one script" |
| 5 | 0:58 | Same cockpit, finding open, then the knob dock | Open the finding; drag eval batch size down; the ring and dials move together | "It says why, and which setting fixes it. Move a setting and the whole run answers at once." | "Predicted" (the app's own label on the dials) |
| 6 | 1:15 | Hard cut to the agent record's cockpit (Oct 31) | The agent map draws: model, tools, the context tank | "Not every job is a training run. This one is a coding agent on Muse Spark, solving programming tasks." | "Agent eval on Muse Spark (Meta) · one run" |
| 7 | 1:28 | Same cockpit, the caching finding open | Hold on "0 of 246,463 input tokens were served from cache" | "Preflight sent its calls through a proxy and counted their tokens. Across 130 tasks in two runs, not one input token came from the cache." | "On screen: 0 of 246,463, the 30-task sample · all 130 tasks, two runs: 0 of 905,739" |
| 8 | 1:45 | The finding's detail ("A prefix that changes every call ... defeats the provider's cache"), then `run.py`'s `system_prompt()` in the editor | Zoom to `f"Current time: {now}\n"` | "In the code, the prompt starts with the current time, so its beginning is different on every call." | "From the script: `run.py`, `system_prompt()`" |
| 9 | 1:55 | Same cockpit, the retry-loop finding (red) | Open it; the detail shows `run.py:121: while True / except: continue` | "It also reads the code: one tool retries forever, out of the model's sight." | "Read from the code" |
| ~~10~~ | | Same cockpit, the context tank and cost readouts | Cut for 11b's time; back only if 11b is dropped (below) | | |
| 11 | 2:07 | A plain card over the cockpit, dimmed | Two lines fade in | "Everything on this screen came from a 30-task sample that ran in under a minute, for 43 cents of Muse Spark calls at list price. All 130 tasks, in two runs, cost a dollar sixty-seven at list price." | "30 tasks · 49 s · $0.43 of Muse Spark calls at list price (`a-agent-sample-1`) · all 130 tasks, two runs: $1.67 at list price" |
| 11b | 2:26 | A's production record from its share link (posted by Tue Nov 3, noon ET), on the screen where the app shows its triage finding. **Not filmable on `main` yet:** the cockpit shows no finding without a node, and triage's has none (dependency below) | Hold on the finding: its title, its one sentence with the quoted log line, and the app's own "Found by Muse Spark Open triage" (`finding.source.spark_open_triage`) | "When a probe fails on the live site, Muse Spark Open reads the end of its log and quotes the line." | "Muse Spark Open (Meta, open weights, on Baseten) · triage of a failed probe's log · a real job on the live site" |
| 12 | 2:36 | Checks page | Scroll the "Each check" list: the caught OOM and the misses | "Every prediction is checked against real runs, misses included." | "Held-out checks on real runs" |
| 13 | 2:46 | End card (ends near 2:58) | Public link and repo | "Preflight. Fly the run before you pay for it." | link · repo · "Built Oct 26 to Nov 4" |

Narration is about 233 words, about 93 s of speech in 2:58; shot 11 is the fullest (38 words
written, about 41 spoken, in 19 s: its 16 s plus 3 s left from shot 10). Shot 11b speaks no number. If 11b is dropped, shot 10
comes back at 2:07 and shot 11 runs 2:20 to 2:36, as before; at 16 s its line is about 2.6 words
a second, so trim "in two runs" if the take runs long.

Spoken numbers: step 200, 189 seconds, about four cents of GPU at list price, 20.4 of
22.06 GiB, 130 tasks in two runs, none of the cached tokens, 30 tasks in under a minute, 43 cents
and $1.67 of Muse Spark calls at list price. Each is in the table above.

## Each shot's dependency, and its fallback

| Shot | Needs | If it isn't ready when we record |
| --- | --- | --- |
| 3 | check-in open on the deploy (`runner.queue`) and the runner up, for a live check-in | Cut it and open shot 4 with the caption "checked in on Oct 28"; give the 20 s to shots 6 to 9 |
| 3 | Muse Spark's script-map call in the recorded run (`harness.spark_script_map`) | Crop the fixed "Muse Spark" tag on Boarding and drop "It reads the script" (C is asked to tag only models that ran: finding 12) |
| 4, 5 | the Oct 28 fine-tune record's share link on the deployed app (A) | none: this is the core |
| 6 to 10 | the Oct 31 agent record on the deployed app with a real share link. Its share token in the repo is now a placeholder (A, `e27d2fc`), so A has to publish it on the deploy | Drop shots 6 to 11 and use `storyboard.md` for plan A's fine-tune-only shots: the cut is recorded on the deployed app or not at all |
| 7 | the caching finding's text as the record holds it (30 tasks) | none: the 130-task caption comes from A's status file, not the screen |
| 9 | the retry-loop finding on screen, with its code line | none: it is a static finding in the record |
| 10 | the agent cockpit's tank and cost readouts | If the engine abstains, record the hatched band as it is and keep the line, since it promises no number |
| 11b | A's production record on the deployed app, from its share link, with `harness.spark_open_triage` on in production; a probe that didn't finish, whose triage finding quotes a log line (not "(no log line quoted)"); the triage call in its provenance; and a screen that shows the finding. On `main` none does: the detail sheet lists only findings with a node (`DetailSheet.tsx`), the map skips info findings and those with no node (`Orbit.tsx`), the guide's tour only those with a fix (`ruleTour.ts`), and triage's finding is info with no node (`harness/triage.py`). C and A are asked (`from-d.md`) | No link by Tue Nov 3, noon ET, or no screen that shows the finding by then: drop the shot, not faked (no sample, fixture, staged or edited record, and no fixed tag standing in for it), and put shot 10 back |
| 12 | the checks page on the deploy, unedited, and B's scorer run on Mon Nov 2 for the README's table | Show the page as it is. If its checks still say "not measured", change the line to "Every prediction gets checked against real runs, misses included." |

## Before recording

- [ ] Every number spoken or captioned is in the table above, with its source.
- [ ] Both records open from their share links on the deployed app.
- [ ] Muse Spark is named only where a record shows it did that work: the agent's own calls
  (shots 6 to 11), and the script map only if the fine-tune record shows the call.
- [ ] Muse Spark Open is named only in shot 11b, on A's production record that shows the triage
  finding (with a quoted log line) and call; no link or no screen for it by Tue Nov 3, noon ET,
  and the shot is dropped, not faked.
- [ ] No sample screens ("Illustrative").
- [ ] The retry loop is called "read from the code", never "caught running", unless a re-sample
  shows the stall and its numbers are added above.
- [ ] Captions on for every narration line.
