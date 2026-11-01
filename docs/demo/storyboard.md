# Demo video storyboard

About 3 minutes, plus a 45-second cut for X. Built on one measured run: the planted-bug LoRA
fine-tune in the zoo (`zoo/finetune_lora_hf/train.py`: Qwen2.5-1.5B on Alpaca, batch 4 with 4-step
accumulation, one L4), checked in on Oct 28.

If plan A's GRPO, Pareto, race or campaign beats aren't ready by Tuesday, record the fallback
cut in `fallback-cut.md` instead (the fine-tune and the agent run, measured numbers only).

## Rules for this video

- **Only measured numbers, and measured time or tokens priced at list rates (below), are spoken
  or captioned.** Each comes from A's status file
  (`docs/status/a.md`, "First public number (one script)") or the script itself. Predicted
  numbers may appear on screen, where the app labels them "Predicted", but the narration never
  quotes them.
- **Dollar figures are spoken and captioned "at list price"** (Chiya, Nov 1): measured time or
  tokens priced at list rates, e.g. "$0.042 of GPU at list price". The README keeps tagging them
  derived; in the video the caption's "at list price" says the same thing.
- **"One script" stays on screen whenever a number is.** One run is an example, not a rate.
- **Only our work, built Oct 26 to Nov 4,** and the video says so. Every Meta model is named on
  screen at the moment it works.
- **Nothing from Chiya's ongoing research.**
- **Record on the deployed app**, with the real record of the Oct 28 run (record
  `ICDYrfHQ2nBNbv3mlRHHMg`), never a sample: samples are labelled "Illustrative".
- **The Spark Open triage shot (12b, and E in the 45-second cut) is a real production record or
  nothing** (Chiya, Nov 1). A posts its share link on Sun Nov 1. If the link isn't there by
  Tue Nov 3 at noon ET, the shot is dropped, not faked: no sample, fixture, staged or edited
  record, and no fixed tag standing in for it.

## Numbers this video may use

| Number | What it is | Source |
| --- | --- | --- |
| 189 s | probe time until the eval out-of-memory was caught (probe `p1-eval`) | measured, A's status, record `ICDYrfHQ2nBNbv3mlRHHMg` |
| 20.4 GiB of a 22.06 GiB card | memory in use at the peak when the eval ran out, on the L4 | measured: 21,902,642,688 bytes at the peak (A, Oct 31) and 23,689,363,456 bytes of L4 (the record's env), shown in GiB as the app shows memory |
| $0.042 | GPU cost of that probe, captioned "$0.042 of GPU at list price" | measured probe time x the $0.80/h L4 list price, A's status (the README tags this kind of figure derived) |
| step 200 | where the real run crashes: its first evaluation | the script (`EVAL_EVERY = 200`), and measured: A's full run of Oct 31 (`a-gt-lora-l4-5`) ran out of memory at step 200, at 22.13 GB (20.61 GiB) allocated |
| 1.68 s | measured step time with eval off (probe `p1`, p50) | measured, A's status |

Not used: the derived "about 6 minutes and $0.09" for the crash it saved (derived, not measured),
any hours or monthly savings, and any accuracy rate (one script gives no rate).

## Shots

| # | Time | Screen | What moves | Narration (one line) | On-screen label |
| --- | --- | --- | --- | --- | --- |
| 1 | 0:00 | Terminal or IDE with `train.py` open, cursor on `EVAL_EVERY = 200` | Slow zoom to the eval batch line | "This fine-tune looks fine. At step 200, its first evaluation, it runs out of memory." | "A real training script, with a planted bug" |
| 2 | 0:10 | Preflight home | Hero title fades in; cursor clicks "Start a preflight" | "Preflight flies the run before you pay for it." | "Built Oct 26 to Nov 4" |
| 3 | 0:18 | Check-in, Repo link tab | Paste the zoo repo link; choose "Finish soonest" and L4; tick "Got it" | "Point it at the code and say what matters." | |
| 4 | 0:30 | Check-in, data-use notice | Hold on the notice for a beat | "It runs a few minutes sandboxed on a rented GPU, with none of our keys." | |
| 5 | 0:38 | Boarding | Stages light in turn; "Reading your script" with the Muse Spark tag | "Muse Spark reads the code and finds the training loop." | "Muse Spark (Meta)" at the moment the stage lights |
| 6 | 0:50 | Boarding | "Running probe steps" lights; route fills | "Then it runs a few real steps on a real GPU." | "L4, measured" |
| ~~7~~ | | Boarding, "Checking the logs" | Cut: shot 12b shows Spark Open at work in a record, and takes these 10 s. Back only if 12b is dropped, under its own rule below | | |
| 8 | 1:00 | Cockpit of the Oct 28 record | Map draws; the memory ring fills past the line; red finding "runs out of memory" | "In 189 seconds, for about four cents of GPU at list price, it caught the crash: the evaluation pass reaches 20.4 GiB of a 22.06 GiB card and runs out." | "189 s of probing · $0.042 of GPU at list price · 20.4 of 22.06 GiB · one script" |
| 9 | 1:15 | Cockpit, detail sheet on the eval node | Open the finding; press "Adjust eval batch" | "It says why, and which setting fixes it." | |
| 10 | 1:25 | Cockpit, knob dock | Drag eval batch size down; dials and ring move together | "Move a setting and the whole run answers at once." | "Predicted" (the app's own label on the dials) |
| 11 | 1:40 | Cockpit, live guide | The guide proposes one knob move; Accept | "Ask the guide, and Muse Spark moves the knobs that matter, one at a time." | "Muse Spark (Meta)" |
| 12 | 1:55 | Fix loop | Two or three candidate fixes run side by side; the measured winner is kept | "Muse Code tries fixes in parallel, and Preflight's own probe picks the winner by measurement." | "Muse Code (Meta)" |
| 12b | 2:10 | A's production triage run of the LoRA zoo script, the same script as the fine-tune story, from its share link (posted by Tue Nov 3, noon ET): the cockpit's Findings band (C, on `main` since round five), reached from the "1 finding off the map" chip, on the triage finding for the eval probe's out-of-memory | Click the "1 finding off the map" chip; the page moves to the Findings band. Hold on the finding: its title, its one sentence, the quoted log line in its own box, and the app's own "Found by Muse Spark Open triage" | "When a probe fails on the live site, Muse Spark Open reads the end of its log and quotes the line." | "Muse Spark Open (Meta, open weights, on Baseten) · triage of a failed probe's log · a real job on the live site" |
| 13 | 2:20 | Checks page | Scroll the "Each check" list: the caught OOM, and the step-time misses | "Every prediction is checked against real runs, misses included." | "Held-out checks on real runs" |
| 14 | 2:33 | Cockpit, then a new tab | Copy the address from the address bar, paste it in a new tab; the same flight opens | "Every flight is a link anyone can open." | |
| 15 | 2:43 | End card (about 10 s, ends near 2:55) | Public link and repo | "Preflight. Fly the run before you pay for it." | link · repo · "Built Oct 26 to Nov 4" |

## Each shot's dependency, and its fallback

| Shot | Needs | Fallback if it isn't ready when we record |
| --- | --- | --- |
| 3, 4, 6 | check-in open on the deploy (`runner.queue`) and the runner up, for a live check-in we record as it happens | Boarding can't be replayed (a finished job's events arrive at once, and the stream needs the runner), so cut shots 3 to 7 and open on the Oct 28 record's cockpit with the caption "checked in on Oct 28" |
| 5 | Muse Spark reading the script (`harness.spark_script_map`), and a recorded run where it did (the record's model calls show it) | Boarding's model tags are fixed labels that show whether or not the model ran: crop or blur the tag and drop the narration line, or cut the shot. C is asked to show tags only for models that ran (from-d.md, finding 12) |
| 7 | Muse Spark Open doing triage in the recorded run, shown in the record | Cut (its 10 s go to 12b). If 12b is dropped, shot 7 comes back only if the recorded check-in's own record shows the triage call; otherwise give the 10 s to shots 8 to 10, and crop the fixed tag |
| 8 to 10 | the Oct 28 record's share link (A) on the deployed app | none: this is the core of the video |
| 11 | the live guide (`c.guide.live`) working on the deploy | Show the rule-based tour, labelled as such, and name no model in that shot |
| 12 | the fix loop (`c.fix_loop`) with a recorded race and a measured winner | Cut the shot and give its 15 seconds to shots 8 to 10 |
| 12b | A's production record on the deployed app, from its share link, with `harness.spark_open_triage` on in production; a probe that didn't finish, whose triage finding quotes a log line (not "(no log line quoted)"); the triage call in its provenance; and C's Findings band, which shows it (on `main` `85fbf9a`; D checked it in Chromium, desk and phone, with a triage finding added to a copy of A's Oct 28 record: the title, the sentence, the quoted log line and "Found by Muse Spark Open triage" all show once the "1 finding off the map" chip is clicked). A's run is the LoRA zoo script, so the failed probe is the eval probe's out-of-memory | No link by Tue Nov 3, noon ET, or no screen that shows the finding by then: drop the shot, not faked (no sample, fixture, staged or edited record, and no fixed tag standing in for it), and give its 10 s back to shot 7's rule. Name Muse Spark Open nowhere else in the video unless a recorded record shows its call |
| 13 | the checks page as on the deploy, unedited | none: show it as it is |
| 14 | the record's share link (A) | none needed: the link is the page's address; copy it from the address bar |

## The 45-second cut (for X), ready to record

Five shots from the main storyboard (1, 8, 10, 12b, 15), recorded once the site is public. It has to
work with the sound off, so **every narration line is also a burned-in caption**, and the first
frame has to stand on its own.

**Format:** 1920 x 1080, 30 fps, H.264 MP4, under 45 s. Browser window at
1440 x 900 inside the frame, dusk theme, browser zoom 110%, no bookmarks bar or other tabs
showing, cursor highlight on. Record shots B and C on the deployed app with the Oct 28
record's share link, and shot E with A's production record's, never a sample.

| # | Time | Screen | Action | Narration (about 2.5 words a second) | Caption, burned in |
| --- | --- | --- | --- | --- | --- |
| A (shot 1) | 0:00 to 0:08 | `zoo/finetune_lora_hf/train.py` in an editor, dark theme, 20 pt | Hold 1 s on the file, then a slow zoom to `EVAL_EVERY = 200` and the eval batch line | "This fine-tune looks fine. At step 200, its first evaluation, it runs out of memory." (16 words) | 0:00 to 0:03 "Your GPU job, flown before you run it" · 0:03 to 0:08 "At step 200, its first evaluation, it runs out of memory" |
| B (shot 8) | 0:08 to 0:24 | Cockpit of the Oct 28 record | The map draws, the memory ring fills past the line, the red finding "runs out of memory" appears; hold 2 s on the finding | "Preflight ran a few real steps on one L4. In 189 seconds of probing, for about four cents of GPU at list price, it caught the crash before the real run." (31 words written, about 35 spoken) | "Caught in 189 s of probing, before the run · $0.042 of GPU at list price · eval peaks at 20.4 of 22.06 GiB · one script" · from 0:08 to 0:12 also "Script read by Muse Spark (Meta)" |
| C (shot 10) | 0:24 to 0:32 | Same cockpit, knob dock | Drag eval batch size from 32 down; the ring drops under the line and the dials move together | "Move a setting, and the whole run answers at once." (10 words) | "Predicted" stays as the app labels the dials; caption "Every reading carries its range" |
| E (shot 12b) | 0:32 to 0:39 | As shot 12b: A's production triage run of the LoRA script, the cockpit's Findings band, on the eval probe's triage finding | Cut in on the Findings band (the chip already clicked); hold on the finding, its quoted log line and the app's "Found by Muse Spark Open triage" | "When a probe fails, Muse Spark Open reads its log and quotes the line." (14 words) | "Muse Spark Open (Meta, open weights, on Baseten) · triage of a failed probe's log · a real job on the live site" |
| D (shot 15) | 0:39 to 0:45 | End card | Public link and repo URL fade in | "Preflight. Fly the run before you pay for it." (9 words) | link · repo · "Built Oct 26 to Nov 4 for the Meta hackathon" |

Narration is 80 written words; spoken, with "200" and "189" said in full, about 86, so about 34 s
of speech in 45 s, and shot B's 35 spoken words take its 16 s at about 2.2 a second (trim
"Preflight ran a few real steps on one L4" to its caption if the take runs long). The only spoken numbers are in
the table at the top: step 200, 189 s and $0.042 said as "about four cents of GPU at list price";
20.4 of 22.06 GiB is captioned, not spoken. Nothing is predicted aloud; the dials in shot C keep
the app's own "Predicted" label. Shot E speaks no number. If shot E is dropped (no production link
by Tue Nov 3, noon ET), C runs 0:24 to 0:36 and D 0:36 to 0:45, as before.

**Before recording this cut** (on top of the list below):

- [ ] The Oct 28 record's share link opens on the deployed app (A's open item in `from-d.md`).
- [ ] The record's provenance shows the Muse Spark script-map call (A measured its tokens:
  $0.017833, run `a-smoke-l4-main-4`); if it doesn't, drop the "Script read by Muse Spark"
  caption.
- [ ] Shot C: dragging eval batch size moves the ring under the line on that record. If the
  engine abstains there, record it as it shows (the hatched band is honest) and change the
  caption to "Outside what was measured: Preflight says so".
- [ ] Shot E: A's production record opens from its share link by Tue Nov 3, noon ET, and the app
  shows its triage finding with a quoted log line; otherwise drop shot E, not faked (never a
  sample or a staged record).
- [ ] Captions are burned in, white on a dark band, at least 36 px tall at 1080p, and none is on
  screen for under 2 s.
- [ ] The repo is public and the link on the end card opens.

## Before recording

- [ ] Every number spoken or captioned is in the table above, with its source.
- [ ] The Oct 28 record opens from its share link on the deployed app.
- [ ] Each Meta model named in the video did that work in the record on screen (that record shows it).
- [ ] Shot 12b (and E): A's production record, from its share link, shows a finding marked
  "Found by Muse Spark Open triage" that quotes a log line; posted by Tue Nov 3, noon ET, or the
  shot is dropped, not faked. Spark Open's work is shown in that record, not the Oct 28 one.
- [ ] No sample screens (they say "Illustrative").
- [ ] Each shot's flag is on and checked on the deploy, or the fallback is used.
- [ ] Captions on for every narration line.
