# Demo video storyboard

About 3 minutes, plus a 45-second cut for X. Built on one measured run: the planted-bug LoRA
fine-tune in the zoo (`zoo/finetune_lora_hf/train.py`: Qwen2.5-1.5B on Alpaca, batch 4 with 4-step
accumulation, one L4), checked in on Oct 28.

## Rules for this video

- **Only measured numbers are spoken or captioned.** Each comes from A's status file
  (`docs/status/a.md`, "First public number (one script)") or the script itself. Predicted
  numbers may appear on screen, where the app labels them "Predicted", but the narration never
  quotes them.
- **"One script" stays on screen whenever a number is.** One run is an example, not a rate.
- **Only our work, built Oct 26 to Nov 4,** and the video says so. Every Meta model is named on
  screen at the moment it works.
- **Nothing from Chiya's ongoing research.**
- **Record on the deployed app**, with the real record of the Oct 28 run (record
  `ICDYrfHQ2nBNbv3mlRHHMg`), never a sample: samples are labelled "Illustrative".

## Numbers this video may use

| Number | What it is | Source |
| --- | --- | --- |
| 189 s | probe time until the eval out-of-memory was caught (probe `p1-eval`) | measured, A's status, record `ICDYrfHQ2nBNbv3mlRHHMg` |
| 20.4 GiB of a 22.06 GiB card | memory in use at the peak when the eval ran out, on the L4 | measured: 21,902,642,688 bytes at the peak (A, Oct 31) and 23,689,363,456 bytes of L4 (the record's env), shown in GiB as the app shows memory |
| $0.042 | GPU cost of that probe at L4 list price | measured (probe time x $0.80/h list), A's status |
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
| 7 | 1:00 | Boarding | "Checking the logs" with the Muse Spark Open tag | "Muse Spark Open, open weights on Baseten, reads the logs." | "Muse Spark Open (Meta, open weights)" |
| 8 | 1:10 | Cockpit of the Oct 28 record | Map draws; the memory ring fills past the line; red finding "runs out of memory" | "In 189 seconds it caught the crash: the evaluation pass reaches 20.4 GiB of a 22.06 GiB card and runs out." | "189 s · $0.042 · 20.4 of 22.06 GiB · one script" |
| 9 | 1:25 | Cockpit, detail sheet on the eval node | Open the finding; press "Adjust eval batch" | "It says why, and which setting fixes it." | |
| 10 | 1:35 | Cockpit, knob dock | Drag eval batch size down; dials and ring move together | "Move a setting and the whole run answers at once." | "Predicted" (the app's own label on the dials) |
| 11 | 1:50 | Cockpit, live guide | The guide proposes one knob move; Accept | "Ask the guide, and Muse Spark moves the knobs that matter, one at a time." | "Muse Spark (Meta)" |
| 12 | 2:05 | Fix loop | Two or three candidate fixes run side by side; the measured winner is kept | "Muse Code tries fixes in parallel, and Preflight's own probe picks the winner by measurement." | "Muse Code (Meta)" |
| 13 | 2:20 | Checks page | Scroll the "Each check" list: the caught OOM, and the step-time misses | "Every prediction is checked against real runs, misses included." | "Held-out checks on real runs" |
| 14 | 2:33 | Cockpit, then a new tab | Copy the address from the address bar, paste it in a new tab; the same flight opens | "Every flight is a link anyone can open." | |
| 15 | 2:43 | End card (about 10 s, ends near 2:55) | Public link and repo | "Preflight. Fly the run before you pay for it." | link · repo · "Built Oct 26 to Nov 4" |

## Each shot's dependency, and its fallback

| Shot | Needs | Fallback if it isn't ready when we record |
| --- | --- | --- |
| 3, 4, 6 | check-in open on the deploy (`runner.queue`) and the runner up, for a live check-in we record as it happens | Boarding can't be replayed (a finished job's events arrive at once, and the stream needs the runner), so cut shots 3 to 7 and open on the Oct 28 record's cockpit with the caption "checked in on Oct 28" |
| 5 | Muse Spark reading the script (`harness.spark_script_map`), and a recorded run where it did (the record's model calls show it) | Boarding's model tags are fixed labels that show whether or not the model ran: crop or blur the tag and drop the narration line, or cut the shot. C is asked to show tags only for models that ran (from-d.md, finding 12) |
| 7 | Muse Spark Open doing triage in the recorded run, shown in the record | Cut the shot; name Muse Spark Open only where a record shows it, and crop the fixed tag |
| 8 to 10 | the Oct 28 record's share link (A) on the deployed app | none: this is the core of the video |
| 11 | the live guide (`c.guide.live`) working on the deploy | Show the rule-based tour, labelled as such, and name no model in that shot |
| 12 | the fix loop (`c.fix_loop`) with a recorded race and a measured winner | Cut the shot and give its 20 seconds to shots 8 to 10 |
| 13 | the checks page as on the deploy, unedited | none: show it as it is |
| 14 | the record's share link (A) | none needed: the link is the page's address; copy it from the address bar |

## The 45-second cut (for X), ready to record

Four shots from the main storyboard (1, 8, 10, 15), recorded once the site is public. It has to
work with the sound off, so **every narration line is also a burned-in caption**, and the first
frame has to stand on its own.

**Format:** 1920 x 1080, 30 fps, H.264 MP4, under 45 s. Browser window at
1440 x 900 inside the frame, dusk theme, browser zoom 110%, no bookmarks bar or other tabs
showing, cursor highlight on. Record the cockpit shots on the deployed app with the Oct 28
record's share link, never a sample.

| # | Time | Screen | Action | Narration (about 2.5 words a second) | Caption, burned in |
| --- | --- | --- | --- | --- | --- |
| A (shot 1) | 0:00 to 0:08 | `zoo/finetune_lora_hf/train.py` in an editor, dark theme, 20 pt | Hold 1 s on the file, then a slow zoom to `EVAL_EVERY = 200` and the eval batch line | "This fine-tune looks fine. At step 200, its first evaluation, it runs out of memory." (16 words) | 0:00 to 0:03 "Your GPU job, flown before you run it" · 0:03 to 0:08 "At step 200, its first evaluation, it runs out of memory" |
| B (shot 8) | 0:08 to 0:24 | Cockpit of the Oct 28 record | The map draws, the memory ring fills past the line, the red finding "runs out of memory" appears; hold 2 s on the finding | "Preflight ran a few real steps on one L4. In 189 seconds, for about four cents of GPU, it caught the crash before the real run." (25 words written, about 30 spoken) | "Caught in 189 s, before the run · $0.042 · eval peaks at 20.4 of 22.06 GiB · one script" · from 0:08 to 0:12 also "Script read by Muse Spark (Meta)" |
| C (shot 10) | 0:24 to 0:36 | Same cockpit, knob dock | Drag eval batch size from 32 down; the ring drops under the line and the dials move together | "Move a setting, and the whole run answers at once." (10 words) | "Predicted" stays as the app labels the dials; caption "Every reading carries its range" |
| D (shot 15) | 0:36 to 0:45 | End card | Public link and repo URL fade in | "Preflight. Fly the run before you pay for it." (8 words) | link · repo · "Built Oct 26 to Nov 4 for the Meta hackathon" |

Narration is 59 written words; spoken, with "200" and "189" said in full, about 66, so about 27 s
of speech in 45 s, and shot B's 30 spoken words fit its 16 s. The only spoken numbers are in the
table at the top: step 200, 189 s and $0.042 said as "about four cents"; 20.4 of 22.06 GiB is
captioned, not spoken. Nothing is predicted aloud; the dials in shot C keep the app's own
"Predicted" label.

**Before recording this cut** (on top of the list below):

- [ ] The Oct 28 record's share link opens on the deployed app (A's open item in `from-d.md`).
- [ ] The record's provenance shows the Muse Spark script-map call (A measured its tokens:
  $0.017833, run `a-smoke-l4-main-4`); if it doesn't, drop the "Script read by Muse Spark"
  caption.
- [ ] Shot C: dragging eval batch size moves the ring under the line on that record. If the
  engine abstains there, record it as it shows (the hatched band is honest) and change the
  caption to "Outside what was measured: Preflight says so".
- [ ] Captions are burned in, white on a dark band, at least 36 px tall at 1080p, and none is on
  screen for under 2 s.
- [ ] The repo is public and the link on the end card opens.

## Before recording

- [ ] Every number spoken or captioned is in the table above, with its source.
- [ ] The Oct 28 record opens from its share link on the deployed app.
- [ ] Each Meta model named in the video did that work in the recorded run (its record shows it).
- [ ] No sample screens (they say "Illustrative").
- [ ] Each shot's flag is on and checked on the deploy, or the fallback is used.
- [ ] Captions on for every narration line.
