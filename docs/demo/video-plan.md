# The 3-minute video: shot list and voice-over (recorded Wed Nov 4)

By D, Mon Nov 2, overnight (Chiya's list, item 3); statuses updated Mon Nov 2, 11:10 ET; shot 9b added 12:05 ET, its stopgap step replaced 16:30 ET; shot 7b added 17:45 ET; shot 6b added 20:15 ET. The plan of record is now the section "At the freeze" below. This plan
replaces the plan A and plan B shot lists in `storyboard.md` and `fallback-cut.md`, which stay as
the earlier drafts. The 45-second X clip has its own section below (new concept, Nov 3).

## The edit as built (Tue Nov 3, evening)

By D, Nov 3, 21:40 ET. Chiya's calls (19:54 to 21:21): **no voice-over**; captions as "bold words
and expressive text", not a box behind a subtitle (Chiya, 21:21). D's design for that: a short sans
lead line over a large serif punch line, key words in the accent colour, set into the frame's empty
space, at most 9 words a line. It carries the film, over a music bed (lofi-tech or nastelbom, both tried), with the Antonov take-off
sound on Check in. Three cuts of one picture, built in HyperFrames from D's takes of a local
production build of `main` `f52f3e44` and Chiya's live takes C1 to C4 (until they arrive, labelled
placeholder cards): lofi-tech 2:48.5, nastelbom 2:48.5, and a shorter nastelbom cut, 2:28, that
ends with the track (no loop). The frame plan, its critique and the replan are in D's video project
(PLAN.md); the shot list below is the earlier plan of record.

The PM's rules review (Nov 3, 21:08), as the edit meets it:

1. **The opening line.** The film opens on a card: "Nothing of this existed on Oct 26." /
   "Everything you'll see was built during the hackathon." (4.5 s; 4 s in the short cut).
2. **Live Meta model calls, labelled, each shown whole (planned; the takes are Chiya's C1 to C3, not
   in yet).** Boarding (shot 4: Muse Spark reads the script, Muse Spark Open reads every log), the
   guide (shot 8: Muse Spark) and the Muse Code lanes (shot 8b) come from Chiya's live takes. No
   cut inside a model's call: each stage plays from its start to its result, labelled by its
   caption. A long call plays faster, with the speed on screen, rather than being cut; only the
   GPU wait between Boarding's stages is shortened. PM: say if "uncut" means real time too.
3. **What's simulated (shot 13).** Every sample is a measured record now (`numbers: "measured"` for all eight
   in `web/src/workspace/record/load.ts`), so no "Example numbers" banner appears in the film. The
   checks shot's caption reads "Misses included · cross-GPU figure simulated on held-out corpus".
4. **"(planted)" (shots 10 and 11)** ends the captions of the agent sample ("an uncapped retry loop (planted)") and
   the GRPO sample ("15 of 20 prompts teach nothing (planted)").
5. **Muse Code (shot 8b).** The caption reads "Preflight re-ran each on an A10 on Nov 2": each
   lane's earlier patch, as the lane's own line says ("An earlier patch for this fix, checked on 2
   Nov on an A10"), beside its "Not verified" for the patch written on film.
6. **Search grounding: in.** `harness.spark_search` is on in production, and the refilled
   LoRA record shows "No known issue found for probe p1-eval's error ... Muse Spark searched the
   web"; it is on screen in the replay shot, in the app's own words. No separate shot. (This
   replaces "cut at the freeze" below.)
7. **Length.** 2:48.5 at most, the campaign shot included; the short cut 2:28 [measured: the built
   timelines] (the scene lengths in D's
   `gen.py`).
8. **The X post.** "Meta hackathon" is in the first post (`docs/submission/x-post.md`).

## Real times, for anyone re-running the video's flows

(Chiya, Nov 4, 6:37 PM: judges run the flows as the video shows them, and the video cuts and speeds
them up to fit 3 minutes.) Every shot that is cut or sped up says so in a small badge on screen
("Cut", "Sped up", "3.3×"). The real times:

| Flow in the video | Real time | Source |
| --- | --- | --- |
| Check in a repo link, to the cockpit (Boarding: the GPU starts, the probes run, the logs are read) | about 5 to 11 minutes | measured: production's two LoRA check-ins |
| The live guide (Muse Spark), one move | about 5 to 15 s a turn, three turns a tour; the panel says "Thinking" meanwhile; after 25 s a turn the rule-based tour takes over (`TIMEOUT_S = 25.0`) | measured: C's D86, nine live turns, Nov 4 |
| Muse Code writes the fixes | 30 s to a minute: 28 s and 34 s for the two lanes on Nov 3; about 50 s from "Write the fixes" to both lanes written in the video's take (Nov 4) | measured: C's live check, Nov 3, 7:53 PM (from-c.md); Chiya's take C3 |
| Moving a knob | at once (139 ms p95 on the live site) | measured: C's stranger test, Nov 3, 7:50 PM (from-c.md, "Live with the LoRA refill") |
| `preflight run` on a laptop GPU, then `preflight watch` | the probes take a minute or two [guess: Windows loads CUDA slowly the first time]; the watched run goes on until it finishes or you press Ctrl+C | Chiya's RTX 4070 takes, Nov 3 |

## At the freeze (Tue Nov 3): the plan of record for Wednesday's recording

By D, Tue Nov 3, 08:25 ET (the PM's 08:07 item 6, plan only). This section supersedes the shot
table under "Shots" below, which stays as Monday's version with each shot's history. The rules
below still hold. One change since Monday: **the fix loop is on in production**
(`c.fix_loop`, Nov 2), so Muse Code gets a shot (8b), the agentic beat the plan was missing.
Two shots stay open until noon today, as the rules say: 9b (Chiya's device) and 6b (C's record).

### Shot list against the freeze

What each shot needs on the production flags at the freeze (`flags.production.json`, main
`555b5af`). "Works" means its flow is on in production and was seen working (the source is in
Monday's table below); "Chiya's live test" means D can't reach the live site to see it.

| # | Time | Shot | At the freeze | For Wednesday |
| --- | --- | --- | --- | --- |
| 1 | 0:00 | The LoRA script in an editor, zoom to `EVAL_EVERY = 200` | Works (no app) | |
| 2 | 0:08 | Live site home, "Start a preflight" | Works | |
| 3 | 0:14 | New flight: repo link, goal, GPU, the data-use panel, "Got it", "Check in" | Works (D's walk, Nov 2: "Got it" sits below the samples list at 1366 px; scroll to it on camera) | Film a fresh check-in |
| 4 | 0:26 | Boarding: "Reading your script" (Muse Spark), "Checking the logs" (Spark Open) | Works | Cut the GPU wait |
| 5 | 0:38 | The Oct 28 record's cockpit: the ring past the line, "runs out of memory" | The share route works on a production build under `wrangler pages dev` (D, Nov 3, with a stand-in record); this record not yet opened in a browser | Open `/s/ICDYrfHQ2nBNbv3mlRHHMg` in Chiya's live test |
| 6 | 0:52 | The triage finding with its quoted log line | Works on A's committed copy (D, Nov 2) | Open `/s/Q3s5qyHKTNf-fCng4dwn6g` live |
| 6b | 1:01 | A finding "Found by Muse Spark diagnosis, checked by Muse Spark Open" | Live since C's 8:07 PM deploy; C's live check (D39) found "Plots and numbers agree" on the einops resend's record | C sends that record's link by noon, or cut (its 8 s go to shot 8b) |
| 7 | 1:09 | Drag "Eval batch" down; the ring drops, the dials move | Works | |
| 8 | 1:17 | The guide: one knob move, Apply | Works (`c.guide.live`); if the guide's daily cap is hit, the rule-based tour shows instead: drop the Muse Spark label and line | |
| 8b | 1:28 | **New:** the eval finding's "Fix it with Muse Code": candidates race; verify rejects the fix that still runs out of memory and keeps the stacked fix at eval batch 8 | On in production (Nov 2); C's race on the LoRA sample: the single fix rejected, the stacked fix passed, eval peak 10.4 GiB, loss within 0.11%, US$0.39 [measured] | Film one race on the LoRA sample (PF102, `lora-qwen`) on an A10, as C's measured race was (`api/preflight_api/fix/races/nov2_stack.json`), so the caption's 10.4 GiB is that race's; about US$0.39 [derived from C's race]; cut the wait. On another record, caption only what the filmed race shows |
| 9 | 1:44 | The exports: batch file, command, W&B helper | Works (`c.exports`) | |
| 9b | 1:51 | `preflight watch` beside the cockpit; an alert in both | Works on a GPU (D's A10 check, Nov 2) | Chiya's call by noon: her RTX 4070, else a Colab T4, else cut (its 10 s go to shots 10 and 11, 5 s each) |
| 10 | 2:01 | PF331, the agent sample: the caching finding | Works (sample says Measured) | |
| 11 | 2:13 | PF510, the GRPO sample: the RL band | Works (sample says Measured) | |
| 12 | 2:22 | New flight with "Private triage"; a private record's "Found by Muse Glimmer" | Works (C's record 01M6EG0284PWZDRXGGQVVZYVND) | Ask C for its link |
| 13 | 2:33 | Checks page, "Each check" | **Keep** (Nov 3, 18:15): B's three strings are on `main` (`checks.title` "Checks against full runs", `checks.col.result` "Scored", `checks.method`), and the page reads B's final table | Film the page as it is; the voice-over and caption below match its words |
| 14 | 2:42 | The share link opened in a new tab | Works: use the `/s/` link, now live, rather than the address bar | |
| 15 | 2:50 | End card | Works | Repository public first |

Cut at the freeze: **7b** (the range test: `probes.lr_range` is off). **Search grounding** was cut
here; it is in as of Nov 3, 21:20 (see "The edit as built", item 6).

### Voice-over with times

About 2.5 words a second; each line is also a burned-in caption (at least 2 s, white on a dark
band). Spoken numbers come from the submission's numbers table, each said as the table says.
Total 2:58.

| Time | Shot | Voice-over | Words | Caption beside it (on-screen label) |
| --- | --- | --- | --- | --- |
| 0:00 to 0:08 | 1 | "This fine-tune looks fine. At step 200, its first evaluation, it runs out of memory." | 15 | "A real training script, with a planted bug" |
| 0:08 to 0:14 | 2 | "Preflight flies your training run before you pay for it." | 10 | "Built Oct 26 to Nov 4" |
| 0:14 to 0:26 | 3 | "Point it at your code and say what matters. It tells you exactly what it sends." | 16 | |
| 0:26 to 0:38 | 4 | "Muse Spark reads the script. A few real steps run on a real GPU, and Muse Spark Open reads every probe's log." | 22 | "Muse Spark (Meta)", then "Muse Spark Open (Meta's open model, on Baseten)" |
| 0:38 to 0:52 | 5 | "In 189 seconds of probing, for about four cents of GPU at list price, it caught the crash before the real run." | 22 | "189 s of probing · $0.042 of GPU at list price · 20.4 of 22.06 GiB · one script" |
| 0:52 to 1:01 | 6 | "Muse Spark Open says what went wrong and quotes the line it read; code checks the quote is real." | 19 | "Muse Spark Open (Meta's open model, on Baseten)" |
| 1:01 to 1:09 | 6b | "Muse Spark reads the run's plots against the raw numbers, and every number it cites is checked." | 17 | "Muse Spark (Meta) · diagnosis, every number checked" |
| 1:09 to 1:17 | 7 | "Move a setting and the whole run answers at once, with its range." | 13 | the app's "Predicted" |
| 1:17 to 1:28 | 8 | "Ask the guide, and Muse Spark moves the knobs that matter for your goal, one at a time. You decide each move." | 22 | "Muse Spark (Meta)" |
| 1:28 to 1:44 | 8b | "Or let Muse Code write the fix. Preflight runs each candidate for real: the one that still runs out of memory is rejected, the one that fits is kept." | 29 | "Muse Code (Meta) · verified by Preflight's own re-probe · eval peak 10.4 GiB · one script" |
| 1:44 to 1:51 | 9 | "Take the plan with you: a batch file, the command, and your settings." | 13 | |
| 1:51 to 2:01 | 9b | "Run it for real and Preflight watches. When the run leaves its predicted band, it says so." | 17 | the app's "Predicted" band |
| 2:01 to 2:13 | 10 | "Not just training. A coding agent on Muse Spark ran 130 tasks through Preflight, and not one input token came from the cache." | 23 | "Agent eval on Muse Spark (Meta) · measured · 0 of 905,739 input tokens from cache, 130 tasks, two runs" |
| 2:13 to 2:22 | 11 | "In this GRPO run, 15 of 20 prompts had no reward spread at all, so they teach nothing." | 18 | "GRPO on an L4 · measured · 15 of 20 prompts with no reward spread" |
| 2:22 to 2:33 | 12 | "For code that mustn't leave, private mode triages on Muse Glimmer, Meta's open weights, on our own GPU." | 18 | "Muse Glimmer (Meta's open weights, on Preflight's own GPU)" |
| 2:33 to 2:42 | 13 | "Every prediction is checked against a real script's full run, misses included." | 12 | the page's own title, "Checks against full runs" |
| 2:42 to 2:50 | 14 | "Every flight is a link anyone can open." | 8 | |
| 2:50 to 2:58 | 15 | "Preflight. Built October 26 to November 4. Fly the run before you pay for it." | 15 | link · repository · "Built Oct 26 to Nov 4" |

If 6b is cut, shots 7 and 8 move up 8 s (1:01 to 1:09, 1:09 to 1:20) and 8b runs 1:20 to 1:44 with the same line, said more slowly. If 9b is cut, shot 10
runs 1:51 to 2:08 and shot 11 2:08 to 2:22, with the same lines.

### The 45-second X clip: "The cheapest place for your training run to crash"

By D, Tue Nov 3, 17:00 ET (Chiya's pick, the PM's 16:08). It replaces the earlier 45-second cut
(its rows A to F); the 3-minute plan above keeps its shots. Plan only; filmed Wed Nov 4.

**The voice.** Hook: "The cheapest place for your training run to crash." Opener: "Your GPU is a
terrible place to discover a typo." Promise: "Model the hours from minutes." Sign-off: "Crash it
here, not on your GPU bill." Dry, short sentences; no exclamation marks.

**Format.** 1920 x 1080, 30 fps, H.264; a 1080 x 1350 crop for the X feed if Chiya wants one.
Captions burned in on a dark band under the frame (white, at least 2 s each), so it works with the
sound off. About 2.5 words a second.

**Rules for this cut** (on top of the Rules below):

- **Two kinds of picture, never mixed.** Seconds 0 to 8 and the end card are motion graphics: title
  cards that are clearly not the app. Seconds 8 to 39 are screen recordings of the live app,
  untouched: no mock-ups, no stamps, badges or arrows added on top. A zoom or a crop is fine; a
  word the app doesn't show is not.
- **Every number on screen or in the voice-over is from a record**, named in the table. Numbers
  the app shows (its ranges) stay the app's; the voice-over only says measured ones.
- **The cold open is real error text**, typed out as it was logged; no invented bill, price or
  "GPU hours wasted" figure anywhere.
- **Full model names**, each as the app or the caption says it at the moment it works: Muse Spark,
  Muse Spark Open, Muse Glimmer, Muse Code. Glimmer isn't in this cut (it's in the 3-minute
  video, shot 12); the reply under the post names it.

| Time | Picture | Voice-over | Caption (dark band) | Source |
| --- | --- | --- | --- | --- |
| 0:00 to 0:04 | Black. Monospace text types out, line by line, at the speed of a log: `OutOfMemoryError: CUDA out of memory.` / `Tried to allocate 530.00 MiB.` / `GPU 0 has a total capacity of 22.06 GiB of which 484.69 MiB is free.` The cursor blinks once after the last line. | "Your GPU is a terrible place to discover a typo." (10 words) | none: the text is the picture | The LoRA zoo script's eval probe, record 01M60H0R28WQS4R3XM0G38BAJP (A's production run, Oct 28, share `ICDYrfHQ2nBNbv3mlRHHMg`), probe `p1-eval`, status oom: its logged error, word for word, cut into three lines |
| 0:04 to 0:08 | Title card: "Preflight" set in the app's serif, then the hook under it. A slow sky gradient behind (the app's colours, below), a thin flight-path line drawn left to right. Clearly a graphic, not a screen. | "Preflight. The cheapest place for your training run to crash." (10 words) | none: the card carries the words | n/a |
| 0:08 to 0:16 | The live site, New flight: Repo link with `github.com/uchuuronin/preflight`, then open "Optional flight details" (closed at first; "Entry point, arguments, extra packages" beside its title) and type Entry point `zoo/finetune_lora_hf/train.py`, "Check in". Hard cut past the wait to Boarding ("Reading your script", "Checking the logs"), then the cockpit's map assembling. | "Point it at your repo. Muse Spark reads the script; real steps run on a real GPU." (17 words) | "Muse Spark (Meta) reads the script" over Boarding's first stage, then "Muse Spark Open (Meta's open model, on Baseten) reads each probe's log" over the second | The fold is C's (ws-c `9d824155`), live once C posts "live with the LoRA refill". Film a fresh check-in (one of Chiya's 3 jobs a day; about 5 to 11 minutes [measured: production's two LoRA check-ins], cut). Fallback: the Oct 28 record's share link, opened, with the check-in from shot 3's take |
| 0:16 to 0:24 | The cockpit's run timeline ("The whole run, step by step"): drag the playhead from step 0 to step 200; the eval node goes red at the first eval, with whatever status word the app puts on a measured out-of-memory (film it as it is). | "Model the hours from minutes. At step 200, the first eval runs out of memory. Measured, not guessed." (18 words) | "Step 200: the first eval runs out of memory · measured on a real GPU" | The filmed record: its `p1-eval` probe ran out of memory, so the app marks it measured. If the fallback is used, the Oct 28 record (the same probe, the same message) |
| 0:24 to 0:32 | **On the LoRA sample, PF102 (`#/cockpit/lora-qwen`)** (live with the LoRA refill, main `f52f3e44`, C, Nov 3, 7:53 PM: PF102 is now A's Nov 3 A10 record; the verdicts are on this sample, not on the fresh check-in): the **Eval loop** node, **Fix it with Muse Code**, **Write the fixes** (C's path on the live site): two lanes write side by side, each to "Fix written" (28 and 34 s live; cut the wait). When they finish, hold on the lanes: "Run evaluation without gradients" reads "An earlier patch for this fix, checked on 2 Nov on an A10: Rejected: the re-run still ran out of memory." in red; "Evaluation without gradients, in smaller batches · 8" reads "... Passed: eval peak 10.4 GiB, loss within 0.11%." in green. Each lane's own "Not verified: Preflight hasn't run this fix yet." stays in shot: it's true of the fresh patch. Film as it is, nothing drawn on top | "Muse Code writes the fixes. Preflight runs each one for real: one still crashes, rejected. The stacked fix passes." (19 words) | "Muse Code (Meta) · the LoRA sample · each fix re-run on a real A10 (2 Nov) · race cost $0.39" (the verdicts and the 10.4 GiB are on screen in the app's words, so the caption doesn't repeat them) | The verdicts: C's race, Nov 2 (GitHub run 41482457185, `docs/status/c.md` D8): round 1, `torch.no_grad()` alone, verify failed (record 01M6EHR42KZR8F3MWZJKAQBWNW); round 2, no_grad with eval batch 8, verify passed, eval peak 10.4 GiB, loss within 0.11% (record 01M6EHYYCM3JRHXKJ9XM5EPJ41); US$0.39 for that race [measured]. Writing the fixes on film costs up to about US$0.26 of Muse Code, paid by Preflight (C's lane cap); the verdicts shown are the Nov 2 checks, not of the patches written on film |
| 0:32 to 0:39 | The same cockpit before and after: the readouts at the record's own setting ("Likely runs out of memory", the dials marked "If it didn't run out of memory"), then after applying the guide's combined move (eval batch 8 and no_grad): the ring back under the line, each dial with its range. One cut between the two, no split screen. | "Before and after. Every readout comes with its range." (9 words) | none, so the app's own labels and ranges are read as they are | The filmed record, on screen only (predicted ranges are the app's; nothing here is spoken) |
| 0:39 to 0:45 | "Take this plan with you": the Plan (the settings as JSON), "Muse Code patches for this record", the "Slurm batch file", each opened for a beat. Cut to the end card: "Crash it here, not on your GPU bill." with the link and "Built Oct 26 to Nov 4 for the Meta hackathon". | "Take the config, patch and batch file. Crash it here, not on your GPU bill." (15 words) | end card: `preflight-cqn.pages.dev` · github.com/uchuuronin/preflight · "Built Oct 26 to Nov 4" | The exports panel on the filmed record (`c.exports`, on in production) |

98 words in 45 s, about 2.2 a second; no line is faster than 2.5 a second.

**Settled (C, Nov 3, 17:40):** before today no screen showed a race's verdicts (the site never
runs verify on a race; the Nov 2 verdicts came from C's script). C's ws-c `6a7fcd66` now puts each
lane's verdict under it, from Preflight's dated check of the same fix on the same sample, with the
"earlier patch" wording because the patch written on film isn't the one verify checked. It is live
after the PM's next merge and C's deploy: check it on the live site before filming. If it isn't
live on Wednesday, cut 0:24 to 0:32 rather than draw the verdicts, and give its 8 s to 0:32 to
0:39 and the end card.

**Making the graphics** (the cold open, the title card, the end card; the skills from the team's
Skills Audit doc, "Video" and "Animation and motion"):

- **`hyperframes`** (read first): the three cards as HTML rendered to video at 1920 x 1080, 30 fps,
  so the type matches the app exactly. `hyperframes-registry` before hand-building anything: its
  kinetic-type and terminal-style blocks for the typing cold open, and its titles for the card.
  `hyperframes-cli` to check frames and render.
- **`media-use`** for the sound: a low, quiet music bed, and soft key clicks under the cold open
  that stop with the cursor. Nothing loud on the "crash".
- **`animation-vocabulary`** to name the motions for whoever edits: a per-character "typewriter"
  reveal at a steady rate (no randomised jitter) for the error text; an "ease-out" fade and slight
  rise for the title; a "draw-on" stroke for the flight path.
- The screen recordings themselves (8 to 39 s) get no graphics, only cuts, a zoom where the text is
  small, and the caption band.

**The look**, from the app's own tokens (`web/src/theme/tokens.ts`), so the cards and the app read
as one film: the night sky gradient from `#0A0720` through `#563A88` and `#7A4389` towards peach;
text `#FFF4EE`; accent peach `#FFB98F` (the hook's key word); failure `#FF7A5C` (only on the error
text in the cold open, nowhere else on the cards); type: Instrument Serif for "Preflight" and the
hook, Instrument Sans for captions, JetBrains Mono for the error text (the same OFL fonts the share
cards bundle).

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

## Shots (Monday's version, kept for each shot's history)

Narration at about 2.5 words a second; every line is also a burned-in caption. About 2:55.

| # | Time | Screen and action | Voice-over | On-screen label | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | 0:00 | `zoo/finetune_lora_hf/train.py` in an editor; slow zoom to `EVAL_EVERY = 200` and the eval batch line | "This fine-tune looks fine. At step 200, its first evaluation, it runs out of memory." | "A real training script, with a planted bug" | **Works now** (an editor, no app) |
| 2 | 0:10 | Live site, home; the title fades in; click "Start a preflight" | "Preflight flies your training run before you pay for it." | "Built Oct 26 to Nov 4" | **Works now** |
| 3 | 0:18 | New flight: paste the repo link of `zoo/finetune_lora_hf`, pick "Finish soonest" and L4, scroll the data-use notice, tick "Got it", press "Check in" | "Point it at your code and say what matters. It tells you exactly what it sends." | | **Works now** (`runner.queue` on in production; the check-in flow passes D's stranger audit at both sizes and themes) |
| 4 | 0:32 | Boarding: the stages light in turn; hold on "Reading your script" with its tag, then "Checking the logs" with its tag | "Muse Spark reads the script and finds the training loop. A few real steps run on a real GPU, and Muse Spark Open reads every probe's log." | "Muse Spark (Meta)" while "Reading your script" lights; "Muse Spark Open (Meta's open model, on Baseten)" while "Checking the logs" lights | **Works now** for Muse Spark (tags come from the record's provenance since finding 12's fix). Spark Open's tag shows only if that check-in's probes were triaged: true on the live site, where `harness.spark_open_triage` is on. Film a fresh check-in and cut the GPU wait (D's A10 probe waited 84 s on Nov 2, `record_nov2_a10.json`; L4 waits have run from 8 to 59 min in A's and B's runs; since A's capacity fix, a job that can't get an L4 falls back to an A10, with the wait shown (Chiya, Nov 2)) |
| 5 | 0:47 | Cockpit of the Oct 28 record (`ICDYrfHQ2nBNbv3mlRHHMg`): the map draws, the memory ring fills past the line, the red finding "runs out of memory" | "In 189 seconds of probing, for about four cents of GPU at list price, it caught the crash before the real run." | "189 s of probing · $0.042 of GPU at list price · 20.4 of 22.06 GiB · one script" | **Should work; open it in a browser before filming:** share pages are live (C, 12:45 ET Nov 2: the Oct 28 record's `/s/` link, card and badge pass `scripts/share_page_check.py`; no one has opened it in a browser since). Open it from https://preflight-cqn.pages.dev/s/ICDYrfHQ2nBNbv3mlRHHMg |
| 6 | 1:02 | Click the Eval loop node: its detail sheet shows the rule checks' findings and the triage finding with its quoted log line | "Muse Spark Open says what went wrong, and quotes the line it read; code checks the quote is really in the log." | "Muse Spark Open (Meta's open model, on Baseten)" over the app's own "Found by Muse Spark Open triage" | **Checked locally on the committed copy; open the live share link before filming:** A's production re-run (`a-triage-lora-prod-2`, record 01M6EGH5YMCXPFPNY5SHPNSYV5, share https://preflight-cqn.pages.dev/s/Q3s5qyHKTNf-fCng4dwn6g, an A10) has the eval probe's triage finding quoting "Ran out of GPU memory during the eval pass. OutOfMemoryError: CUDA out of memory"; D checked A's committed copy in Chromium (Nov 2, 18:00 ET): the Eval loop node's sheet shows it with "Found by Muse Spark Open triage", desk and phone. It's a different record from shot 5's (an A10, not the L4): see the one-record draft below. The live share page hasn't been opened yet (D's workspace can't reach the live host) |
| 6b | (1:12) | A finished record's Findings: the finding labelled "Found by Muse Spark diagnosis" (where the memory timeline or the loss plot disagrees with the raw probe numbers); hover its number to show the claim check's mark | "Muse Spark reads the record's memory timeline and loss plot against the raw numbers, and code checks every number it cites." | "Muse Spark (Meta) · diagnosis, every number checked" | **Needs by Tuesday noon:** `harness.spark_diagnosis` and `harness.spark_claim_check` are on in production (PM, 7:35 PM Nov 2) and go live with A's runner redeploy and C's deploy; C's live check of one finished record's diagnosis findings (the PM asked C) gives the record to film. About 8 s, taken from shot 7's 12 s if both are ready; otherwise not in the 3-minute cut. Speak no number from the finding |
| 7 | 1:14 | The knob dock: drag "Eval batch" down; the ring drops under the line, the dials move; press "Possible outcomes" | "Move a setting and the whole run answers at once, with its range." | the app's "Predicted" on the dials | **Works now** (`c.hops` on in `flags.production.json`) |
| 7b | (1:24) | The LoRA script checked in with `learning_rate=1e-3` (ten times its author's 1e-4); its cockpit's range-test curve: loss against learning rate, the script's rate marked past the curve's lowest point, the suggested rate marked lower down | "Set the learning rate ten times too high and a hundred quick steps on a rising rate show it: past the bottom of the curve. Preflight suggests one where the loss falls fastest." | "Learning-rate range test · measured on an A10" | **Needs by Tuesday noon:** B's re-check of the range test's fixes, `probes.lr_range` on and the runner redeployed (A, the PM), and C's curve in the cockpit; the GPU check passed (section below). About 10 s; it takes 9b's slot if 9b is cut, otherwise it's not in the 3-minute cut |
| 8 | 1:26 | The guide panel: "Start the tour"; the guide proposes one knob move and moves it on the map; Accept | "Ask the guide, and Muse Spark moves the knobs that matter for your goal, one at a time, and you decide each move." | "Muse Spark (Meta)" | **Works now** (`c.guide.live` on). If the live guide is refused (its daily cap), the panel falls back to the rule-based tour: then drop the label and the line |
| 9 | 1:40 | Scroll to the exports: the Slurm file, the run command, the Weights & Biases helper | "Take the plan with you: a batch file, the command, and your settings and predictions in Weights and Biases." | | **Works now** (`c.exports` on in `flags.production.json`) |
| 9b | (1:50) | Terminal beside the cockpit: `preflight watch` running the real script; the cockpit's in-flight panel follows the live record; an alert line appears in both | "Run it for real and Preflight watches. The moment the run leaves its predicted band, it says so." | the app's "Predicted" band; no Meta model (no model call here) | **Works on a GPU** (D's A10 check, Nov 2: both alerts fired with step 5's change; on a CPU the step alert can't fire). **Needs by Tue noon:** Chiya's laptop check, else Colab on a T4; otherwise cut (section below) |
| 10 | 1:50 | Hard cut to the agent sample measured on Oct 31 (PF331): the agent map, then the caching finding open | "Not every job is a training run. Here a coding agent on Muse Spark ran 130 programming tasks through Preflight's proxy, and not one input token came from the cache." | "Agent eval on Muse Spark (Meta) · measured" · "0 of 905,739 input tokens from cache, 130 tasks, two runs" | **Works now** (sample PF331 says Measured; its record holds 0 of 246,463 for 30 tasks; the 130-task total is A's status, so it's a caption, not the screen) |
| 11 | 2:05 | The GRPO sample measured on an L4 (PF510): the RL band | "In this GRPO run, 15 of 20 prompts had no reward spread at all, so they teach nothing." | "GRPO on an L4 · measured · 15 of 20 prompts with no reward spread" | **Works now** (sample PF510 says Measured; `engine.rl` on in `flags.production.json`) |
| 12 | 2:15 | New flight with "Private triage" ticked; then the record of a private check-in: its triage finding | "For code that mustn't leave, private mode triages on Muse Glimmer, Meta's open weights, on our own GPU." | "Muse Glimmer (Meta's open weights, on Preflight's own GPU)" | **Works now:** C's private check-in (Nov 2, 11:03 ET), record **01M6EG0284PWZDRXGGQVVZYVND**, triaged by Glimmer (four `modal-glimmer` / `muse-glimmer` triage calls in its provenance, as C reports; finding "Muse Glimmer on probe p1-eval: oom"), and the live cockpit shows "Found by Muse Glimmer" on it since C's 12:45 deploy. Its link stays private (C, Nov 2): ask C for it to film. The "Private triage" box is live (its two flags, `harness.spark_open_triage` and `harness.glimmer_private`, are on in `flags.production.json`; C's check-in used it). If it fails on the day, the captioned card over the agent sample ("The same 26 tasks: Muse Spark 23, Muse Glimmer 15, Muse Spark Open 14 ended on a tool call", A's `a-pareto-*` runs) stands in |
| 13 | 2:27 | Checks page: scroll "Each check": the caught OOM and the misses | "Every prediction is checked against real runs, misses included." | "Held-out checks on real runs" | **Needs:** the checks page from B's final table (regenerated Nov 3). If its checks still say "not measured", cut the shot and give its 10 s to shot 14 |
| 14 | 2:37 | Copy the cockpit's address, open it in a new tab: the same flight opens | "Every flight is a link anyone can open." | | **Works now** (the link is the page's address). The share page's preview card needs `workers/` in the deploy; not shown here |
| 15 | 2:45 | End card, about 10 s | "Preflight. Fly the run before you pay for it." | link · repository · "Built Oct 26 to Nov 4" | **Works now** |

## Draft for Chiya, not applied: one record for the fine-tune shots

Shots 5 and 6 now show two records: the Oct 28 L4 record (`ICDYrfHQ2nBNbv3mlRHHMg`, shot 5's
numbers, no triage) and A's Nov 2 A10 re-run (`Q3s5qyHKTNf-fCng4dwn6g`, shot 6's triage). One record for both
would mean every spoken number re-taken from the re-run's record (A's committed copy,
`api/tests/fixtures/a-triage-lora-prod-2.record.json`) [measured]:

- Shot 5's voice-over would become: "In about two and a half minutes of fresh probing, for under
  five cents of GPU at list price, it caught the crash before the real run." From the record: p1
  110.5 s and p1-eval 44.6 s of A10 GPU (155 s); p2 and p3 came from Preflight's probe cache (no GPU
  time); GPU US$0.047 at the A10's list price; model calls US$0.026.
- Its caption: "155 s of fresh probing · $0.047 of GPU at list price · 21.63 of 22.06 GiB in use ·
  one script" (the eval OOM's own line: "GPU 0 has a total capacity of 22.06 GiB ... Process 1 has
  21.63 GiB memory in use").
- **What changes on screen:** this record's `memory.peak` estimate abstains, so the memory ring
  doesn't fill past its line as in the Oct 28 record; the headline "Likely runs out of memory on
  A10 (100% chance)" is there instead (seen in D's Chromium check). Shot 5's action would hold on
  the headline and the red finding, not the ring.
- Shots 1 and 2 don't change. The X posts (`docs/submission/x-post.md`) would say "on one A10 ...
  155 seconds of probing, for $0.047 of GPU at list price" and link `/s/Q3s5qyHKTNf-fCng4dwn6g`; that link's card
  isn't pre-rendered.
- The submission's numbers table (`docs/submission/README.md`) would need these rows added before
  any of it is spoken.

The trade: one record and a live triage on it, against the Oct 28 record's ring and its
pre-rendered card. Chiya's call.

## Shot 7b: a planted bad learning rate (Chiya, Nov 2, 17:33)

`d-lr-range-planted-1e-3` (GitHub 41538650072, Nov 2, 22:36 UTC, `ws-d` `9b6ddb1`, an A10, US$0.14
[derived: 469 s of container at list price]) [measured]: the LoRA zoo script with its
`learning_rate=1e-4` changed to `1e-3`, then the range test and 100 steps held at each rate.

- **It catches it.** The range test reads the curve the same way as on the unchanged script (the
  ramp sets the rate, not the script): suggested 7.9e-5, the smoothed curve's lowest loss at 7.7e-4 (the raw curve's at 3.4e-4), divergence at
  1.0e-2, and the ramp checked. The script's 1e-3 is **past the curve's lowest point**, 12.6 times
  the suggestion and 10 times below where it diverges.
- **The suggestion is sane:** 7.9e-5 is 0.8 times the author's own 1e-4.
- **What 1e-3 costs here, honestly:** not a blow-up. Over 100 steps on the same batches, the loss
  at 1e-3 sat above the suggested rate's in every 10-step block after the first, by 0.01 to 0.05
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

**On a GPU it works: the steps below, checked on an A10** (`d-watch-gpu-9b`, GitHub 41539605466,
Nov 2, 22:43 UTC, `ws-d` `406fa38`, US$0.10 [derived: 315 s of container at list price])
[measured]:

- The record got bands: its GPU setting is the card it ran on (A10), `time.step` measured at 1.4 to
  2.2 ms a step, `memory.peak` extrapolated at 0.27 to 0.36 GiB (no probe ran the optimizer step). (On a CPU, `time.step` abstains: "the probes ran
  without a known GPU".)
- With step 5's change, **both alerts fired**: `step 201: peak memory 0.40 GiB, above the predicted
  0.27–0.36 GiB`, then `step 401: 0.00329 s a step, above the predicted 0.0014–0.00216 s`, about
  1.3 s into the run [derived: 401 steps at 300 a second; the log has no timestamps].
- Speed: about 1,250 steps a second unchanged, 300 with the change, so about 18,000 changed steps
  make a minute on an A10.
- **One false alarm to know about:** the unchanged script, watched as a control, also alerted once:
  `step 401: 0.000784 s a step, below the predicted 0.0014–0.00216 s`. The real run steps faster
  than the probe's timed steps on this tiny model, so "below the band" fires with no change. Don't
  film the unchanged run, and don't say the alert means something is wrong; it means the run left
  its band. Told B.

**The decision (Chiya, Nov 2, 17:33: "propose a way to film it on a real GPU run, or cut it";
decide by Tue Nov 3, noon ET):**

1. **First, your laptop's RTX 4070, if you have the time (about 5 minutes):** steps 1 to 4. If step 4 prints a `time.step` range,
   film there: a real GPU, a local file, the cockpit following it live. By the code it should: the
   engine treats the card the probes ran on as no move even when it's outside its GPU table
   (`_gpu_move` in `engine/api.py`), so the step time stays measured; that's read from the code,
   not run on that card.
2. **If it abstains: Colab with a T4** (free; the T4 is in the GPU table), steps 1 to 5 and 7 in
   Colab's terminal. The alert lines appear live in that terminal; the cockpit can't follow a file
   on Colab, so download the live record when it stops and open it in the in-flight panel (it reads
   once), and caption that part "the run's record, opened after". Not checked on a T4.
3. **If neither gives a step range by Tue noon: cut 9b** (its 10 s goes to 7b if that's ready, else
   to shot 13 as planned).

From a clone of the repository:

1. `pip install -e ".[probe]"` (the repository isn't public until Nov 3 to 4; after that,
   `pip install "preflight[probe] @ git+https://github.com/uchuuronin/preflight"`).
2. `mkdir film && cp zoo/notebook_tiny/tiny_train.py film/`, and in `film/tiny_train.py` set
   `max_steps = 18000` (about a minute on an A10 with step 5's change; another GPU differs).
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
   `batch_size = 4096`. On the A10 this fired both alerts (above).
6. Open `https://preflight-cqn.pages.dev` in Chrome or Edge (they follow the file; other browsers
   read it once), go to the Logbook and open "Watch a real run" (needs `c.inflight` on in
   production).
7. In a terminal beside it: `preflight watch records/<share>.json film/tiny_train.py --window 200`.
   It prints the live record's path (`records/<new>.json`). In the browser, choose that file (not
   the `.live.jsonl`); the cockpit follows it, rereading every 3 s.
8. Expect `[preflight] step 201: peak memory ... GiB, above the predicted ...` after the first
   window, and `step 401: ... s a step, above the predicted ...` after the second (as on the A10);
   the in-flight panel shows the same alerts. Film from step 7 to the alerts.

**Needs, in order:** the laptop check (decision step 1), else Colab; `c.inflight` on in production
(on since the PM's 4:25 PM merge, Nov 2; live once C deploys); then Chiya's take. The GPU test is
done (above), and the stopgap step is gone: `preflight run` fills predictions (B's fix, on `main`).

## Before filming (as soon as C posts the live deploy)

Chiya, Nov 3, 18:12: no fixed time; film as soon as C posts the live deploy, with the product
complete by then. Shot 9b on Chiya's RTX 4070 on native Windows (`docs/demo/shot-9b-laptop.md`);
the clip's 0:24 to 0:32 on C's race lanes on the live LoRA sample. D ran the Windows check on
`main` `2b62f25c` (run 41739673434, passed); the CLI is the same on `f52f3e44`.

- [x] C's deploy is posted, and the live site shows the race lanes' verdicts on PF102 (C, Nov 3, 7:53 PM: "Live with the LoRA refill", main `f52f3e44`).
- [ ] Each "needs" row above is ready, or cut as its row says.
- [ ] Every number spoken or captioned is in the submission's numbers table.
- [ ] Each Meta model named is labelled on screen while it works, and its record shows the call.
- [ ] No sample that says "Illustrative" is on screen while the narration talks about
  measurements.
- [ ] Captions burned in for every line, at least 2 s each, white on a dark band.
- [ ] The repository is public and the end card's link opens.

## What is left out, and why

- **The range test (7b):** `probes.lr_range` is off in production.
- **Search grounding:** in since Nov 3, 21:20 ("The edit as built", item 6).
- **The campaign calendar: in** (the PM, Nov 3, 21:21). The earlier reason here, "its sample is
  illustrative", was wrong: PF420 has been measured since Nov 2 (C's D51: rebuilt from the four A10
  cells measured on Nov 2). It gets an 8-second shot (6 s in the short cut, D's call to keep that cut
inside the track): the sample's "12 probes"
  chip, then its Campaign calendar with each person's runs and when the last one lands.
- **The divergence warning:** its classifier misses (AUROC 0.66 against 0.85) and isn't used.
