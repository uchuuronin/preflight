# Status: D (usability add-ons)

## Decisions while Chiya is out (Nov 2, 11:15 to about 17:00 ET)

| # | When (ET) | What | Why | Cost | How to undo |
| --- | --- | --- | --- | --- | --- |
| B5 | 15:40 | Watcher for the PM's afternoon "main is ready" stopped at its time limit with no 2:30 post (main's last is 1:15, `12058ca`); checked by hand since, the 4:15 one still to come | Its limit | none | none needed |
| B6 | 15:45 | Took A's review of the range test (ws-a `from-a.md`) as "its review is back" (the 12:53 list's item 2) and fixed its items 1, 2, 3 and 6 on `ws-d`; 4 asked of B (the engine's `OK_PROBE` filters are B's), 5 of A (`scripts/make_golden.py` is A's) | Those are the owners; the flag stays off until A re-checks | none | Revert the commit |
| B7 | 15:45 | "Checked" now needs the updates to grow with the ramp (slope of log update-to-weight on log rate at least 0.5 [guess]); with too few measured steps the rates are dropped as `unverified` rather than shown unchecked | A suggestion we can't show took effect shouldn't read as one; dropping it is the conservative side | none | Lower or drop `UPDATE_SLOPE`, keep rates on `unverified` |
| B8 | 15:40 | Installed CPU torch, transformers and accelerate in my local environment to run the recorder tests (CI's torch job installs the same) | A's review listed tests that need torch | none | `uv sync` |
| B9 | 16:25 | Merged `main` `8ec03bc` (the PM's 4:25 PM post) into `ws-d`; one conflict in `tests/test_pipeline.py` (A's import-knob test beside my range-test tests), kept both | The PM's "main is ready" | none | Revert the merge commit |
| B10 | 16:25 | Re-ran the range test's GPU check (`d-lr-range-gpu-2`, the same app and scripts, A10 or L4) with the new "checked" | The 12:35 run's "checked" was the old test, which proved nothing; the new one should be seen on a GPU before A re-checks | about US$0.19 [guess: the 12:35 run's cost], inside item 3's US$2 | None needed (an ephemeral app) |
| B11 | 16:30 | The PM's 4:25 list gives D "A's review items on search grounding": fixed (no retries on `/responses`, the search fee recorded, `max_tool_calls` 4, at most 2 searches a job and none started after 120 s, the error text marked as data and its own addresses and Preflight's host never cited, the exact annotated URL stored); flag still off | The PM's order; A's six items | none yet | Revert the commit |
| B12 | 16:30 | The search fee is US$2.50 per 1,000 queries, cited from eesel.ai's summary of Meta's pricing page, not Meta's own page (my fetch can't reach it) | Recording $0 for a fee we know exists is worse than a cited figure; the tag says where it's from | none | Change `SEARCH_USD_PER_QUERY` |
| B13 | 16:30 | Shot 9b's stopgap step (filling predictions by hand) replaced by a check that the record has bands, after B's CLI fix reached `main`; checked on a CPU (memory band yes, step-time band no) | The 12:53 list's "the in-flight steps once B's GPU test and the CLI predictions fix land": the fix landed; B's GPU test of the shot's own script hasn't | none | Restore the step from git |
| B1 | 12:55 | Merged `main` `7ba177b` into `ws-d` (no conflicts) | The PM's "main is ready" (12:53) | none | Revert the merge commit |
| B1b | 13:20 | Merged `main` `12058ca` into `ws-d`; one conflict in `runner/sources.py` (A's new `data` parameter for `probes.repo_install` against my `_git_read`/`_clone_read`), kept both | The PM's 1:15 "main is ready" | none | Revert the merge commit |
| B2 | 13:00 | Big repositories (Chiya's new item 1) are read as the entry point plus its import closure only with a new flag, `runner.read_closure` (off); with it off, the refusals now say how big the source is and carry `submission.too_large`, not `syntax_error` | Her 11:15 rule: runner and harness changes behind new flags, off, for A's one redeploy. The message fix is a plain bug fix, so it isn't flagged | none | Leave the flag off, or revert the commit |
| B3 | 13:00 | The closure is static (`import a.b`, `from a import b`, relative imports, package `__init__.py` files, the repo top, `src/` and the entry point's folder as roots), breadth-first, nearest first, bounded at 600 files and 2 MB (the static mapper's own budget), so a partial read never fails in the mapper | Imports by string (registries, importlib, pyproject entry points) can't be followed without running the code; 2 MB keeps the map and the read the same | none | Constants in `runner/sources.py` |
| B4 | 13:00 | The record carries an info finding (`source.partial`) whenever the source was read or mapped in part, naming what was read and what was left out | A stranger must be able to tell a finding-free record from a partly read one | none | Drop `_reading_findings` |
| A1 | 11:30 | Added `harness.spark_search` and `probes.lr_range` to `flags.json`, both off | Chiya's items 1 and 3 each name a new flag, off | none | Remove the two keys |
| A2 | 11:35 | Before building item 1, a probe script (`scripts/spark_search_probe.py`) asks Meta's Responses API one question with `web_search`, `web_search_preview` and no tool, to learn whether Muse Spark can search and how it cites | Meta's docs I can reach don't say whether a search tool exists or how citations come back; building on a guess would waste the cap | a few cents of Meta credit (counted against item 1's US$1) | Delete the script |
| A3 | 11:40 | Item 1 built as a Responses API call with Meta's `web_search` tool (`harness/search_grounding.py`, `ModelClient.respond`), wired after triage in `runner/pipeline.py` behind `harness.spark_search` (off) | A2's probe showed Muse Spark searches and returns `url_citation` annotations. Code keeps a source only if the response annotates its URL (and any attempt to open it completed); with none left the finding says "no known issue found". That stops a made-up link only if Meta annotates nothing but search results, which the validation tests with a planted fake link. Never for a private or verify job; never for Preflight's own failures that name no library (budget, no GPU, sandbox ended, timeout, too few steps); once per distinct error per job; the error's URL queries, key-shaped strings and home folder names masked before sending | none (no call while the flag is off) | Revert the commit; the flag stays off either way |
| A4 | 11:40 | Validation cases are every distinct failed probe in the non-unseen records on `remote-runs` (10; 7 reach the model, 3 are skips), frozen in `tests/fixtures/spark_search_cases.json` | Chiya named the zoo's real failures; the records hold the exact error text the pipeline would send. Unseen runs excluded by name | none | Delete the fixture |
| A5 | 11:50 | Item 2's check, "does `preflight watch` work on a CPU run", runs as `scripts/d_watch_cpu_check.py` through `remote-run.yml` on GitHub's CPU runner (it installs CPU torch, as CI's torch job does), not on my workspace | My workspace has no torch and can't reach PyTorch's wheel index; the runner can, with no GPU and no model call | none (a GitHub runner's minutes) | Delete the script |
| A6 | 12:00 | The in-flight shot goes into `video-plan.md` as shot 9b, after the exports, needing B's GPU test, and taking its 10 s from shot 13 only if 13 is cut | My CPU check showed `watch` works on a CPU but no alert fired (CLI records carry no predicted bands: the CLI never fills predictions), so the shot needs a stopgap step and a GPU test; the plan is full at about 2:55, and 13 is the shot most likely cut | none | Delete the 9b row and its section |
| A7 | 12:10 | Item 3's range test is set in the recorder's optimizer pre-step hook (every param group, keeping its ratio to the largest rate), not by rewriting the script's learning rate | A hook overrides any schedule the script runs (nanoGPT's per-step rate, the HF Trainer's scheduler) and needs no knob; rewriting the code would leave the script's warm-up in charge | none | `Recorder(lr_range=None)` (the default) leaves every probe as before; revert the commit |
| A8 | 12:10 | The range probe `p1-lr` keeps its curve and reading in extras and drops its memory, timings and loss series; it goes into the record but not into triage, search or the fits | Its steps ran at rates the script never uses, so its loss would mislead the loss and divergence models; its memory duplicates p1's | none | Same |
| A9 | 12:10 | Its GPU check is a separate ephemeral Modal app (`runner/lr_range_app.py`), not the deployed runner or A's `ops_app.py` | The runner isn't redeployed until A finishes the unseen scripts; an ephemeral app runs this branch's code on the runner's probe image, and keeping it out of `ops_app.py` leaves A's file alone | up to about US$1.10 (two 30-min containers on an A10) against item 3's US$2 | Delete the file |

## Decisions overnight (Nov 2; Chiya offline 00:16 to about 05:30 ET)

| # | When (ET) | What | Why | Cost | How to undo |
| --- | --- | --- | --- | --- | --- |
| 1 | 00:00 | Third eval-sweep try on an A10 rather than the L40S | Chiya allowed either for one more try under $1; the A10 is cheaper (US$1.10 an hour against the L40S's US$1.95 [cited: `probes/gpu_choice.py`]) and A's ground truths ran on it | US$0.12 by the record | Nothing to undo; the record is `zoo/eval_sweep_reasoning/record_nov2_a10.json` (delete the file to drop it) |
| 2 | 00:15 | Item 19 marked done with that A10 record; no further L4 tries | It measured the planted bug (8 of 16 cut off); the two L4 failures are A's runner question, asked in `from-d.md` | none | Re-open item 19 in this file (the 00:15 entry) and add a line to the item-19 request in `from-d.md` |
| 3 | 00:20 | Didn't read Modal's logs | My only route is `remote-run.yml`, and a log-reading script would be a new kind of run, which still needs a yes; A is adding a "waiting for a GPU" event | none | Say yes and I add a read-only `scripts/` log reader |
| 4 | 01:10 | Ran the stranger audit on the deployed commit served by Vite in production mode, not the live host | The live host is refused by my workspace's network allowlist; the dev server has the same code and flags, with the API mocked (not the built bundle) | none (local) | Re-run `docs/audit/scripts/stranger.mjs` against the live URL from a machine that can reach it |
| 5 | 01:10 | Committed four audit screenshots (about 178 KB) and the audit log under `docs/audit/nov2-stranger/` | C needs the evidence for S1 and S2, and my scratch folder isn't visible to C | none | `git rm docs/audit/nov2-stranger/*.jpg docs/audit/nov2-stranger/*.png` |
| 6 | 01:20 | Re-ran CI's failed `python` job on `49e36af` instead of changing code | It failed in A's `tests/test_agent_recorder.py::test_parallel_calls_cant_spend_past_the_cap` with "Connection reset by peer", a network flake in a file my push didn't touch; the re-run passed | none | Nothing to undo; A can harden the test |
| 7 | 01:45 | Wrote `docs/demo/video-plan.md` as the plan of record for Tuesday's 3-minute video, superseding the plan A and B shot lists in `storyboard.md` and `fallback-cut.md` (kept as drafts; the 45-second cut there is unchanged) | Chiya's item 3 asks for live-site flows that work today with every Meta model labelled, which the older plans predate (the guide, triage and Glimmer) | none | Point the PM back at `storyboard.md` and delete `video-plan.md` |
| 8 | 02:00 | In the portal and numbers table, replaced "60% within 20% on an unseen GPU" (four GPUs) with B's five-GPU result, 57% of 6,694 cases | Chiya's item 4 names the five-GPU table; it's B's newer, wider measurement (`calibration-l4-phase1.md`, "The Nov 1 top-up") | none | Restore the 60% row and sentence from git |
| 9 | 02:05 | The X post links the cockpit, not the share page, with two variants (with and without the triage shot) | `deploy.yml` doesn't ship `workers/`, so `/s/<token>` isn't live; a cockpit link opens but previews as text | none | Swap in the `/s/` link once `workers/` deploys (TODO(PM) in the file) |
| 10 | 04:10 | Audited the deployed `1ea2dd6` with `d.notebook` held off (`VITE_FLAGS`), and `main` `6f49c9c` with it on, rather than only one | The first is what visitors have now; the second is what the next deploy brings, with the Notebook tab C ships | none (local) | Nothing to undo |
| 11 | 04:15 | `readme-final.md` takes the PM's 03:55 `README.md` edits: the notebook tab on in production (A's `a-notebook-checkin-3`) and Glimmer's private mode "off until the data-use notice describes it"; my behind-a-flag notebook row is gone | `readme-final.md` stays the source of truth until Tuesday's swap (Chiya), so it must carry the PM's changes | none | Revert the commit |
| 12 | 05:30 | The free-tier chip's "Not trusted yet" now reads "Not measured here" (the PM's 4:00 ask, which I'd missed at 4:15); and I carried the same word into the share cards (`workers/functions/_card/`), their README, `readme-final.md`, `portal.md` and the X post's reply | C renamed the abstain word app-wide; the cards promise "the app's own word for its status", so they must match | none | Revert the commit; the hero card re-renders with `npm run hero-cards` |
| 13 | 05:30 | Re-rendered the hero card (`workers/static/cards/ICDYrfHQ2nBNbv3mlRHHMg.png`) | Its footer names the status words; `hero-cards:check` fails until it's re-rendered | none | Comes back with the revert above |
| 14 | 05:35 | Audited C's deployed `5058361` from a `git archive` of that commit, with its own flags | `main` had moved on (`ae0e900`, Glimmer's private mode on), and the audit must show what visitors have | none (local) | Nothing to undo |


## Cloud runs

Every run through `remote-run.yml` or `amd-run.yml`: date, GPU, minutes, dollars.

| Date | GPU | Minutes | Dollars | What |
|---|---|---|---|---|
| Nov 2, 02:56 UTC | L4 (asked) | about 20 (cap) | at most about 0.60 [guess]; 0.0179 of Muse Spark by the record | `d-eval-sweep-l4` (GitHub 41380341849): cancelled while probing, no measurement |
| Nov 2, 03:19 UTC | L4 | about 22 | 0.6423 by the record | `d-eval-sweep-l4-2` (41382389659): probe ended `recorder` after 1,280 s, no measurement |
| Nov 2, 04:59 UTC | A10 | about 5 | 0.1203 by the record | `d-eval-sweep-a10` (41391396558): the eval-sweep record, one probe ok |
| Nov 2, 16:19 UTC | none (Meta API) | about 1 | about 0.053 of Muse Spark [derived: tokens at list price; any search-tool fee unknown] | `d-spark-search-probe` (41480275641): `web_search` and `web_search_preview` both work, with `url_citation` annotations; no tool, no citations |
| Nov 2, 16:52 UTC | none (Meta API) | about 2 | 0.107 of Muse Spark [derived: tokens at list price; any search-tool fee unknown] | `d-spark-search-validate` (41485593441): 8 calls on the zoo's failures, 1 source kept, 0 made up or irrelevant (`docs/decisions/spark-search.md`) |
| Nov 2, 16:52 UTC | none (GitHub CPU runner) | about 3 | 0 | `d-watch-cpu-check` (41485598315): `preflight watch` on `zoo/notebook_tiny` on CPU works; no alert, as CLI records have no bands |
| Nov 2, 17:20 UTC | A10 (both scripts) | about 11 (633 s of containers) | 0.19 [derived: container time at list price] | `d-lr-range-gpu` (41490183749): the range test on LoRA and nanoGPT, both train at the suggestion (`docs/decisions/lr-range.md`) |
| Nov 2, 18:21 UTC | none (Meta API) | about 2 | 0.110 of Muse Spark [derived: tokens at list price] | `d-bigrepo-spark-map` (41499720563): Muse Spark's script map on the three big repositories, flag on |
| Nov 2, 21:24 UTC | A10 (both scripts) | about 10 (596 s of containers) | 0.18 [derived: container time at list price] | `d-lr-range-gpu-2` (41528070349): the range test with the new "checked"; both scripts checked, the same suggestions (`docs/decisions/lr-range.md`) |
| Nov 2, 21:37 UTC | none (Meta API) | about 1 | 0.142 of Muse Spark [derived: tokens at list price plus the search fee, US$2.50 per 1,000, second-hand] | `d-spark-search-validate-2` (41529989287): the fixed search on the same cases; Meta takes `max_tool_calls`; no annotations this time, so the known issue's three sources were dropped (`docs/decisions/spark-search.md`) |

## Mon Nov 2, 16:55 ET (both re-runs: the range test checked on a GPU; search checked on Meta's API)

CI on `5dcf16a`: green, all seven jobs.

- `d-lr-range-gpu-2` (A10, US$0.18 [derived]): both zoo scripts read **checked** under the new test,
  with the same suggestions (LoRA 7.9e-5, nanoGPT 2.5e-4); details in `docs/decisions/lr-range.md`.
  Item 3's spend: US$0.37 of its US$2.
- `d-spark-search-validate-2` (Meta API, US$0.142 [derived, fee second-hand]): Meta takes
  `max_tool_calls`, and its search actions are typed `search`. 6 of 7 "none found" as before; the
  known pyarrow issue's three sources (GitHub pull requests naming the exact conflict) were all
  dropped because this response carried no citation annotations at all (the first run kept one).
  So recall under the annotation rule is the open question for turning it on; told A. The
  finding no longer says "found none" when it dropped sources it couldn't check.

This push sends: `harness/search_grounding.py` (the dropped-sources wording),
`tests/test_d_search_grounding.py`, `docs/decisions/spark-search.md`, `docs/decisions/lr-range.md`,
`docs/requests/from-d.md`, this file.

## Mon Nov 2, 16:35 ET (the PM's 4:25 list: A's search grounding review fixed; main merged; shot 9b)

The PM's 4:25 PM "main is ready" (`8ec03bc`) merged into `ws-d` (one conflict, both sides' tests
kept); 1,721 non-GPU tests passed on the merge (CPU torch). The PM's list for D: "fix A's review
items on search grounding and the range test". The range test's went out at 16:20 (CI green,
`7263013`). Search grounding's six items are fixed, still off: `docs/decisions/spark-search.md`,
"A's review (Nov 2) and what changed". Two rounds of read-only review: the first found that an
error could still get its own page cited (an address ending a sentence, without a scheme, another
path on the same host) and that nothing had been checked against Meta's real API; the first is
fixed, the second is the validation re-run after this push (about US$0.11 plus the search fee).
The second round found nothing blocking for the merge, and one thing before the flag goes on (a
library's error naming its own docs dropped those docs, e.g. NumPy's migration guide), fixed with
library docs hosts read page by page, plus three small holes.

Shot 9b: `preflight run` fills predictions now (B's fix, on `main`), so the video plan's stopgap
step is replaced by a check the record has bands. On a CPU, `memory.peak` had a range (604 to 978
MiB) and `time.step` abstained. B's GPU test of the shot's own script is still asked.

This push sends: the merge of `main` `8ec03bc`; `python/preflight/harness/search_grounding.py`,
`harness/spark.py` (`_post`'s `retries`, `respond` never retried), `runner/pipeline.py` (search
cap), `tests/test_d_search_grounding.py`, `tests/test_pipeline.py`, `docs/decisions/spark-search.md`,
`docs/demo/video-plan.md` (shot 9b), `docs/requests/from-d.md`, this file.

## Mon Nov 2, 16:20 ET (12:53 item 2: the range test, A's review fixed; still off)

A's review of the range test came back (ws-a `from-a.md`): ready to merge off, six fixes before
the flag goes on. Items 1, 2, 3 and 6 fixed here; 4 asked of B and 5 of A (their files). What
changed, and the CPU tests for each: `docs/decisions/lr-range.md`, "A's review (Nov 2) and what
changed". The main one: "checked" now reads whether the updates grew with the ramp, not the rate
the recorder had just written, so the GPU check's "the ramp checked" (12:35) doesn't count; the
GPU check is re-run after A's re-check (US$0.19 last time, inside item 3's US$2).

Three rounds of read-only review on the fixes. First: two blocking (zero-start weights such as
LoRA's B made a ramp that took effect read as not set; the check app's single-rate runs came
back empty), fixed. Second: one blocking (fp16 accumulation of 64 stopped before the first
step), fixed by reading a stalled scaler from its scale's drops. Third: nothing blocking; its
suggestion (each scaler tracked on its own) taken. Tests (CPU torch, transformers and accelerate, CI's
non-GPU set): 1,692 passed before the review fixes, 1,699 after the first round; the range tests
(32) after the last two.

This push sends: `python/preflight/probes/lr_range.py` (`follows_rate`, `losses`),
`recorder/runtime.py` (the range test's update sample, `initial_lr` ratios, other and rebuilt
optimizers, the stall rule), `recorder/run.py` (four summary keys, range tests only),
`probes/local.py` (`with_lr_range`'s reasons), `runner/pipeline.py` (the timeout cap, a cached
`p1`), `runner/lr_range_app.py`, `tests/test_d_lr_range.py`, `tests/test_d_lr_range_probe.py`,
`tests/test_pipeline.py`, `docs/decisions/lr-range.md`, `docs/requests/from-d.md`, this file.

## Mon Nov 2, 13:40 ET (item 1: Muse Spark's map on the three repositories)

CI on `ad6290e`: green, all seven jobs.

`d-bigrepo-spark-map` (GitHub 41499720563, US$0.110 [derived], three calls of 19,002 to 20,905 input tokens, all ok) [measured], with `runner.read_closure` on, each repository fetched and mapped as a check-in would:

| Repository | Static map | With Muse Spark | Judged |
| --- | --- | --- | --- |
| lm-evaluation-harness | `pretrain`, no knobs | `eval`, framework `other`; risks name it evaluation-only | Right lens now; no knobs (likely because its settings are command-line flags; the run doesn't show why) |
| vLLM | `serving`, `vllm`; knobs batch_size 1, max_new_tokens 16, seq_len 1024, temperature, pin_memory, precision | the same, plus group_size and n_tasks; its risks say the static map's temperature, pin_memory and precision came from files other than the entry point, and that it invented a Qwen3-0.6B model and training settings and an agent loop | Right lens; the merged map still keeps the knobs Muse Spark flagged, so its knobs can't be trusted as the benchmark's settings |
| torchtune | `pretrain`, knobs gpus 2 and temperature | unchanged: `pretrain`, the same two knobs (though its own risks say single-device only) | Wrong lens, and no real knobs. The repository was read whole and its YAML configs were among the mapped files, but nothing links the recipe to the config its command names (`qwen2_5/1.5B_lora_single_device` in the unseen list's command; this run passed no arguments) |

So the fix ends the refusals, but two of three still map poorly, and the records say what was read, not that the map is right. That's the mapper's reading of big code (A's, with Muse Spark), not the reading: three follow-ups, none built:
- **Settings from imported files:** with a closure, the static map's knobs can come from library code (Muse Spark says so of three of vLLM's), and the merge keeps them. Knobs should come from the entry point and the files its settings flow through, not every file read.
- **Configs named on the command line:** torchtune's settings are in a YAML file its command names; linking the recipe to that file (from the arguments) would give the map its knobs.
- **Lens on recipes:** the static reader calls a LoRA recipe `pretrain`, and Muse Spark kept it.

Item 1's spend: US$0.11 of its US$1. Told A in `from-d.md`. Fact-check (read-only, against the run's output): numbers matched; it caught "library defaults" naming knobs the output doesn't attribute, "honest about what they read", the torchtune config cause (the configs were read; nothing links them), and the lm-eval reason as stated fact. Corrected.

This push sends: `docs/status/d.md`, `docs/requests/from-d.md`.

## Mon Nov 2, 13:25 ET (Chiya's new item 1: repositories too big to read; built behind `runner.read_closure`)

Built (B2 to B4): `harness/closure.py` (the entry point's static import closure, breadth-first, nearest first, bounded at 600 files, 2 MB and 30 s, with caps on hostile names and lines), `runner/sources.py` (a repository over 2,000 files or 25 MB of text is read as its entry point and its closure, with the flag on; with it off, or with no entry point named, the refusal says how big it is), `harness/static.py` `mapped_files` (over 2 MB of Python, the map reads the closure; an entry point over 2 MB alone, or not a Python file, is refused saying so), the Spark mapper's prompt ordered by import distance (flag on), and an info finding `source.partial` saying what was read. The refusals carry `submission.too_large`, never `syntax_error`, with a hint that fits.

On the three pinned repositories (`docs/validation/unseen-scripts.md`; local, static map only, no spend) [measured]:

| Repository | Flag on | Flag off |
| --- | --- | --- |
| lm-evaluation-harness (`lm_eval/__main__.py`) | read: the entry point and 46 Python files it imports, of 15,449 text files | refused: "the repository (15,449 text files, 12 MB) is more than Preflight reads" |
| vLLM (`vllm/benchmarks/throughput.py`) | read: 160 imported files, stopped at 2 MB with at least 206 more left out; lens serving, framework vllm | refused: 6,510 files, 71 MB |
| torchtune (`recipes/lora_finetune_single_device.py`) | the whole repository fits; the map reads the recipe and 195 imported files (1.6 of 3.8 MB of Python) | refused: "3.77 MB of Python in 498 files, more than Preflight maps" |

Still wrong after the fix, and not this item's: the static lens. torchtune reads as `pretrain` with two knobs, lm-eval as `pretrain` with none. Muse Spark's map runs next (`scripts/d_bigrepo_spark_map.py`, cap US$1) to see whether it corrects them. And a probe runs only on what was read, so a partly read repository can still stop at an import (the finding says so).

Review (read-only, git-free copy): one high finding (a hostile dotted name could use gigabytes in the resolver) and medium ones (an entry point over the budget never read, a non-Python entry point crashing the mapper, a hint that didn't fit, `setup.py` followed and counted, every root's match taken instead of the first, `TYPE_CHECKING` imports followed); all fixed, with tests. The second pass found a module hidden by a data folder of the same name (`data.py` beside `data/`), "the 0 files" wording, and a long dotted name hidden across backslash-continued lines; fixed, with tests. Known and left: the 30 s deadline is checked between layers, so one parse or read can run past it; with A's `probes.repo_install` on, a partial read doesn't include the small `.csv`/`.tsv` files it wants (told A); and the static mapper's own `ast.parse` takes seconds on a pathological dotted name, flag on or off, as before (told A).

This push sends: `python/preflight/harness/closure.py` (new), `harness/static.py` (`mapped_files`, `SourceTooLarge`), `harness/mapper.py` (prompt order, flag on), `runner/sources.py` (`_git_read`, `_clone_read`, accurate refusals), `runner/pipeline.py` (`_reading_findings`), `runner/failures.py` (the `submission.too_large` hint), `flags.json` (`runner.read_closure`, off), `tests/test_d_closure.py` (new), `tests/test_static_lenses.py` (the oversize test's wording), `scripts/d_bigrepo_spark_map.py` (new), this file; and the merges of `main` `7ba177b` and `12058ca`.

## Mon Nov 2, 12:35 ET (item 3: GPU check passed; A reviews, B and C next)

CI on `f909ac3`: green, all seven jobs; the torch job ran the four new range-test tests (118 to 122 passed).

`d-lr-range-gpu` (GitHub 41490183749, A10, US$0.19 [derived]) [measured]: LoRA suggests 7.9e-5 against its own 1e-4, diverges at 7.6e-3; nanoGPT suggests 2.5e-4 against its own 6e-4, diverges at 1.0. 50 steps held at the suggested rate didn't diverge on either (LoRA 1.287 to 1.216, nanoGPT 8.74 to 6.01); at the script's own rate, 1.215 and 6.03 by step 50. LoRA's two runs can't be told apart at 0.8 times the author's rate; nanoGPT starts slower and ends level. So the suggestion lands just below the authors' rates and didn't diverge in 50 steps (one seed); it's not better than them on these two scripts. nanoGPT's divergence is read late (only at the ramp's end, though its curve was unstable from about 0.4). Write-up: `docs/decisions/lr-range.md`. Item 3's spend: US$0.19 of its US$2.

Check passed (the ramp applied on both, a suggestion that trains, divergence found). Its curve goes into the record, which the live guide (Muse Spark) reads, so C should check the data-use notice's probe-measurements wording covers it. Before the PM turns `probes.lr_range` on: A's review and A's redeploy. Asked of A (review), B (the guide) and C (the curve) in `from-d.md`.

Fact-check (read-only, against the run's raw outputs): every number matched; it caught "steps" that were points kept, "barely diverged" for nanoGPT, "safe", "trains" and "level with" as overclaims, and a wrong reason for needing no data-use line. All corrected.

This push sends: `docs/decisions/lr-range.md` (new), `docs/requests/from-d.md`, this file.

## Mon Nov 2, 12:20 ET (item 3: the LR range test, built behind `probes.lr_range`; GPU check next)

Built (A7 to A9 above): `probes/lr_range.py` (the ramp, 1e-7 to 1 over 100 steps; one divergence rule, `Divergence`, used by the recorder to stop and by the reading; the reading: steepest fall of the zero-lag-smoothed loss against log rate, lowest point, divergence; extras only, floats), the recorder's `lr_range` (rate set in the optimizer pre-step hook, every group keeping its ratio to the largest; stops at divergence), `probes/local.py` `with_lr_range` (curve and reading into extras, measurements dropped; the rate the optimizer held is checked against the ramp: `lr_range.checked`, else `lr_range.not_set` and no rate is reported), the pipeline's `p1-lr` point (plan jobs only, finetune and pretrain, after p1 ran ok, inside the job's budget, not for multi-GPU; in the record, not triaged, searched or fitted), and the GPU check app `runner/lr_range_app.py` (the range test, then 50 steps held at the suggested rate and 50 at the script's own as the baseline). Nothing changes with the flag off: `lr_range` is in the cache key only when set.

Review (read-only, git-free copy, three rounds): round one found a missing loss read as divergence, no stop on a NaN, the blow-up leaking into the smoothing, a possible multi-GPU hang, the confirmation running inside the scripts' warm-ups, and a predictions test that compared two empty sets; round two found rates reported when the ramp wasn't applied, a summary line reading a renamed key, a stop test that might not test the script stopping itself, and a hostile summary able to crash the reading. All fixed. The torch tests (`tests/test_d_lr_range_probe.py`) can't run in my workspace; CI's torch job runs them.

This push sends: `python/preflight/probes/lr_range.py` (new), `python/preflight/recorder/runtime.py` and `recorder/run.py` (`lr_range`), `python/preflight/probes/local.py` (`ProbeSpec.lr_range`, `with_lr_range`), `python/preflight/runner/pipeline.py` (`_lr_range_point`, the cache key), `python/preflight/runner/lr_range_app.py` (new), `tests/test_d_lr_range.py` and `tests/test_d_lr_range_probe.py` (new), `tests/test_pipeline.py` (one test), this file.

## Mon Nov 2, 12:05 ET (item 2: the in-flight shot's steps; needs B's GPU test)

`d-watch-cpu-check` (GitHub 41485598315, no spend) ran `preflight run` and `preflight watch` on `zoo/notebook_tiny/tiny_train.py` on a CPU [measured]: both work and the live record streams (50,000 steps in 28.6 s); but with `batch_size` raised from 64 to 1024 (2.4 times as long) no alert fired. The review of these docs found why: `preflight run` never fills predictions (only the hosted runner does), so no CLI record has `time.step`, `time.total` or `memory.peak` bands, on any device, and `watch` alerts only against those. I wrote my first draft as "a CPU record has no bands"; that was wrong, and is corrected. Asked of A in `from-d.md`; the shot's steps carry a stopgap one-liner (`with_predictions(..., enabled=True)`, checked on a copy of a committed record). The exact steps (8, from a clone to the alert) are in `docs/demo/video-plan.md`, "Shot 9b", marked "needs B's GPU test"; asked of B in `from-d.md`. The shot also needs `c.inflight` on in production, which waits on C's README line (C's `from-c.md`).

CI on `03d0e5f` (item 1's push): green, all seven jobs.

Review (read-only, git-free copy, against both runs' raw outputs): the CLI-predictions finding above; the dropped pages' "completed" status (the run kept only that a status was present); "one hit in five" (production sends three of those errors); the data-use line's "one per failed probe" (it's once per distinct error); the CPU check's torch build (CUDA build on a CPU) and 2.3 (2.4 over the whole run); step 8's "at once" (a tiny model's batch may not move reserved memory or step time; step 5 now makes a bigger change). All corrected.

This push sends: `docs/decisions/spark-search.md` (new), `docs/demo/video-plan.md` (shot 9b), `docs/requests/from-d.md` (A's review and the CLI predictions, C's data-use line, B's GPU test), this file.

## Mon Nov 2, 11:55 ET (item 1: validated; A reviews, C's data-use line next)

The validation ran (`d-spark-search-validate`, GitHub 41485593441, US$0.107 [derived]); the hand judgement is `docs/decisions/spark-search.md`. 8 failed probes sent (the 3 Preflight-own kinds production skips included, to test "none found"): 7 "none found", all right or acceptable by my judgement; 1 "found" (pyarrow needs NumPy 2), keeping Apache Arrow's install page, which documents the cause. Links in findings: 1, relevant; made up: 0; irrelevant: 0; weak citations: 0. The code dropped 2 sources the model had opened but the response didn't annotate; their titles match the error, so that cost recall, not accuracy. The planted made-up link was never repeated by the model, so that check was inconclusive. Item 1's spend: about US$0.16 of its US$1 (the probe and this run).

Check passed for the flag's own bar ("none found" rather than weak). Before the PM turns `harness.spark_search` on: A's review and C's data-use line, both asked in `docs/requests/from-d.md`. It rides A's redeploy, off.

CI on `03d0e5f`: green (above).

## Mon Nov 2, 11:45 ET (item 1: Muse Spark search grounding, built; validation next)

`2c532df` (pushed 11:35) sent: `flags.json` (`harness.spark_search`, `probes.lr_range`, both off), `scripts/spark_search_probe.py`, this file.

Built, behind `harness.spark_search` (off): when a probe fails with an error, Muse Spark searches the web through Meta's Responses API (`web_search`) for a known library or version issue matching it and answers with at most 3 sources or "no known issue found". Code keeps only sources the response annotates with a `url_citation` (and drops one whose page failed to open); any other named source is dropped and counted in the finding's evidence (`search.dropped_unannotated`). Whether Meta annotates only real search results is tested by a planted made-up link in the validation. Findings: `search.known_issue` (with `citations`) or `search.none_found`, `source: "model"`, one per distinct error in a job, every call in the record's provenance and cost. Not for a private or verify job, nor Preflight's own failures that name no library; one search per distinct error per job; URL queries, key-shaped strings and home folder names in the error are masked first. Whether a cited issue truly matches is the model's call under a strict prompt, so it's judged by hand on the zoo's failures next (`scripts/spark_search_validate.py`).

Before the flag can go on: the data-use line must say a failed probe's error text (the end of it, up to 2,500 characters, with URL queries, common secret shapes and home folder names masked), its code, the job's lens and its library versions go to Muse Spark on Meta's API, and the model's search queries go to a web search. I'll ask C for it once the validation passes.

This push sends:
- `python/preflight/harness/search_grounding.py` (new) and `tests/test_d_search_grounding.py` (new)
- `python/preflight/harness/spark.py`: `ModelClient.respond()` for the Responses API; `_post` takes a path
- `python/preflight/runner/pipeline.py`: `_search` after `_triage`, behind the flag; `tests/test_pipeline.py`: one test (off, on, private)
- `scripts/spark_search_validate.py` (new) and `tests/fixtures/spark_search_cases.json` (new)
- `scripts/d_watch_cpu_check.py` (new; item 2's CPU check)
- this file

Review (read-only, git-free copy): one high finding (an opened page counted as retrieved even if the open failed) and medium ones (annotation trust untested, non-library errors sent, no scrubbing, verify jobs searched, cancellation ignored); all fixed as above. Its re-check found `scrub()` masking names the search needs (`hf_hub_download`, `pad_token = ...`) and missing environment-variable secrets (`DB_PASSWORD=`, `GITHUB_TOKEN=`); both fixed, with tests for each. It also reviewed `scripts/d_watch_cpu_check.py` (write the output after every case, take the record's path from the CLI, keep the run's secrets out of the subprocesses' environment); fixed. Local checks: ruff clean; `pytest tests` 1,460 passed, with the 13 torch-needing sandbox and overhead tests failing here as before my change (no torch in my workspace; CI has it).

## Mon Nov 2, 11:10 ET (Chiya's answers; the video plan's statuses)

`f48b589` CI green. Merged `main` `f7f10b0`. Chiya (10:57): the 14 overnight decisions stand. The
unseen runs are on today, so I touch no engine or harness code.

- **Closed:** PF102 now opens the Nov 2 live record. **Explained:** the two empty L4 eval runs were
  Modal failing to schedule an L4; A's capacity fix falls back to an A10 with the wait shown (A to
  confirm in `from-a.md`).
- **Video plan** (`docs/demo/video-plan.md`), statuses now: shot 5 opens from `/s/` once C confirms
  `workers/` (today, a preview first); shot 6 waits on A's triage re-run (Tue noon); shot 12 needs
  C's deploy today (the "Private mode" tick; the flag is on in `main` and A's runner) and C's private
  check-in record id; shot 13 waits on B's table (likely tonight).
- **X post:** switches to the `/s/` link when C confirms `workers/`.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the merge of `main`, the video plan's statuses, two requests and this entry.

Cloud runs: none.

## Mon Nov 2, 09:00 ET (check-in: the $8 cap is deployed; three requests closed)

`8bcd99a` CI green. Merged `main` `d625a39` (the validation freeze declared at `56c2141`; I touch
nothing of the unseen list).

- **The cap from Nov 5 is deployed:** `queue.py` returns $8 from Nov 5 (`c061626`), and A's runner
  was deployed from `1ea2dd6`, which contains it (Nov 2 08:24 UTC); until Nov 5 it applies $10. (An
  environment variable, `PREFLIGHT_DAILY_GPU_CAP_USD`, can override it; nothing I can see sets it.) Per Chiya, the README draft and the
  builder kit now give the number.
- **Closed with A:** private jobs never go to Baseten (A's fix, deployed); `d.notebook` passed (with
  A's own training notebook) and the $8 cap is in.
- **Still open:** A's production LoRA triage re-run (deadline Tue Nov 3, noon ET); the L4 eval-sweep
  question and the items 42/19 note to A; PF102's numbers with B and the PM; S1 and S2 with C.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the merge of `main`, the cap sentences, the closed requests and this entry.

Cloud runs: none.

## Mon Nov 2, 05:40 ET (overnight: the PM's third merge; audit of C's deploy; the status word)

`adef21e` CI green. The PM's 5:15 merge made `main` `16e0bb3` and then `ae0e900`
(`harness.glimmer_private` on in production, with C's data-use line); `ws-d` is merged up to it.

- **Free-tier chip** (the PM's 4:00 ask): "Not measured here", C's word; also in the share cards
  and the drafts (decisions 12 and 13). 252 of 252 add-on tests, 18 of 18 worker tests, the
  typecheck and the hero-card check pass [measured].
- **Stranger audit** of C's deployed `5058361` (`d.notebook` on): 104 of 108 clean, S1's four the
  only failures, the Notebook tab clean (decision 14).

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the merge of `main` `ae0e900`, the status word in the chip, cards and drafts, the
re-rendered hero card, the audit log and this entry.

Cloud runs: none.

## Mon Nov 2, 04:20 ET (overnight: the PM's two merges; audit of C's deploy)

`6c05323` CI green. The PM's 2:40 merge took `ws-d` `6c05323` into `main` `1ea2dd6`; the 03:55
merge made `main` `6f49c9c` (`d.notebook` on; Glimmer's private mode held for its data-use line).
`ws-d` now equals `main` `6f49c9c`.

- **Stranger audit** after C's deploy of `1ea2dd6`: 100 of 104 clean, S1 and S2 unchanged, nothing
  new; and `6f49c9c` with the Notebook tab: 104 of 108 clean, the notebook states all clean
  (decision 10; `docs/audit/nov2-stranger/README.md`).
- **README draft** follows the PM's edits (decision 11).

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the audit script's notebook states, the two audit logs, the README draft and this entry.

Cloud runs: none.

## Mon Nov 2, 02:05 ET (overnight: video plan, submission refresh, X post)

`49e36af` CI green on its second attempt (decision 6).

- **Video plan** (item 3): `docs/demo/video-plan.md`, 15 shots, about 2:55, each with its
  voice-over line, on-screen label and status. Works now: the script, home, check-in, Boarding's
  Muse Spark and Spark Open tags, the knobs and possible outcomes, the live guide (Muse Spark), the
  exports, the measured agent and GRPO samples, the link and the end card. Needs by Tuesday: the
  Oct 28 record's share link opening on the live site (A); A's production triage re-run (shot 6);
  `harness.glimmer_private` on with one private check-in triaged by Glimmer (shot 12; otherwise a
  captioned card of the three Pareto arms keeps Glimmer labelled); B's scored checks page (shot
  13). Each "needs" row says what replaces
  it if it isn't ready by Tue Nov 3, noon ET.
- **Submission drafts** (item 4): `portal.md` now says what the live site offers (check-in, the
  guide, the exports) and drops three stale TODOs; adds the eval sweep's catch (8 of 16 cut off,
  A10, D's run); the five-GPU cross-GPU result (57% of 6,694, B) in place of the four-GPU 60%; the
  corpus labelling (Spark Open: 1,117 of 1,522 agree, worse than "always stable" on divergence runs,
  $8.79); and the three Pareto arms on the same 26 tasks (Muse Spark 23, Glimmer 15, Spark Open 14
  ended on a tool call; $0.39, about $0.22 of GPU, $0.33). Each is a row in the numbers table with
  its source.
- **X post** (item 5): `docs/submission/x-post.md`, two variants (262 and 269 characters), linking
  the cockpit until `workers/` deploys (decision 9).

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the video plan, the portal and numbers table, the X post, and this entry.

Cloud runs: none.

## Mon Nov 2, 01:15 ET (overnight: stranger audit of `cfcf83c`)

`a86b0ec` CI green (all seven jobs, `workers` included now).

- **Stranger audit** (overnight item 2) after C's deploy of `main` `cfcf83c` (deploy run
  41385425843): 100 of 104 states clean [measured, Chromium] on that commit served by Vite in
  production mode (decision 4 above); every detail state opened a sheet, and every check-in state
  reached "Your flight record is ready.". Every sample, opened and with a node's sheet open; check-in through
  Boarding to "Your flight record is ready"; desktop and phone, dusk and day. Two bugs, both C's,
  filed: S1, a contrast failure on "Tight: measure first" in the day theme (3.99:1); S2, overlapping
  and cut-off map nodes with a detail sheet open. None in D's code. `docs/audit/nov2-stranger/`,
  script `docs/audit/scripts/stranger.mjs`.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the audit script, its README and screenshots, the request to C and this entry.

Cloud runs: none.

## Mon Nov 2, 00:15 ET (the eval-sweep record measured on an A10; a notebook that trains for A)

`56a2d32` CI green. Merged `main` `cfcf83c`. `197e650` CI **red**: ruff formats notebooks too, and
the new `train_mlp.ipynb` had spaces; fixed (formatted, its recorded conversion regenerated, both
notebook suites and the CPU probe re-run), in this push.

- **Eval-sweep record, item 19: done on the third try**, on an A10 (Chiya allowed an A10 or the L40S
  for one more try under $1). `d-eval-sweep-a10` (GitHub 41391396558), record
  `01M6DA13XWGBRY2X5P2XHMP9TS`: one probe, ok, 177.9 s on the A10, after 83.8 s waiting for the GPU
  (the recorder's `gpu_wait_s`) [measured]. It caught the planted no-thinking-cap bug: "8 of 16
  generated sequences ran to max_new_tokens=2048 without ending (95% interval 28%–72%)"
  (`generation.truncated`) [measured], beside the static `eval.activation_dump`. Every estimate
  abstains (the record's reasons: no probe timed a step; fewer than two probe sizes). Cost by the record: US$0.1203 (US$0.0545 GPU [derived:
  GPU seconds at list price], US$0.0470 sandbox [guess rate], US$0.0189 model calls [measured]).
  Committed as `zoo/eval_sweep_reasoning/record_nov2_a10.json` with a placeholder share token, as A
  does. The two L4 runs stay unexplained (one cancelled while probing, one `recorder` after
  1,280 s); A is adding a "waiting for a GPU" event. Item 19's spend in all: at most about US$1.36
  [guess: the first L4 run's GPU time is unknown].
- **Modal logs:** I can't read them; my only way in is `remote-run.yml`, and a log-reading script
  would be a new kind of run. Told Chiya where to look in the dashboard.
- **`d.notebook`:** A's check-in showed notebooks are converted and run; it used a fixture with no
  optimizer, so `too_few_steps` was right. New fixture `train_mlp.ipynb` (a small MLP, an AdamW
  step every batch, no downloads); A's converter's output is recorded beside it, both notebook test
  suites pass, and the converted script probes cleanly on a CPU (three probes ok) [measured]. A has
  the command to run it through the deployed runner. The README draft's notebook rows say so.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the notebook formatting fix, the eval-sweep record and this entry.

Cloud runs: `d-eval-sweep-a10` (above), A10, US$0.12 by the record.

## Sun Nov 1, 22:50 ET (eval-sweep record: two runs, no measurement; stopped and asked)

`0eaa25d` CI green.

Item 19, the eval-sweep record, on an L4 through `remote-run.yml` (`ops_app.py::smoke` on
`zoo/eval_sweep_reasoning`, visitor `d-eval`, ref `main` `c084bee`):

| Run | Result | Spend |
| --- | --- | --- |
| `d-eval-sweep-l4` (GitHub 41380341849), 20-minute cap | Mapped (Muse Spark refined the map), then "probing" for 19.5 min with no event; cancelled at the cap. Record `01M6D30EYGANSVKSA2SG4EBSAE`: no probe results, only the static `eval.activation_dump` finding | The record's provenance: US$0.0179 of Muse Spark [measured] (its cost block says US$0); Modal's bill unknown, at most about US$0.58 if a GPU and sandbox ran the whole time [guess] |
| `d-eval-sweep-l4-2` (GitHub 41382389659), 30-minute cap | Done, but its one probe ended "The probe sandbox ended without a result. Step: starting Preflight's recorder" (`recorder`) after 1,279.8 s, with 0 steps. Spark Open's triage: `infrastructure`, quoting "The probe sandbox ended without a result". Record `01M6D493CJCMMMV0C0P75FT5QH` (share `hXW24O9aS_aAHOZG5i4lrg`): memory estimates and `risk.oom` extrapolated from the static map; time and cost abstain | Record: US$0.6423: US$0.2844 GPU [derived: 1,279.8 s at the L4's US$0.80 an hour, queue included], US$0.3377 sandbox [guess rate], US$0.0202 model calls [measured] |

The second run ended as A's first eval-sweep smoke did on Nov 1 at 04:30 UTC ("The probe sandbox
ended without a result" after about 1,280 s), and its step says the recorder was still starting, so
A's `fda1762` stop never came into play. My reading [guess]: the runner or the model download
inside the probe, not the eval hooks; asked A. Per my plan I've stopped rather than retry: asked A (`from-d.md`)
and Chiya. Today's eval spend: at most about US$1.24 [guess: the first run's GPU time is unknown].

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): this entry and the request to A.

Cloud runs: the two above.

## Sun Nov 1, 22:00 ET (items 42 and 19 taken over from A; the 8 PM README mirrored)

`0afe0c5` CI green. Merged `main` `c084bee` (round seven).

- **The 8 PM README, mirrored.** The draft's "On in production" table is now the PM's, row for row
  (`c.hops` and `c.exports`; the three add-ons), and "Behind a flag" takes the PM's rows for
  `d.exports_cli` and `d.notebook`. What still differs from `README.md` is newer than it: the
  Glimmer private-mode row, the guide and triage on in production in the Meta-models table, and
  the Add-ons paragraph.
- **Item 42, pip packaging and the quickstart (Chiya, 21:48).** Packaging already worked:
  `pyproject.toml` builds the `preflight` command with hatchling. Checked in a clean Python 3.11
  venv with pip 24: `pip install git+file://` of this repository (the same install path as the
  GitHub URL, which needs the repository public) took 20 s the first time and 7 s again [measured];
  `preflight --help` and `preflight map` work; `preflight demo` without torch says what to install;
  with torch, `preflight demo` in an empty folder ran three probes on the CPU, all ok, in 23 to
  24 s, and wrote a record with the probes and two findings [measured]. The extras form
  (`"preflight[probe] @ git+file://localhost/<this repository>"`) resolves too: pip found torch
  already satisfied [measured]. **Fixed:** two hints, the demo's and the
  `submission` failure's next step (shown to visitors), said `pip install "preflight[probe]"`, which
  installs an unrelated PyPI package of that name; both give the git URL now. The demo's last line
  said to open the record in the web app, which opens records by share link only; it now points at
  the hosted check-in. The README draft's quickstart leads with `preflight demo`. A run on a CPU
  carries no predictions (the prediction step returned none here even with `engine.memory` on).
- **Item 19, the eval-sweep record.** A's `fda1762` (an eval probe stops after two generate calls)
  is in the deployed runner, so I dispatched `ops_app.py::smoke` on `zoo/eval_sweep_reasoning` on an
  L4 through `remote-run.yml` (`d-eval-sweep-l4`, GitHub run 41380341849, visitor `d-eval`, a
  20-minute cap, so at most about $0.60 [guess: 20 min x ($0.80 GPU + a $0.95 sandbox guess) an
  hour]). Result in the next entry. Asked A whether the recorder's eval hooks need anything: it
  records generate's lengths and truncation, but not the activation dump's bytes per question.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the merge of `main` `c084bee`, the packaging commit (CLI hints, `failures.py`, README
draft), the request to A and this entry.

Cloud runs: `d-eval-sweep-l4` (above), L4; result and cost in the next entry (22:50).

## Sun Nov 1, 21:45 ET (check-in: Glimmer measured, so its wording moves; stale model rows fixed)

`0f4ac80` CI green. `main` is still `50db9b6` on GitHub (`d5062bc` hasn't landed), so the README
diff waits.

- **Glimmer:** the endpoint is live (B's `b-glimmer-deploy`; A's runs call it) and A measured its
  cost per call, so per Chiya it moves from "price table only" to private mode, hosted on our own
  GPU, behind `harness.glimmer_private` (off): US$0.24 for one call through a 411 s cold start
  [measured, `a-triage-glimmer-5`] and US$0.33 for 11 calls in one run, likely one cold start among
  them [measured, `a-triage-agreement`] (no warm per-call figure: the run total mixes the two); the
  same category as Spark Open on 9 of 11 failed zoo probes [measured]. A's run times are UTC.
  Updated in `readme-final.md` (moved from "Not used" to "Behind a flag", and its Meta-models row),
  `meta-and-partners.md` and `portal.md` (the open-weights section and "What's next"), with the
  numbers in the submission's table. The costs come from A's status on `ws-a`, not B's.
- **Stale rows fixed while there:** the guide and Spark Open triage are on in production (the
  drafts still said off); the open-weights section now gives the first real run
  (`a-triage-lora-prod`, 4 calls, $0.0184) and that Spark Open quoted a line on all 11 failed zoo
  probes since triage reads the recorded error. A TODO(PM) waits for the production re-run's ID.
- **Asked A:** with `harness.glimmer_private` off, `_triage_client` sends a private job to Spark Open
  on Baseten, while its docstring and the contract say a private job never goes there. No screen
  sets `private` today, but the API schema takes it.
- **Not landed:** A's production triage re-run with a new share token and a committed record copy;
  A's notebook check-in; A's $8 cap is on `ws-a` but not on `main` or in a logged deploy, so the
  README keeps the cap after Nov 4 unnumbered.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): those drafts, the request to A and this entry.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 20:15 (the draft mirrors the 8 PM README)

`b3c5c4d` CI green. Chiya (20:07): `main` `d5062bc` has `readme-final.md` as `README.md`, with
`c.hops`, `c.exports`, `d.expertise`, `d.currency` and `d.freetier` on in production and
`d.notebook` off until A's real notebook check-in passes. `d5062bc` isn't on GitHub yet (main is
still `50db9b6` at 20:12), so I mirrored the change from her description: three rows move to "On in
production" (possible outcomes, the exports with the Weights & Biases helper, and the three add-ons),
"Behind a flag" keeps the notebook tab and the exports' CLI item, and the Add-ons section says
which are on. Once `d5062bc` lands I'll diff the draft against `README.md` and match its wording.
The cap after Nov 4 gets its number once A's $8 is deployed.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the draft's tables and this entry.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 18:45 (D's four flags checked for strangers before the 8 PM change)

`b8963a9` CI green. Chiya (18:20): A's redeploy is confirmed with the limits live, so the README's
limits TODO(PM) is gone. The GPU cap after Nov 4 is $8 a day, not $30 (Chiya), but
`runner/queue.py` still says $30, so the README gives only the $10 through Nov 4 and "a daily GPU
spend cap" otherwise (Chiya allowed either wording); A is asked to change the code.

**Checked** on `main` `50db9b6` [measured, Chromium], before the free-tier fix: `addons.mjs` passes
20 of 20 add-on states (desk and phone, dusk and day, flag on and off). On a production-mode build
with the four flags added, the home page, New flight, the logbook and seven sample cockpits load at
both sizes with no page or console errors (only the missing API behind the dev server). After the
fix: 252 of 252 add-on tests and the typecheck pass, and the production-mode build shows the
free-tier line with its status word [measured].

| Flag | Ready for strangers? | Basis |
| --- | --- | --- |
| `d.expertise` | **Yes** | The dial and glossary cards on every sample cockpit; the dial's order stays frozen (zoo check unchanged on the engine goldens) |
| `d.currency` | **Yes** | Every price converts through the ECB's reference rates of Oct 26, 2026, shipped with the app (`rates.json`, no network call), with the source and date shown |
| `d.freetier` | **Yes, with tonight's fix** | Its "Needs N GiB" line gave a number with no status word; it now says where it comes from, as the readouts do: "(illustrative)" on a sample, the estimate's own word on a real record (new test). Its session limits are cited per tier, with the dates read |
| `d.notebook` | **Code ready; one run short** | The API accepts kind `notebook`, the runner converts it, and A's deployed `d73b4e1` has that path; D's parity test against A's converter passes. No notebook check-in has run end to end on the deployed runner in A's status: asked A to confirm, ideally with one real check-in before 8 PM. Until A confirms, I'd leave it off |

**Also seen, not D's flag:** the PF102 sample (Qwen2.5-1.5B LoRA, engine-filled golden) reads
"Runs out of memory on L4: peaks at 117 GiB", labelled Illustrative; the real run peaks at 20.4
GiB. Asked B and the PM (`from-d.md`).

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the free-tier status word and its test, the README's limits sentence, the kit's limits
lines, the requests to A and B, and this entry.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 18:20 (round six merged)

`b43c386` CI green. Merged `main` `50db9b6` (round six; a fast-forward, since `main` already held
`ws-d`). On it: add-on tests 251 of 251, typecheck clean, worker tests 18 of 18, hero card up to
date, ruff clean on the GNN script [measured].

- **Limits sentence:** the TODO(PM) stays (Chiya, 18:11) until A confirms. A's status already
  reports the redeploy with the limits (`a-deploy-main-19`, 20:20 UTC, from `ws-a` `d73b4e1`,
  which contains `9a2bb82` and `5a78a60`); the TODO now says so, so the PM can check it at the
  8 PM swap. The builder kit's two limits lines say the same, as A's report, and keep the "watch
  for more than 3 check-ins" fallback until A confirms.
- **Triage shot:** that redeploy also has A's fix `b300684` (triage reads the probe's recorded
  error), so A's re-run of the production LoRA triage can now quote the OOM line. No re-run is
  posted yet.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the TODO wording, the kit's two limits lines and this entry.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 16:05 (the corpus on Hugging Face, linked)

`ca58c59` CI green.

- **The corpus** (Chiya, 15:57): https://huggingface.co/datasets/chaoschia/preflight-corpus, commit
  `0aac218`, files `README.md`, `SHA256SUMS`, `preflight-corpus-0.3.0.jsonl.gz`; private until it
  goes public with the repository on Nov 3 to 4. Linked under "The corpus published with a
  datasheet": a new section in `readme-final.md` (before Validation), a paragraph and a link in
  `portal.md`, and a row in the submission's numbers table. The counts (2,339 rows: 1,519 kernel
  configs, 100 HF jobs, 720 divergence runs) and the licence (Apache 2.0) come from B's datasheet;
  I can't open the private dataset from here, so the drafts don't restate its checksum.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): those three files and this entry.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 15:50 (check-in: A's first triage run; Glimmer deployed, no cost per call yet)

`a15a173` CI green (all six jobs). `main` is still `6757667`; nothing to merge.

- **A's production triage run** (`a-triage-lora-prod`, share `jgk05gwUgV-Bo78sErzKFQ`, A's status on
  `ws-a`): L4, the planted eval OOM caught (530 MiB asked, 486 MiB free of 22.06 GiB [measured,
  A]). Not filmable yet: the eval probe's triage said "unclear" and quoted no line (A's fix
  `b300684` is live at A's next deploy), and the shot needs a quoted line. Its finding sits on node
  `eval`; checked in Chromium (desk and phone) with such a finding added to a copy of the Oct 28
  record, it shows in the Eval loop node's detail sheet, below the rule checks' findings, the
  quoted line inline. The storyboards now film it there.
  Asked A (`from-d.md`) to re-run after the deploy and commit a copy of the record. Noted in both
  storyboards' dependency rows.
- **The one-record version:** A's record meets Chiya's condition (L4, the same eval OOM), so I'll
  offer it, but every spoken number has to come from the record, and the record isn't in the repo
  or reachable from my workspace. I'll draft it once A commits the copy (better from the re-run).
- **Glimmer:** B deployed the endpoint (`b-glimmer-deploy`, scaled to zero, proxy auth), with smoke
  latencies, but no measured cost per call from it yet, so the README and submission wording waits.
- Also: re-wrapped `vendor.mjs`'s header comment, which my last edit had run together, and added
  that CI and the deploy don't need to run it (as told to C).

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the storyboard notes, the request to A, the comment wrap and this entry.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 15:25 (card fonts committed)

`749edd4` CI green (all six jobs). The kit is with Chiya; she sends me the sheet and form after the
session to rank.

- **My bug, caught by C:** the card's four fonts were never in the repo. The root `.gitignore`'s
  `*.bin` kept them out, so a clean checkout couldn't run the card tests, and D2's deploy copy
  would have shipped a bundle importing missing fonts. Chiya: commit them only if the licence
  allows redistribution. It does: the OFL 1.1 allows bundling with software when each copy carries
  the copyright notice and licence (both in the `OFL-*.txt` files beside them) and the fonts
  aren't sold alone; none declares a Reserved Font Name. The files are byte-identical to the
  packages' `.ttf` [checked with `cmp`]. Force-added; basis in `docs/licences.md`.
- `vendor.mjs` takes the MPL-2.0 text from `workers/node_modules/lightningcss` (the same file as
  web's), so it no longer needs web's packages. Re-running it on a clean copy changes nothing.
- **Checked on a clean copy of the repo** (`git checkout-index`, `npm ci`, no vendor step): 18 of 18
  tests, typecheck clean, hero card up to date [measured]. Told C to drop `npm run vendor` from the
  CI job and the deploy step.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the four fonts, `vendor.mjs`, the licence note, the workers README line, the reply to C.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 14:10 (README: check-in and the guide on in production; builder kit ready)

`f7e4306` CI green (all six jobs). `main` is `6757667`, which already holds `ws-d`; nothing to merge.

- **Found: the demo limits aren't in the deployed runner.** They're on `main` (`9a2bb82`,
  `5a78a60`), but A's last logged runner deploy (`a-deploy-main-18`, from `8154268`, 13:04 ET)
  predates them, so with `runner.queue` on, a visitor isn't limited yet. The README row has a
  TODO(PM) to keep the limits sentence only once A redeploys; the kit tells the observer to check.
- **README draft** (`readme-final.md`, for the PM's 8 PM swap): `runner.queue` (check-in) and
  `c.guide.live` (the guide) move to "On in production" (Chiya, Nov 1; both true in
  `flags.production.json`). The check-in row carries the demo limits: 3 probe jobs per visitor per
  day and the site's daily GPU spend cap, $10 a day through Nov 4 and $30 from Nov 5 (cited:
  `runner/queue.py`); past either, samples only until midnight ET. The guide row keeps its smoke
  run and gains its spend limits from `api/preflight_api/ledger.py` ($0.40 a tour, derived; $10 a
  day, the code's value pending Chiya's answer), and says the rule-based tour starts if the live
  guide fails before answering (`GuidePanel.tsx`). The Notebook tab is off in production, so the
  rows say repo or script. "Try it" and the
  diagram no longer say check-in is behind a flag.
- **Builder kit** (`docs/builders/`), ready for Chiya to hand over tonight: the live link in the
  brief; the brief's data-use summary now follows the desk's notice as the local production build shows
  it (Spark Open reads the last 3,000 characters of each probe's log, the live guide, uploads kept); a "limits on the public site" note so a limit refusal isn't
  read as the builder's fault; the Notebook tab is off in production, so notebooks go in as a `.py`
  upload. Checked on a local build with `--mode production`: home, New flight, two tabs (Repo link,
  Upload), the three goals, "Got it", the samples. The live site can't be reached from my
  workspace, so Chiya opens it once before the session. The sheet gains "opened the guide?". The
  results come back to me to rank as onboarding fixes.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; CI result in the next
entry): the README draft's status rows and the builder kit.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 12:55 (round five merged; finding 12 fixed; triage finding visible; worker tests fixed)

**CI rule (Chiya, 12:43):** after every push to `ws-d`, check its CI run; a red run is fixed
before the next push, and a push isn't done until CI passes (a cancelled run means a newer push
replaced it). Today's pushes: `b781a91`, `ead1b6c` and `dcda920` green; `f4cf04d` cancelled,
replaced by `ead1b6c` (green).

**Missing from the log: push of `dcda920`, 11:32** (gitleaks clean; reviewed read-only): the
triage shot on A's production LoRA run and C's list, the Glimmer note. CI green.

**Chiya, 11:42:** "same script" is right; the fine-tune shots stay on the Oct 28 record and its
numbers. Only if A's production LoRA record runs on an L4 and catches the same eval OOM do I offer
a version with both shots on that record, every spoken number re-taken from it; otherwise no
change.

Merged `main` `85fbf9a` (round five: contract 0.6.2, engine-filled goldens). Checked on it:
- **Finding 12: fixed.** `model-tags.mjs` passes 12 of 12 in Chromium, desk and phone, the
  same-page case included [measured]. Moved to Done.
- **C's Findings band shows a triage finding.** In Chromium at 1440 x 900 and 390 x 844, with a
  triage finding (info, no node) added to a copy of A's Oct 28 record: the cockpit shows a
  "1 finding off the map" chip, and clicking it shows the title, the sentence, the quoted log line
  and "Found by Muse Spark Open triage", with the page moved to the Findings band (screenshot)
  [measured]. The storyboards now say to click the chip.
  The shot waits only on A's production link (none yet).
- **Two of my worker tests went red on the engine goldens, with CI green:** they pinned the
  fixture golden's "8.3 to 10.1 GiB, illustrative". Now they mark a copy's models as fixtures and
  check relations, so they hold on either golden set (18 of 18 on each [measured]), plus a new
  test that the real golden is called illustrative only if its models are fixtures. CI never ran
  `workers/`, so C is asked to add it.
- **Add-on tests 251 of 251, typecheck clean, hero card up to date** [measured].
- **The dial on main's engine goldens** [simulated: the golden probes are hand-made]: the same
  counts as before (3 of 3 in the dock on each record; with C's worth-checking list, 2 of 3 and
  1 of 3; all nine orders tie at 3 of 6), now with B's `run-cost@0.1.0` among the models. The
  order stays frozen for Nov 2.
- **B's report** is on `main`; the PM already linked it in `readme-final.md` (`3e47fe9`), and I
  removed the TODO from the submission table's row.
- **Not landed:** B's Glimmer endpoint with a measured cost per call (the Glimmer wording stays
  as it is), and A's production triage share link.

**This push to `origin/ws-d` sends** (gitleaks clean; reviewed read-only; its CI result goes in
the next entry): the merge of `main` `85fbf9a`, the worker test fix, and these doc updates.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 11:20 (standing yes; the Spark Open triage shot in both cuts; push log)

**Standing yes (Chiya, 11:14):** pushes to `ws-d` go without asking once gitleaks is clean and one
read-only review has checked the change; runs under $1 on a T4, L4 or A10, up to $5 a day, without
asking. Still asked first: anything past those, A100 or H100, a new kind of run, deploys or
anything that keeps billing, anything to `main`, production flags, contract changes.

- **Spark Open triage shot** (Chiya: one shot in both video cuts, labelled on screen, on a real
  production record; A posts the link today; no link by Tue Nov 3, noon ET, and it's dropped, not
  faked):
  `storyboard.md` 12b (3-minute plan A) and E (45-second cut), `fallback-cut.md` 11b (plan B).
  Each holds on the triage finding, where the app writes "Found by Muse Spark Open triage"
  (`finding.source.spark_open_triage` in `en.json`), with a burned-in caption naming the model.
  No number is spoken in it, and the line says only what the code does: when a probe fails, Spark
  Open reads the end of its log and quotes the line.
- **Blocker, found in review: no screen shows a triage finding.** Triage's finding is info with
  no node; on `main` the detail sheet shows only findings with a node, the map skips info and
  node-less ones, and the tour takes only findings with a fix. So the shot can't be filmed today,
  even with A's link. Asked A (tie the finding to its probe's node: the smaller change) or C (list
  node-less findings), for Chiya's next batch. The storyboards say so in the shot's row.
  **11:30, Chiya:** C adds the cockpit list; A's production triage run is the LoRA zoo script, so
  the failed probe is the eval probe's out-of-memory and the shot sits on the same script as the
  fine-tune story. The room-making trade-offs stand. Storyboards and `from-d.md` updated.
- **Glimmer (Chiya, 11:30):** back in, hosted by B on our own Modal GPU (Apache 2.0). The README
  draft and the submission (`readme-final.md` lines on Muse Glimmer; `meta-and-partners.md`;
  `portal.md`) keep today's wording until B's endpoint is live, then say "private mode: hosted
  on Preflight's own GPU, behind its flag" with its measured cost per call. Watching for it.
- **Decision: what makes room.** The 3-minute video has a hard 3-minute limit (Judging playbook)
  and both 3-minute cuts already end near 2:55 to 2:58. Plan A drops shot 7 (Boarding's "Checking
  the logs" tag), which says the same thing with a label that shows whether or not the model ran
  (finding 12); plan B drops shot 10 (the tank and cost readouts, the shot with no measured
  number). The 45-second cut takes 4 s from shot C and 3 s from the end card. Each comes back as
  before if the triage shot is dropped. Chiya can swap which shot gives way.
- **Asked A** (`from-d.md`): triage makes a finding only for a probe that didn't finish, so the
  record needs a failed probe whose finding quotes a log line; post the link.
- **Open:** the submission's Spark Open lines (`meta-and-partners.md`: "not yet run on a real
  job"; `portal.md`) change once A's record exists; I'll update them when the link lands.

**Push to `origin/ws-d`, 11:25** (gitleaks clean; reviewed read-only: the first three at 11:00,
the rest below):
- `9c55415` GNN zoo script: `N_TRAIN_NODES` as a plain top-level setting.
- `ea4bbf2` README draft and submission: the divergence refit numbers (0.66, lead 138).
- `8b94541` Review fixes for those two.
- This commit: the triage shot in both cuts, the request to A, and this entry.

**11:30: B's report landed** (`1cfa689` on `ws-b`, `python/preflight/corpus/reports/divergence_nov1.json`).
It agrees with Chiya's numbers: AUROC 0.657 (0.563 to 0.765), lead 138.5 steps (3 to 269), 720
runs, held out by settings family [measured: B's report]. The README draft, the submission table and the portal now
link it, tagged measured as B's status does, and say "held out" again. The validation block's
divergence row is the scorer's to fill (B: it reads the saved report). Merged `main` `f017010`
(contract 0.6.1): add-on tests 251 of 251, worker tests 17 of 17 [measured, Nov 1].

**Push to `origin/ws-d`, 11:22** (gitleaks clean; reviewed read-only): the merge of `main`
`f017010`, and the README and submission link to B's report with this entry.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 11:00 (pushed; GNN run length; waiting on B's report)

Pushed `adc9ca3`..`b781a91` with Chiya's yes (10:57).
- **GNN run length:** `zoo/gnn_pyg/train.py` gains a plain top-level `N_TRAIN_NODES = 160_000`, used by the sampler in place of `int(0.8 * n)`. Basis: A's recorder can't count nodes when the script samples by hand, and B's engine needs a run length. It's 157 steps an epoch at batch 1,024, the same as before (0.8 x 200,494 = 160,395 also gives 157), so the run's length in steps is unchanged [derived]; each epoch trains on 395 fewer seed nodes, all in the last, partial batch, so peak memory is unchanged and wall time falls slightly. An assert guards a smaller graph. The planted fan-out `[15, 25, 25]` is unchanged. A is asked to read it into `data.n_examples`.
- **"C's one-line fix to build.test.ts":** not on GitHub yet (no branch changes that file), and not needed. My `adc9ca3` already makes that test hold on both golden sets. A's own flagged `make_golden.py` (`ws-a`) produces goldens byte-identical to the ones I tested.
- **B's divergence refit** (`1cfa689`): not in any branch on GitHub yet (fetched 11:00). The README draft, the submission table and the portal text now carry Chiya's numbers from it (AUROC 0.66, 0.56 to 0.76; lead 138 steps, 3 to 269; still a miss, not used), tagged `cited: B's fit report 1cfa689, via Chiya, Nov 1`, with a TODO(PM) for the link path. Basis: Chiya's message gives the numbers and the commit; I couldn't read the report itself, so the check-in compares the numbers against it and adds the path when it lands.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, 08:00 (pushed; tests that pass on both golden sets; hero cards; the dial)

Pushed `09648e7`..`cbbfe73` with Chiya's yes (07:42). Chiya's calls: card rendering per request stays off; the hero records get pre-rendered cards; B will commit its divergence fit report, which I'll link when it lands.

- **Golden-record tests.** A's `make_golden.py` with `PREFLIGHT_GOLDEN_ENGINE=1` isn't on any branch yet. So I applied B's exact call (`from-b.md`: `fill_predictions` after `model_validate`) to a scratch copy, which reproduces the committed goldens byte for byte without the flag. I regenerated both sets into the scratch folder and swapped each in to test.
	- The whole web suite on the engine goldens: 12 failures. 2 are in my files (`addons/exports`); 10 are in C's (`guide/`, `workspace/provider`, `workspace/rate`, `workspace/record/samples`).
	- My 2 now assert relations, not fixture figures. The plan names the abstained and missing readouts, and every number it gives is the record's own. The "sample" test builds its own all-fixture copy of the golden record.
	- D's add-on suite passes on both sets, 251 of 251 each. The goldens themselves aren't committed; the PM regenerates them at the next merge.
- **The dial on the engine goldens** (`make_golden.py`'s path, B's real models) [simulated: the golden probes are hand-made]: identical to the fixture set and to my Oct 31 scratch run. Every planted knob is in the dock on screen today (3 of 3 on each record). With C's list, fine-tune has 2 of 3 in the dock and pretrain 1 of 3. All nine orders tie at 3 of 6. Decision: the order stays frozen for the Nov 2 held-out run, which remains the real test.
- **Hero cards.** The LoRA OOM record's card (`ICDYrfHQ2nBNbv3mlRHHMg`) is pre-rendered with the same card code from A's copy of that record (`api/tests/fixtures/a-smoke-l4-main-4.record.json`). It's committed as `workers/static/cards/<token>.png`, and the share page points that token's preview at it, so its links preview with an image on the Free plan; every other record previews as text.
	- `npm run hero-cards:check` re-renders and fails on a stale card; I checked that it fails on a corrupted copy.
	- The X post's link is that record's `/s/` page.
	- C is asked to copy `static/cards` into the deployed site and run the check in CI.
- **Read-only review of these two commits:** no blocking problems; every check passed on both golden sets. Hardened from it:
	- With card rendering off, `/card/<token>` no longer renders on demand, which would have failed with Error 1102 on the Free plan. It redirects a hero record to its file and answers 404 otherwise.
	- C's deploy step copies the functions and the cards in one step, and fails if a card didn't land. Otherwise a preview could name a card that isn't deployed and get the app's page back.
	- The hero script refuses a card file with no entry in the list and a record path outside the repository, and creates its folder. Both guards were checked to fail on bad input.
	- The README notes that a hero card is a snapshot of its record file, so it's re-rendered if that record gains estimates.
	- The exports test also checks that a real run never says "illustrative".

- **Finding 12, pre-checked on `ws-c` `fb30b3e`** (C's fix `5850bca`: Boarding clears its tags when the job changes): `model-tags.mjs` passes 12 of 12, the same-page case included [measured, Chromium, a scratch copy of `ws-c`]. It moves to Done once re-checked on `main`.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, morning (Chiya's queue of 07:04: README, D2, licences, submission, D4)

1. **Final README:** `docs/readme-final.md`, for the PM to swap in for `README.md`. It combines C's README on `main` with my draft, which it replaces; the test now guards the final file. A new Status section, from `main` `a52753f`, says what's on in production, what's behind a flag, and what's not used or not measured. A read-only research agent compiled it from every status file and the code, with sources.
	- Decision: the divergence classifier's "AUROC 0.58 on held-out runs" is cited as given by Chiya, with a TODO for B's report. Basis: the number isn't in any branch or doc in the repo.
	- Decision: the flags that gate nothing (most `engine.*`, `campaign.planner`, `corpus.sweep`, `probes.*`) aren't presented as switches. The README names what each really does: the engine's own switch in `engine/api.py`, and `engine.memory` gating every prediction.
	- Decision: every dollar figure is tagged derived (tokens or time at list or standard prices), as the README already tagged its own costs.
2. **D2 share cards** (`workers/`): `/s/<token>` (link-preview page, then on to the cockpit), `/card/<token>` (1200 x 630 PNG) and `/badge/<token>` (SVG). Every number comes from the record, with the app's status word. Checked three ways:
	- 15 tests;
	- the Workers runtime (`wrangler pages dev` with C's functions, against a mock API);
	- four rendered cards, looked at.
	- Decision: resvg-wasm and the fonts are vendored unmodified, so C's deploy only copies files. Basis: a Pages Function bundles from `web/`, so imports from `workers/node_modules` wouldn't resolve after the copy. The bundle is 1.1 MB gzipped [measured].
	- Decision: responses are cached for 2 minutes. Basis: the API marks records no-cache because a running job revises them.
	- Asked C to copy `workers/` in at deploy and run its tests in CI.
3. **Licences after round four:** one new image (torchtitan: 90 packages, 75 permissive; 12 NVIDIA libraries and 3 MPL-2.0 flagged). The repository now bundles torchtitan's own files (BSD-3-Clause) and a C4 slice (ODC-BY 1.0; attribution in the README), plus D2's vendored resvg (MPL-2.0) and fonts (OFL-1.1). No new engine, API or web dependencies.
4. **Submission drafts** (`docs/submission/`): portal answers, the gallery blurb, the Meta models and partner tools table, and the X post. Every number is in one table with its source; dollar figures say "at list price".
	- Spark Open triage is described as it runs now: wired in behind `harness.spark_open_triage` (off), not yet run on a real job. There's a TODO to add the first real run before submitting, since the prize needs the partner to verify real calls.
5. **D4 against real runs:** not possible yet (`docs/decisions/freetier-one-session.md`). On A's L4 records for retrieval, GNN and vision, the engine abstains on `time.total` because the script maps leave `data.n_examples` empty. The LoRA record has no estimates, and no run was timed to the end.
	- The simulation stays, labelled simulated.
	- A new `check.realruns.test.ts` checks D4 on those four real records: no session count without a trusted time, memory judged on the record's own peak, and abstains where it should.
	- Asked A to fill `n_examples`. That also unblocks time and cost on those three lenses.

**Independent read-only review of the batch,** and what changed from it. It checked the README's status against the code, the share-card code and its injection safety, the licence counts, every submission number against A's records, and D4's fixture. Fixed:
- **Planted bugs presented as discoveries.** The submission copy presented the fine-tune, agent and GRPO problems as discoveries. They were planted in our own test scripts, and the copy now says so everywhere.
- **The fix loop's 19 s.** It sat next to "lands only when the re-probe verifies it". Gating on the re-probe isn't wired end to end yet (C's status), so the README, portal and tools table now say Muse Code produced the correct patch in 19 s, and that gating isn't wired yet.
- **The share card under-read peak memory.** It took the train steps' 14.6 GiB, not the eval pass's 20.4 GiB, on the record the badge uses. It now reads both peaks, and the OOM line names the eval pass and its batch size (`error_code` `oom.eval`).
- **API outages.** A 502 or 503 was cached as a generic card for 2 minutes. It's now `no-store` with its real status.
- **The CPU limit.** A card render takes about 140 ms of CPU (88 to 284 ms) [measured, Node], against the Workers Free plan's 10 ms a request (Error 1102 past it; Pages Functions are billed as Workers). So card images are off by default behind `PREFLIGHT_SHARE_CARD_IMAGES`; the share page then gives text previews and the badge still works. **Decision for Chiya:** the Workers Paid plan, from $5 a month, to turn card images on.
- **README.** The badge is commented out until `workers/` is deployed. Check-in, the guide, exports and the fix loop's re-probe are drawn as behind a flag or not wired. The scorer command gets its manifest argument. The diagram SVG is on white, which GitHub's dark mode needs.
- **Tags.** The AUROC is now cited (Chiya), not measured. The corpus figures are tagged, the dollar figures are tagged derived in the submission table too, and four portal numbers that had no row now have one.
- **D4.** The GNN's peak is 5.0 GiB (4.5 to 5.6), not "6.0 GB", which was the interval's top in decimal units.

Seen, not yet confirmed: A's agent re-sample (GitHub run 41301359530 on `origin/remote-runs`) stopped after 120 s with no model calls, "up to 4 may be stuck". That fits plant 3 firing. Once A logs it, the fallback cut can carry the measured stall.

Cloud runs: none. GPU minutes 0, dollars 0. A local `wrangler pages dev` and a mock API only.

## Sun Nov 1, 07:00 (pushed; waiting on C's fix for 12 and A's goldens)

Pushed `0683d27`, `868195d` and `b2f14f0` with Chiya's yes (06:59). `ws-d` now has `main` `a52753f` (round four and contract 0.6.0) merged in. Contract 0.6.0 needs nothing from D: every field is optional, D's 247 add-on tests and the typecheck pass, and so do D's Python tests. C has the fix for finding 12; it isn't on `main` or `ws-c` yet. Once it lands I re-run `model-tags.mjs` on `main`, and a check-in is scheduled for 09:00. The dial rerun waits on A's `make_golden.py` change (Chiya has asked A).

## Sun Nov 1, morning (round four: audit re-check on `main`)

Pushed `ae32870` with Chiya's yes ("push it all", 06:13). `main` had moved to `c4b95d4` (round four), so I merged it locally as `0683d27`. There were no conflicts and gitleaks is clean, and D's tests pass, 36, the README hook test included: it now runs against B's real scorer. That merge came after the yes, so it waits for one of its own.

**Audit re-check on `main` `c4b95d4`**, 12 first, as planned, then an independent read-only review (`docs/audit/nov1-main/README.md`) [measured, Chromium]:
- **12 partly fixed, still open.** Each job's tags now come from its record, and the five single-job cases pass at both sizes (new `docs/audit/scripts/model-tags.mjs`). But the read-only review found that tags carry over to the next job in the same page: after a triaged job, one that fails before any call shows both tags. I reproduced it, and tested a one-line fix (`useEffect(() => setCalls([]), [jobId, open])`) on a scratch copy, C's file untouched there: 12 of 12 pass, against 10 of 12 without it, and the typecheck and C's 41 app tests pass. Sent to C with that fix. The check now has the same-page case and exits 1 on a mismatch.
- **7, 10, 8, 11, 9 and 4 fixed** (the review re-ran or spot-checked each):
	- 40-state sweep: no axe violation in any state, no element without a visible focus ring, no overflow, no app console errors;
	- the "About ..." buttons are 24 px;
	- Boarding after a failure shows "Stopped here", and the route labels show whole.
	- These moved to Done in `from-d.md`.
- **6 not fixed** (the job map on phones): at 390 px the share lines render at 5.5 px and the names at about 6.6 px. It stays open with C.
- **Minor, not failures** (axe passes both on spacing): "Your limits" is 20 px tall, and the "Got it" label is 22 px. Sent to C as a note.
- **Add-on check: 20 of 20 now** (from 14). The six nodes axe can't measure still pass by pixels, at 5.88:1 or better.
	- Decision: the check warns only on the dev server's 502 for `/v1/guide/limits/state`, the cockpit's own limits request, and still fails on any other `/v1` 502, an add-on's included (checked both ways). Basis: that request needs the API, which the dev server doesn't run, so the 502 comes from the test setup, not an add-on. A first version warned on any `/v1` path; the review found that too broad.

## Sun Nov 1, morning (pushed; dollar figures back in the cuts)

Pushed `88a1c72` and `8588646` with Chiya's yes (05:59). Chiya's ruling: dollar figures go back into both video cuts, spoken and captioned "at list price" (e.g. "$0.042 of GPU at list price"), because the rubric asks for dollars and that wording is accurate. The README keeps tagging them derived.
- Plan A's storyboard: $0.042 is back in shot 8 and in the 45-second cut's shot B, spoken and captioned. Shot B is now about 35 spoken words in 16 s, so the plan trims its first sentence to the caption if the take runs long. Both rule lines now name list-price dollars next to measured numbers. A narrow read-only review checked every dollar value and basis against A's status file; its five wording and count fixes are in.
- Fallback cut: shot 4 carries $0.042. Shot 11 carries $0.43 for the 30-task sample on screen ($0.4289, `a-agent-sample-1`) and $1.67 for all 130 tasks ($0.4289 + $1.2371). The numbers table gives each one's basis. The end moves to about 2:58 to give shot 11 room.

## Sun Nov 1, small hours (Chiya's queue of 01:19)

`613ae92` was already on `origin/ws-d` when the yes came, so there was nothing to push. `main` is still at `90fcc40`: the audit re-check (12 first, then 7 to 11) and the dial rerun wait for round four.

- **Fallback 3-minute cut** (`docs/demo/fallback-cut.md`, linked from `storyboard.md`, which stays as plan A's fine-tune shots). 13 shots, about 2:58: the fine-tune's OOM at step 200, then the agent run. Every spoken number is measured and sourced: step 200, 189 s of probing, 20.4 of 22.06 GiB, 130 tasks in two runs, 0 of 905,739 input tokens cached, and 30 tasks in 49 s. Decisions, with bases:
	- The retry loop is shown only as "read from the code" (`run.py:121`). Basis: A measured 0 runaways in the 130 tasks (`a-agent-builds-1`), because they ran before plant 3 was moved to rows 26 to 29, and A hasn't re-sampled since. If a re-sample shows the stall before Tuesday, its numbers go in the table.
	- The turn-11 overflow and the full-run cost aren't spoken. Basis: the first is derived and the second predicted; they may appear on screen under the app's own labels.
	- The agent shots need the Oct 31 record published on the deploy with a real share link, since its token in the repo is now a placeholder (A, `e27d2fc`).
- **Validation table hook.** B's scorer (`scripts/score_validation.py --readme`, `ws-b` `6ce7730`) already writes between `<!-- validation:start -->` and `<!-- validation:end -->`. The README draft now holds those markers around the scorer's own output on an empty manifest, so Monday's run replaces only that block.
	- End to end: B's script with B's fixture manifest, run on a copy of the draft, filled the block and left every line outside it unchanged.
	- `tests/test_d_readme_validation.py` checks the markers now, and the round trip through B's `into_readme` once B's module is on the branch. It passed against B's module from a `ws-b` copy.
	- Two strings are B's, so I asked B for them: "not measured" becomes "not measured yet" (Chiya's wording), and a blank line goes before the footnote. Without it, GitHub's Markdown renders the footnote as a table row (checked with marked 14.1.4).
- **Pretraining-grid sample.** Nothing in the repo mirrors the lab project. The campaign fixtures already use batch size × sequence length × seed, and no branch has a language × filter grid. The "3 languages × 6 filters × 3 data sizes" row is only in the plan PDF's bug-zoo table (page 16), which I can't edit, so it's back to Chiya with a replacement row.

- **Read-only review of this batch,** and what changed from it. No number lacked a source, and the sums, the retry-loop treatment, the seeded block and the GFM claim checked out. Fixed:
	- **Dollar figures.** The cut spoke $0.042 and $1.67 as measured, but they are measured time and tokens priced at list rates, which is how the README draft already tags its own costs: derived. Decision: no dollar figure is spoken or captioned in either cut. Plan A's storyboard is corrected too (its 45-second cut said "about four cents"). If Chiya rules that a list-price cost may be captioned with a "derived" tag, they can go back as captions.
	- **Overclaims fixed:**
		- shot 4 now says "189 seconds of probing";
		- shot 7's caption separates the record's 30 tasks from the 130 in two runs;
		- shot 8 states the timestamp as a fact in the code, not a measured cause;
		- shot 11 now says the screen came from a 30-task sample in 49 s, not "130 tasks and $1.67";
		- the full run's cost on screen is the record's $7.15, "Predicted", not A's $6.4.
	- **Dependencies.** The dev-server fallback is gone (the cut is recorded on the deploy or not at all). Shot 12 lists B's Monday scorer run as a dependency. The retry-loop line notes that `run.py:121` is the line in the script as the record read it.
	- **Test.** It now asserts that the seeded block is exactly the scorer's output, and it fails on a one-word hand edit, which I checked. A third string went to B: the GNN row's "+4.8%" is percentage points.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, night (add-ons "done means"; correction pushed)

Pushed `613ae92` (the status correction) with Chiya's yes. `main` is still at `90fcc40`, so none of C's audit fixes have landed; the re-check on `main` starts with 12 (model tags) once they do.

I read the PM's new project doc `unseen-scripts.md` (the held-out validation set, private to Chiya until the validation runs). It stays out of the repo, and nothing I build or tune uses it. The dial's order stays frozen for the Nov 2 test.

**"Done means" for D1 to D5**, the plan item no other agent's work was holding up:
- A README line for each add-on, with its flag: a new "Add-ons" section in `docs/readme-draft.md`, covering D1 detail level, D3 currency, D4 free GPUs, D5 notebook and exports.
- A render check, `docs/audit/scripts/addons.mjs`, for each add-on in desk and phone, dusk and day. With its flag on, it must render on its host screen, with no page or console errors, no axe violations inside it and nothing under 24 px; it also opens D1's glossary card and loads a real notebook into D5 so the preview is checked. With its flag explicitly off, the host screen must load without errors and the add-on must be absent (for D5, no Notebook tab). It prints each axe node's colours and exits 1 on any failure.
- Result on `ws-d` `613ae92` (`main` `90fcc40` plus D's commits) [measured, Chromium, Nov 1]: 14 of 20 states pass. All add-ons render where they belong and are absent with their flags off, with no errors and no targets under 24 px. The 6 failures are day-theme colour contrast in D1, D4 and exports: all 44 failing nodes are #3B5F70 on #9EBAE0 at 3.45:1, the shared `--pf-fg-dim` token on the sky (audit finding 7, with C). No other pair fails, D4's chips included. My Oct 30 note that D's panels had no axe violations didn't hold for the day theme. Asked C to run the check nightly (CI is C's).
- Independent read-only review of the check and the README rows, and what changed from it:
	- the check now proves the colour pair instead of me asserting it;
	- the flag-off half waits for the host screen, fails on errors, and uses `?flags=-<flag>`; I confirmed that it can tell (Notebook tabs: 1 on, 0 off);
	- it covers the glossary card and the notebook preview, which it hadn't;
	- the console filter is narrowed to the tunnel and the font host;
	- it says dev server only, since preview builds ignore URL flags;
	- the README rows now say only what ships: D5 previews but checks nothing in while `runner.queue` is off; D3 covers 30 ECB currencies; D1's warnings rule waits on C's findings list; and the exports CLI item and link name are corrected.
	- A second read-only review found the D1 flag-off half opened no knob, so it couldn't fail for the glossary card, and the notebook's "What will run" stayed closed. Both are now opened. The check also warns on text axe can't measure. It warned on 6 nodes, which I measured from screenshots with the text hidden: D3's select is at least 8.89:1 and D5's code blocks at least 5.88:1, so both pass [measured, Nov 1]. Results are labelled `ws-d` `613ae92`, the tree they ran on.
	- Decision: not claiming that C's token fix turns the check green. That depends on the value C picks, and the check will show it.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sun Nov 1, early (agent plant 3 made to fire)

Pushed `372b60e` with Chiya's yes, as `ee55977` (my merge of `main` `90fcc40` on top, then the push, at 00:29:56 and 00:29:58). **Correction, Nov 1, 01:35:** I first reported that something else had made that merge and push. It was my own turn, answering Chiya's yes. That turn was interrupted and the message re-sent, and after the restart I couldn't see the first attempt, so my second push found nothing to do. The record shows it: this session's transcript (my command at 04:29:49 UTC) and `origin/ws-d`'s ref log ("update by push" at 00:29:58). No background agent, process or hook pushed: none is running (ListAgents, `ps`), the stop hook only reads git, and the environment manager holds only signing config. Then `6215d1b` was pushed with Chiya's yes at 01:33. The data-use gap is settled: A gates the Muse Spark call on `harness.spark_script_map`, C's notice follows the same flag, and the flag goes on for the demo.

**Plant 3 now fires** (`zoo/agent_eval_spark/run.py`; this commit, not yet pushed). A ran 130 agent tasks and the retry loop never started, because MBPP's builds almost never fail. Now rows 26 to 29, the last 4 of the first 30 tasks, get a test fixture that imports a module that was never shipped (`from mbpp_fixtures import cases`). Their `run_tests` fails with the same `ModuleNotFoundError` on every call, whatever code the model writes, and `call_tool` retries it without end. The header says so as planted. Plants 1 (no caching) and 2 (overflow at turn 11) are unchanged in the code.

Decisions, with bases and tests:
1. **The last 4 rows, not a random 4.** A's sampler takes tasks in the script's order (the script doesn't read `PREFLIGHT_TASK_INDICES`) and stops the whole script once no model call has come through for its stall time: 120 s on `ws-a` (Chiya's 2 minutes), still 600 s on `main`. With 4 workers, the run stalls as soon as all 4 are stuck, and no later row starts. My first, seeded pick (rows 4, 10, 12, 20) would have stalled after row 20, cutting A's 30-task sample to 21 tasks and thinning plant 2's evidence; the reviewer caught it. With rows 26 to 29 all 30 tasks start first. Measured offline, against a local fake API on 30 MBPP-shaped tasks whose fake model always submits a correct solution: all 30 started, 26 passed, and exactly rows 26 to 29 looped, about 420 tries each in 15 s. Test: `test_agent_retry_loop_is_planted_on_the_rows_the_header_names`. It checks that the header's rows match the code, that `run_tests` uses the fixture for those tasks, and that the fixture, run as `run_tests` runs it (a temp file, no `-I`), fails with the same `ModuleNotFoundError` three runs out of three.
2. **The file stays near the size A measured**, because the echo sets the context growth behind turn 11. I trimmed the header and three comments to pay for the new lines: 7,699 bytes against 7,641. What was dropped: "the probe needs both, and a route to that host", "500 tasks", "30 tasks" and "none ran past turn 4" (both still in my status), and "any other tool error loops the same way" (still true in the code). Derived, scaling A's measured growth by the echo's share: about 2,764 tokens a turn, so a typical task holds about 25,612 tokens at turn 10 (388 under 26,000; it was about 300 before this change, and 562 at A's exact size) and 28,376 at turn 11. A's re-sample measures it. Test: the size guard (within 300 bytes) and the replay (the script's own check still stops at turn 12).
3. **Side effects, judged low:** each retry writes a temp file (`delete=False`), so a looping task leaves a few thousand files in the probe's temp folder before the 120 s stall stops it. And a model could in principle make the import succeed by writing `/tmp/mbpp_fixtures.py` through `run_python`, since the echoed header names the module. Neither changes what the probe sees.

No paid runs: A re-samples once this is pushed.

**Audit 7 to 12, pre-checked on C's branch** (`ws-c` `80d4217`, not yet in `main`; `docs/audit/nov1-wsc/README.md`), measured with the same scripts on an export served locally:
- 12 (model tags) is not fixed: a job that ran out of memory while probing still shows "Muse Spark Open" on the triage stage;
- 7, 10, 8 and 11 are not fixed (same axe states and targets);
- 9 (phone route labels) and 4 (the stopped stage) are fixed there.

Each moves to Done once re-checked on `main`. New and minor: the "Your limits" summary is 20 px tall; axe passes it on spacing.

## Sat Oct 31, late night (submission materials)

Pushed `d93618a` with Chiya's yes. Chiya kept the batch time as it is: the predicted upper bound rounded up, plus the user's own start-up time, with no number of ours. Finding 4 is with C.

**README draft** (`docs/readme-draft.md`, for the PM to merge), following the plan's submission checklist:
- what Preflight is in two lines; the one-command install; a quickstart with the CLI commands;
- the architecture as Mermaid, drawn from the code on `main` rather than the plan's widget (whose image I can't read as text); also rendered to `docs/readme-assets/architecture.svg` and checked to parse with `mmdc`. What isn't running is dashed or labelled: the fix loop, Spark Open's claim check, and AMD (CLI only);
- the built-during-the-hackathon statement, citing the first commit (`11a0ce9`, Oct 26, 22:25 ET);
- the Meta models and partner tools with where each is used and its status on `main`, from A's, B's and C's status files and the code. The status column says what isn't live:
  - Muse Code's fix loop isn't reachable from the app;
  - Spark Open's claim check isn't wired;
  - Glimmer is only in the price table;
  - AMD has no MI300X result;
  - the public deploy on Modal and Cloudflare is pending.
  The PM updates these at merge;
- one run's cost and time, each tagged: 6 min 45 s and 386 s of probes [measured, A, Oct 28], which is over the plan's 5-minute target; GPU $0.0858 and Muse Spark $0.0178 [derived: measured time and tokens at list prices];
- the empty validation table with all 15 targets from the plan's "Validation targets", for B's scorer;
- licences: the repo's, a link to the dependency table, and every zoo script's model and data.

The `pip install git+...` line is checked only from this checkout [measured: three CPU probes ran and wrote a record]; Chiya will try the URL once the repo is public. Opening a CLI record in the web app isn't built (the app opens records by share link), and the draft says so. TODO(PM) marks the public link and the status column.

**Licences** (`docs/licences.md`), 377 packages, a snapshot of Oct 31:
- Sources: both `uv.lock` files walked into runtime, extras and dev; both `package-lock.json` files; and the four Modal images' pins resolved with `uv pip compile`. The images were built earlier, so their indirect versions can differ.
- Each licence comes from PyPI's JSON API or the npm lock entry. Where that metadata is empty or vague, the row says so and names the source used instead. Our own `preflight` package is marked Apache-2.0 by hand, because the PyPI package of that name is someone else's.
- No GPL, AGPL or SSPL anywhere, and all 26 packages that ship in the browser are permissive. CI already checks the Python environments and `web/node_modules`; the Modal images are what this adds.
- Flagged and explained:
  - NVIDIA's CUDA libraries (proprietary, or no licence in their metadata; never in the repo);
  - weak copyleft we don't distribute: certifi, tqdm and lightningcss (MPL-2.0), pycountry (LGPL-2.1, serving image);
  - three share-alike datasets the zoo downloads at run time: TinyStories, dolly-15k and Natural Questions.

**The 45-second X cut** (`docs/demo/storyboard.md`), ready to record: four shots with timecodes, exact screens and actions, and burned-in captions, including the crash line, so it works with the sound off. Narration is 59 written words, about 66 spoken, so about 27 s of speech; the GiB figure is captioned, not spoken. It speaks only measured numbers. Step 200 is now also measured: A's full run of Oct 31 ran out of memory at step 200, at 22.13 GB (20.61 GiB). I dropped two claims about X (its length limit, and that most people watch muted) because I found no primary source for either.

**Audit findings 7 to 12, ranked, to C** (`from-d.md`):
1. 12, model tags shown whether or not the model ran (misleads judges; the Judging playbook asks for every Meta model labelled at the moment it works);
2. 7, day-theme contrast (16 states);
3. 10, focus rings;
4. 8, eval-marker targets;
5. 11, small targets;
6. 9, phone route labels.

Each request names its WCAG criterion and how it's re-checked: the sweep measures 7, 8, 10 and 11, and 9 and 12 are a look at the boarding screen. None is fixed on `main` or `ws-c` yet. The sweep and targeted-check scripts are now in `docs/audit/scripts/` (run with `NODE_PATH=web/node_modules`).

**Found on the way, to C and A:** the data-use notice shows "Muse Spark reads the code" only when `harness.spark_script_map` is on (off everywhere). But the runner maps every job with Muse Spark whenever the Meta key is configured, with no flag check. So visitors aren't told their code goes to Meta's API. Request in `from-d.md`.

A read-only review (git-free copy) found 15 issues in these materials; all are fixed above.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sat Oct 31, night (round three merged; agent overflow at turn 11)

Merged `main` at `006bc57` (round three, contract 0.5.0) as a fast-forward. On it: Python passes 1177 with the `torch`-marked tests left out, `web` passes 396 in 55 files, and `tsc` is clean.

**Agent overflow is at turn 11 by Muse Spark's count (correction from A), derived from A's measured growth.** A's run of 30 of the 500 MBPP tasks on muse-spark-1.3 (Oct 31, run 41250821790; `zoo/agent_eval_spark/record_oct31.json` on ws-a) grew the history by 2,745 tokens a turn by the provider's count (median of 43 turn pairs, 2,653 to 4,013). My header used characters / 4, about 2,450 a turn. Turn 11 is derived, not observed: no task in the sample went past turn 4 (largest context 10,862 tokens). From the record's per-turn counts (`context_tokens_by_turn`), a typical task starts at 733 tokens (698 to 946) and grows 2,745 a turn, so it holds 25,438 tokens at turn 10 and 28,183 at turn 11, passing 26,000 at turn 11; across the middle 80% of the measured growth (2,676 to 3,187 a turn) that's turn 9 to 11. A reviewer's turn-10 estimate assumed a 2,627-token start, but the record's first turns are 698 to 946. The header now gives turn 11 with that tag and range. The header now says turn 11, and that the script's own character-based check stops the task a turn later. I kept the header short so the script stays near the size A measured, because it echoes its own source every turn: 7,728 bytes against 7,641, about 30 tokens a turn more, which still leaves turn 10 under the limit. Test: `test_agent_script_stays_the_size_a_measured` (within 300 bytes of the measured version, about 4% of the growth; a bigger change needs A to measure again) and `test_agent_scripts_own_check_stops_at_turn_12_with_short_replies` (the script's own check).

**Audit re-check on round three** (`docs/audit/oct31/README.md`, screenshots there). Findings 1 (the bad-link message), 2 (the phone hint to "Got it"), 3 (honest sample labels) and 5 (the measured check-in time) are fixed, each checked in Chromium. Finding 4 is not fixed: a failed job's boarding screen marks no stage, and nothing says "Stopped here". The whole 40-state sweep matches Oct 30: no overflow, no console errors, colour contrast in the same 16 day-theme states, eval markers under 24 px, the same two job-map nodes with no focus ring. Findings 4 and 6 to 12 stay open with C, which lists them as next.

**Dial on B's engine output.** B's `make_golden.py` hasn't run on the golden records yet (B: it waits on A's 30-task agent top-up), so I ran B's `engine.api.fill_predictions` on copies of the three golden records in my scratch folder (CPU, no cost) and ran `dial.zoo.test.ts` once against them, unchanged except for where it reads records. The dial counts only records whose script has written ground truth (fine-tune and pretrain); the agent record has none in the dial's table, so it's read but not counted. Simulated: B's models on the golden records' hand-made probes, not on measurements. In-sample, 2 records, 6 planted problems:
- On screen today, both records have 3 of 3 planted knobs in the dock (Wilson 95%: 44 to 100%).
- In the design check with C's list, all 3 are visible on both; in the dock, finetune has 2 of 3 and pretrain 1 of 3.
- Every one of the nine order variants ties at 3 of 6 (19 to 81%), so nothing can beat today's order and it stays frozen.

That is identical, count for count, to the fixture predictions. The engine output is B's real models (memory-analytical@0.3.1, time-total@0.1.0, cost-gpu-hours@0.1.0, steptime-phases@0.2.0 and others), but the records' probes are still the hand-made golden ones, so this is not yet a run on real measurements. The real test stays the held-out Nov 2 run, with the policy frozen. Nothing from the scratch run is committed. Test: the dial test itself; rerun it when `make_golden.py` writes engine output into `schema/examples/`.

**Exports moved onto C's hook.** Round three brings `registerExportsPanel`. Its "Take it with you" section hands each panel C's `ExportPlan`, and C asks exports to format only, with no numbers of their own, and to say "not verified" when `codeChangesVerified` is false. The panel now registers there instead of the general `below` slot, sits under C's heading, and formats the plan: changes with A's code locations, GPU, launch line, readouts (null where the cockpit abstains), `wallTimeHiS`, patches and the share link.
- The batch time is the plan's upper time, rounded up to 5 minutes. I dropped my 15 minutes of setup, which was a number of my own; the file now says to add the cluster's start-up time if it counts. The time limit still caps it.
- "Not verified" shows on screen, in the plan JSON, in the edit script and in the batch file whenever a code change lacks a passed patch.
- The edit script's safety work is unchanged.

A read-only review (git-free copy) found these, all fixed and tested:
- A GPU count the script sets on its launch line (`--nproc_per_node`) was treated as launch-only, so Slurm got 8 GPUs while the script ran on 4. It is now an edit.
- Rounding up to 5 minutes could pass the time limit; the ask is now held to the limit.
- A template-only change was listed as a code change that C's `codeChangesVerified` doesn't count, so it never showed "not verified". Only what C counts is listed as code now.
- GPUs per node was a number of my own; the file now asks `--gpus=<the plan's count>` for the whole job.

Checked in Chromium on `lora-7b` with `c.exports` on: one panel, inside C's section, and the panel and file agree (4 h 55 min, `--time=04:55:00`). Test: `build.test.ts` and both panel test files now run on C's own `exportPlan` (49 tests), and the registration test checks it's in `exportsPanels` and not in `panels`.

**Notebook intake now calls C's `checkinBytes`**, as asked on Oct 30, so the panel and the desk can't measure differently. Test: the notebook suite (50 pass).

Test runs: `web` 55 files, 402 tests pass, and `tsc` is clean. Python passes 1178 with the `torch`-marked tests left out. The audit's 16 contrast states are the same states as on Oct 30, checked state by state.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sat Oct 31, evening (pushed; the exports CLI item)

Pushed `ws-d` with Chiya's yes (`895633b..129bfb1`: 3590fa7, 59acc38, 4ff522e, 129bfb1), as a normal push after merging `main` (no change, still 3c243b5).

Chiya's decision on the exports CLI line: build it now, behind a setting that turns on when the repo goes public (Nov 3 to 4). It's a new item, "Check it again on your machine", behind the new flag `d.exports_cli` (off in `flags.json`). When on, it shows `pip install git+https://github.com/uchuuronin/preflight` and `preflight run . --entrypoint=<entry> --out .preflight/records -- <args>`.

Decisions, each with its basis and the test that fails if it's wrong:

1. **The plain install is enough; no extra.** Measured: installed from this checkout (the same package a `git+` install builds, but not the URL itself, which can't be tried while the repo is private) in a fresh Python 3.11 venv that also has torch 2.5.1. Run as a non-root user on a small torch loop, the generated command ran three CPU probes (all `ok`) and wrote a record; run twice, both records carry the same content hash. Without torch the CLI installs and maps, but its probes can't start, so the item says to install it where the job's torch is. psutil (the `probe` extra) isn't needed. Test: none in CI, since it needs a network install. The steps are here to repeat, and the URL install should be tried once the repo is public.
2. **Offered only where it can work.** Not for API runs (the CLI measures nothing for them), samples (no folder of the visitor's holds the script) or notebooks (the entry point is a generated `.py`). Derived from A's `pipeline.py` and `sources.py`. Test: `build.test.ts`, "offers no re-check", and `ExportsPanel.cli.test.tsx` (no item on a sample).
3. **The command's shape.** It runs from the project's top folder with `.`, the same folder the edit script and the run command use. `--entrypoint=` keeps a name starting with a dash from being read as an option, which argparse does otherwise. Records go to `.preflight/records`, a dot-folder the CLI skips when it reads the project, so a second run doesn't read the first one's output. Arguments are shell-quoted after `--`, and any holding a control character are left out, with a note, here and in the run command. The copy says "on its GPU if it has one", because `preflight run` falls back to the CPU. Test: `build.test.ts` (the CLI tests and "leaves out arguments").
4. **Hidden until the flag is on.** `flag()` is `=== true`, and URL overrides are off in production builds. Test: `ExportsPanel.test.tsx` (no item and no `pip install` text with the real flags) and `ExportsPanel.cli.test.tsx` (shown when on; the flag is off in `flags.json`).

A read-only review (git-free copy) found decisions 2 and 3's gaps (the item on API runs, samples and notebooks; records read back on the next run; dash names; control characters; "your own GPU" when it may run on the CPU), all fixed above.

Test runs: `web` 47 files, 364 tests pass, and `tsc` is clean; Python passes 974 with the `torch`-marked tests left out.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sat Oct 31, late morning (exports panel)

The exports panel, moved from C, is in `web/src/addons/exports/`: a band under the cockpit's whole run (C's `below` slot) behind the existing flag `c.exports`, so no flag key changes. From the record and the settings on screen it offers five things to copy, and the three files also download: the plan (`preflight-plan.json`), the edits (`preflight-edits.sh`), a Slurm batch file (`preflight.sbatch`), the command, and the record's link. Checked in Chromium at 1280 and 390 px on the `lora-7b` sample with `?flags=c.exports`: it renders, every control is at least 24 px, and there's no sideways scroll.

Decisions, each with its basis and the test that fails if it's wrong:

1. **"Patch" is an edit script, not a unified diff.** Derived: a diff needs the script's text, and the browser has only the record, which gives each settable knob a file, line and expression (A's script map). An edit is exact only when the expression is a plain `name = value` or `name=value` holding the record's value, and the new value has the same type; anything else (precision written as `bf16=True`, list elements, type changes) is listed to make by hand. Each exact edit is one `perl -i -pe` line, so it works on Linux and macOS, that changes that expression on that line only and stops the script, changing nothing, if the line no longer holds it. Test: `build.test.ts`, which runs the script with `sh` and `perl` on the zoo fine-tune script (exactly lines 32, 88 and 91 change; a moved line is refused with its reason; quotes, pipes, hashes and backslashes survive; `$` and `@` are refused).
2. **The edit script is safe to run on a record someone else made.** Reviewer-proven before the fix: a line break inside a value, expression or template in a `#` comment ran `$(touch pwned)`; `lr=1e-4` also matched inside `warmup_lr=1e-4`, and `n = 4` matched a line now reading `n = 40`; a file named `-e...` reached perl as an option; and a failing second edit left the first applied. Now the Perl program is fixed text, and the file, line and both expressions reach it only as quoted arguments and environment variables. An expression must appear on its line exactly once, as a whole token. Every edit is checked before any is made. Every comment passes through `oneLine`, and `--` comes before each file name. Test: `build.test.ts`, "edit script safety": seven cases run with `sh` and `perl`, covering the injections through a value, an expression, a template and a GPU name, `warmup_lr`, `n = 40`, `$ @ | # \` and quotes as data, a dash file name and the partial run.
3. **Muse Code patches are listed by title and verify status only.** Their diffs are `blob://` references and nothing serves blobs to the browser. Request to C below. Test: `ExportsPanel.test.tsx`, which lists patches with their status.
4. **Batch time = slow end of the predicted range + 15 minutes, rounded up to 5 minutes, never above the record's time limit.** Derived: the same rule as the free-tier band (D4), so the two never disagree. When the plan needs longer than `wall_time_cap_s`, the file asks for the limit (Slurm won't start a job asking for more) and says to checkpoint and resubmit; the panel shows the same figure from the same function. With no estimate it says to set `--time` yourself. API runs ask for no GPU; `--gres` counts GPUs per node. Test: `build.test.ts` (sbatch block) and `ExportsPanel.test.tsx`.
5. **The link is the cockpit's own address**, `#/cockpit/<ref>`, from the address bar when it names one, else the record's share token. The record's internal id never appears in any export. It opens the record at its own settings, and the panel says changes travel in the plan file. Test: the link and `record_id` checks in both test files.
6. **The command is the record's launch line plus its arguments, shell-quoted**, falling back to `python <entrypoint>`; A's launch line `python <entrypoint>` is rebuilt with the path quoted, so a space survives. I left out a "re-check with Preflight's CLI" line: the repo is private (GitHub API, Oct 31), so a stranger has no way to install the CLI, and I'd be offering a command that can't run. Decision for Chiya below. Test: `commandText` in `build.test.ts`.
7. **Everything is generated in the browser from what C's registry passes in.** Pure functions, no network, so the files move with every knob and are identical for the same record and settings. Test: the rerender test in `ExportsPanel.test.tsx`.

A second read-only review of the panel found the problems in decision 2 plus smaller ones, all fixed: the panel and the file now count exact edits the same way, "1 needs" grammar, an API run with no estimate no longer says it asks for a GPU, copy messages keep the item's name and repeat for screen readers, downloads report when the browser can't save, a float stays a float (`lr = 1.0` set to 2 becomes `lr = 2.0`), and the copy says to run `sh preflight-edits.sh`.

Test runs: `web` 46 files, 358 tests pass, and `tsc --noEmit` is clean. Python passes 974 with the `torch`-marked tests left out; this box has no torch, and those 10 tests are A's.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sat Oct 31, morning (six zoo scripts)

Six scripts under `zoo/`, each with a header listing its planted problems, the findings A's static reader raises from the code, what only the probe can find, and "Ground truth: not measured yet". Public data and public models, except the agent's model, which is Meta's hosted Muse Spark as the plan asks. Nothing ran on a GPU or called a paid API. `tests/test_d_zoo_scripts.py` reads each script with A's `map_static` exactly as an upload would be read: 32 pass, 1 expected failure (decision 9).

Two independent read-only reviews (git-free copies). The first found five problems that would have stopped scripts before their planted problems showed, plus header arithmetic I had wrong. All are fixed below; each fix was checked, not taken on the reviewer's word. The second confirmed the fixes and left wording only (serving's vLLM sentence, its sequence count now 16 to 21 with prompts, an untagged GNN accuracy claim now dropped, stale replay figures here), all fixed.

Decisions, each with its basis and the test that fails if it's wrong:

1. **Agent overflow lands at turn 12 with a 26,000-token budget.** Simulated: the history after 10 rounds of short `run_python` calls is 24,749 tokens and after 11 is 27,208 (the script's own count, characters / 4, with its own source as the echo), so 26,000 sits mid-window. Run end to end offline against a local fake API (openai 1.57.0, httpx 0.28.1): both tasks overflow at turn 12, all 22 calls carry a different system prompt (no caching), and a failing `run_tests` was still looping when an 8-second timeout stopped it (planted problem 3). Can fail because the model's replies set the growth; A's probe measures the real turn. Test: `test_agent_context_overflows_at_turn_12_with_short_replies` (also catches the script growing or shrinking by more than about 400 characters).
2. **Agent pins `openai==1.57.0`, not 1.55.0.** Measured: the probe image resolves to httpx 0.28.1 (`uv pip compile` of `PROBE_PACKAGES`); in a Python 3.11 venv with httpx 0.28.1, `OpenAI()` on 1.55.0 raises `TypeError: ... unexpected keyword argument 'proxies'`, and on 1.57.0 it works and ran the offline check. Test: A's probe install and first API call.
3. **Agent fixes from review.** "Reply with code only" told the model to answer without a tool, ending tasks at turn 1; the prompt now asks for a tool call every turn. A malformed tool call or an API error now fails that task, not the run, and a failed task keeps its token counts (the offline run showed overflowed tasks reporting 0 prompt tokens, which would hide the caching evidence). The client is made in `main()`, so reading the script needs no key. Test: the offline run above (prompt_tokens 220, cached 0).
4. **Eval sweep uses DeepSeek-R1-Distill-Qwen 1.5B and 7B, not Qwen3.** Cited: Qwen3 needs transformers 4.51 or later (transformers v4.51.0 release notes, github.com/huggingface/transformers/releases/tag/v4.51.0), and the probe image pins 4.46.3. Review fixes: each batch's outputs are freed before the next `generate` (the reviewer estimated the 7B run's second batch at about 22.4 GiB with the old batch still held; a guess, as no GPU ran), the prompt no longer gets a second BOS token, and "cut off" (no `</think>`) is counted apart from "finished with no number". Test: A's probe.
5. **Serving: about 20 sequences fit at a time, against 256 admitted.** Derived: 57,344 bytes of KV per token (28 layers x 4 KV heads x 128 x 2 x 2 bytes, Qwen2.5-7B config); L4 22.06 GiB at 90% less 15.2 GB of weights leaves about 70,000 to 90,000 tokens (vLLM's activation reserve is a guess the probe replaces); replies now run to 4,096 tokens (`ignore_eos`, as vLLM's own benchmark does), so 256 ask for about 1.05 million. Review fix: chunked prefill at 8,192 tokens, so vLLM's start-up profile doesn't take a full 32,768-token prefill and refuse to start. Test: the probe's KV capacity and preemption count.
6. **Serving can't install in the standard probe image.** Checked with `uv pip compile` against the image's resolved pins: every vLLM release built for torch 2.5.1 (0.6.4.post1 to 0.7.0) needs numpy below 2, the image has 2.1.3, and 0.7.2 needs transformers 4.48.2; with numpy 1.26.4 (inside the image's own `>=1.26,<2.2`), vLLM 0.6.4.post1 still needs `pyairports`, which has no wheel. So A needs a serving image (request in `from-d.md`). The header says so. Test: A's install step.
7. **GNN: the outer hop carries about 3.6 million messages and a batch touches the whole graph.** Simulated: the script's sampler on CPU at full size, seed 0: 15,360, 323,000 and 3,591,700 edges per hop, and all 200,494 nodes in one batch (the old header said up to 9.6 million, ignoring de-duplication). FakeDataset draws its node count from Python's `random`, so the script now seeds it; its graph is undirected, so the degree is 50, not 25 (checked). Test: `test_planted_settings_are_found_with_their_values[gnn_pyg]` (the fan-out knobs); the hop counts are the probe's to measure on the GPU.
8. **Vision pins `torchvision==0.20.1` with `timm==1.0.12`.** Checked: both resolve against the image's pins, and torchvision 0.20.1 has a Python 3.11 wheel. Smoke-run on CPU with 130 generated images and an untrained ResNet-50. The beans dataset is served as parquet with no loading script (its Hugging Face file listing, read Oct 31), so the probe can load it. Test: A's probe install.
9. **One known gap in A's reader, kept visible.** It treats a vision script as using channels-last if the text `channels_last` appears anywhere, comments and docstrings included, so the header naming `vision.no_channels_last` hides it. `test_known_gaps_in_the_reader` is an expected failure that turns red once A's reader raises it. Request to A in `from-d.md`.
10. **The findings test is exact.** Every finding the reader raises must be planted or named on an "Also read, not planted:" line, so a new rule firing on a zoo script shows up. GNN and vision raise `data.repetition` (several epochs, info level, normal for both). Test: `test_planted_problems_seen_in_the_code_are_raised`.
11. **Retrieval batch is 8, and training is the long pole.** Derived: A's rule fires under 64; one epoch is 12,279 steps (98,231 pairs / 8). That training outlasts encoding the corpus is a guess the probe's times settle. Test: `test_planted_problems_seen_in_the_code_are_raised[retrieval_st]`.
12. **Agent settings match A's golden agent record except the context limit** (golden 32,768, script 26,000 for decision 1). Test: `test_agent_script_is_the_golden_agent_records_job`; raised with A.

Cloud runs: none. GPU minutes 0, dollars 0.

## Sat Oct 31: every plan item D owns

Sources: the Contracts tab (Contract D, the Oct 27 review, the competitive review's adopted items) and the main tab (bug zoo, exports, validation targets), plus the items moved to D on Oct 31. "On main" means merged into `main`; "on ws-d" means pushed to `ws-d` but not yet merged. Hours are D's own work, honest, not counting waits on other agents.

| Item | Where it is | Hours left | Waiting on |
| --- | --- | --- | --- |
| **Zoo scripts with planted problems (moved from A, Oct 31)** | | | |
| Agent eval on Muse Spark (no prompt caching, context overflow at step 12, a tool that retries forever) | written, local commit on ws-d; run offline against a fake API | 0 | A's probe needs the Meta API key and host (request filed); A probes it (paid, Chiya's yes) |
| Eval sweep (no thinking cap, activations dumped at every layer) | written, local commit on ws-d | 0 | as above |
| Serving with vLLM (max context too high for the concurrency) | written, local commit on ws-d | 0 | a serving image from A (vLLM can't install in the probe image; request filed) |
| Retrieval with sentence-transformers (contrastive batch too small) | written, local commit on ws-d | 0 | as above |
| GNN with PyG (fan-out too high on the outer hops) | written, local commit on ws-d; CPU smoke run | 0 | as above |
| Vision (data-bound loader, no channels-last) | written, local commit on ws-d; CPU smoke run | 0 (A's reader misses one finding; request filed) | as above |
| **Exports panel (moved from C, Oct 31)**: config, patch, sbatch, CLI command, share link, from the record | on ws-d (4ff522e, 129bfb1); CLI item behind `d.exports_cli` (local); both flags off | 0 | Muse Code diffs need C's blob route; unified diffs need A; the CLI item turns on when the repo is public (Nov 3 to 4) |
| **Add-ons (Contract D)** | | | |
| D1 Expertise dial (glossary, levels, "New to this" view, census baseline) | on main; Oct 30 glossary entries on ws-d; flag off | 0.5 | the worth-checking list waits on C's slot |
| D1 dial re-run on real grids | not started | 1.5 | B's grids in the golden records |
| D2 Share cards and README badge | not started | 5 | C's share route and `workers/` in the deploy |
| D3 Local currency and regional prices | currency on main, flag off; regional rows written | 0 | regional rows deferred (schema field and B's wiring, not a G2 priority) |
| D4 Free-tier presets | on main, flag off | 0.5 | the one-session check against A's real runs; T4/P100/L4 on the grid (B) |
| D5 Notebook intake | on main, check-in wired; desk-limit fix on ws-d; flag off | 0 | nothing |
| D6 Jobs like yours | dropped (review, Oct 27) | 0 | |
| D7 Localisation | deferred beyond the guide's language field | 0 | |
| D8 Hugging Face quick start | not started | 4 | A's `template` job source |
| D9 Energy and carbon | dropped | 0 | |
| D10 Command palette | deferred | 0 | |
| D11 Phone view and dark theme | dark theme done by C; phone view not started | 2.5 | C's share route |
| D12 Alerts outside the app | dropped | 0 | |
| "Done means" for D1 to D5: a README line each and a nightly check that each still renders | written (README draft's "Add-ons"; `docs/audit/scripts/addons.mjs`, 14 of 20 pass, the rest are finding 7) | 0 | C runs it nightly; flags on in production are Chiya's call |
| **Follow-ups and support** | | | |
| Builder kit (competitive review item 10) | on ws-d | 0 | Chiya schedules the sessions (paid check-ins) |
| Stranger-path audit, 12 findings to C | on ws-d | 1 | re-check the top four when C's fixes land |
| Demo storyboard | on ws-d (GiB figure Oct 31) | 0.5 | the record's share link (A) |
| Validation rows D feeds: planted problems caught on the six new zoo scripts; the dial on real grids; the free-tier one-session check | not started | counted above | A's probes and full runs, B's grids |

Order today: the six zoo scripts (agent first), then the exports panel, then the audit re-check and the dial. About 13 hours of D's own work before the waits: the six scripts 6.5, exports 5, README lines and render checks 2.

## Fri Oct 30, evening (builder kit, stranger-path audit, demo storyboard)

**Done** (committed on `ws-d` as Shreya; push requested below with `4ba6c18`). No tokens, GPU or Actions minutes spent: the audit ran on a local dev server with the job API answered by the test.

- `docs/builders/`: a one-page brief, three tasks (open a sample; check in your own script or notebook; act on one finding), an observation sheet (hesitations, stuck points, time to first map, acted on a finding) and a 5-question form. Results are counts ("2 of 3"); stuck points become ranked requests. The public link is a placeholder until the deploy; the sessions' paid check-ins need Chiya's yes first.
- `docs/audit/oct30/`: the stranger path in Chromium (home, check-in on each tab, boarding mid-run and failed, cockpit, checks) at 1440 and 390 px in both themes, 40 states, plus D's panels with their flags on. No overflow, no console errors; axe found day-theme contrast (16 states) and 14 px flight-plan markers. Twelve findings filed to C, ranked by how many strangers each would stop, with screenshots and fixes. D's own fix: the free-GPU band's source links now have a 24 px tap area.
- `docs/demo/storyboard.md`: 15 shots, about 3 minutes, on the Oct 28 LoRA run, with a 45-second cut. Only measured numbers (189 s, $0.042, step 200, 1.68 s), each with its source; every shot's flag dependency and fallback listed.

**Basis for every decision in this commit**

| Decision | Basis | Test that fails if it's wrong |
| --- | --- | --- |
| Builder results as counts, not percentages | Chiya, Oct 30 (three builders) | the tally table has count fields only |
| Kit timings and costs | measured: A's status (upload to record 6 min 45 s; GPU $0.086 and Muse Spark $0.018 measured; sandbox about $0.10 a guess, so the session cost is a lower bound) | review against `docs/status/a.md` |
| Findings ranked by how many strangers they would stop | Chiya, Oct 30 | the ranking states who each finding stops |
| Contrast fixes #2D4855, error panel over glass-hi, #1F5A3D | derived: WCAG 2.x contrast formula on the measured backgrounds (axe's fg and bg colours) | axe colour-contrast in the audit script, re-run after C applies them |
| D's source links get a 24 px tap area | cited: WCAG 2.2, 2.5.8 target size | the audit's small-target check on D's panels (none left) |
| Storyboard numbers | measured: A's status, record `ICDYrfHQ2nBNbv3mlRHHMg`; step 200 from the script | review against the sources; the "Before recording" checklist |
| 21.9 GB kept out of narration | the unit (GB or GiB) isn't stated in A's status | marked "confirm against the record" in the storyboard |

**Pushed** at `895633b` (Chiya's yes, Oct 30).

## Fri Oct 30 (round two merged; Notebook tab checks in)

**Done** (committed on `ws-d` as Shreya; push requested below)

- Merged `main` at round two (408a7c7). One conflict, in `convert.ts`: both sides made `NotebookSource` a type alias; kept main's comment. All 316 web tests pass, as do the typecheck and build; D's Python tests 33 passed; Python suite 944 passed, with the same 10 environment failures as before (A's sandbox, overhead and pip-install tests need a non-root, networked runner).
- Notebook tab checks in through C's hand-off. End to end in Chromium on ws-d itself after the round-two merge (also earlier on ws-d merged with ws-c; `POST /v1/jobs` answered by the test): check-in off before a notebook, on after one, off again after leaving the tab, never on for a notebook with no code; clicking it sends kind `notebook` as `train_1.ipynb` without outputs (1,065 bytes, from a 5.4 KB file), with an idempotency key, and the desk moves to boarding. The Upload tab no longer offers `.ipynb`. With check-in closed the panel says so and the button stays off. No overflow at 390 px, no console errors.
- Notebook limit is C's 5 MB, measured as the desk measures it: the whole source as JSON in UTF-8 bytes. The largest notebook D takes fits the desk and one line more is refused, for ASCII and two-byte text.
- `knob.optimizer` (Optimiser) and `knob.lr_schedule` (Learning rate schedule) added to en.json with glossary entries, in one commit, as C asked.
- Read-only review (a git-free copy, no remote): no blocking defects; fixed its points (non-ASCII boundary case, optimiser memory consistent with the ZeRO entry, the 8-bit argument and bitsandbytes, cosine T after warm-up). Asked C for one shared byte-count function.
- No tokens, GPU or Actions minutes spent.

**Basis for every decision in this commit**

| Decision | Basis | Test that fails if it's wrong |
| --- | --- | --- |
| Notebook limit = the desk's measure (whole source, UTF-8 bytes, 5 MB) | cited: C's `NewFlight.tsx` handedTooBig and `upload.ts` MAX_TOTAL_BYTES; API body cap 6 MiB | `convert.test.ts` boundary tests (ASCII and é; checked failing with a files-only measure and with a character count) |
| `NotebookSource` is a type alias | Chiya, Oct 30; derived: only an alias is assignable to `JobRequest["source"]`, an index-signature type | `tsc -b` on ws-d and on ws-d merged with ws-c |
| Optimiser memory: 8 bytes moments + 4 bytes master copy in mixed precision; 8-bit about 2; SGD momentum 4 | cited: ZeRO (Rajbhandari et al.), bitsandbytes 8-bit optimisers | `glossary.test.ts` (coverage, length, spelling); accuracy by review |
| Cosine schedule formula, t and T after warm-up; HF cosine ends at 0 | cited: SGDR (Loshchilov and Hutter); transformers `get_cosine_schedule_with_warmup` | `glossary.test.ts`; accuracy by review |
| Check-in flow (enable, clear on tab change, refuse no-code) | C's hand-off contract (from-c.md, Oct 28) | Chromium end-to-end run above, on ws-d (a browser test, not in CI) |

**Pushed** at `b4a0801` (Chiya's yes, Oct 30).

## Wed Oct 28, afternoon (merged `main` at contract 0.4.0)

**Done** (committed on `ws-d`; `ea2f578` pushed with Chiya's yes)

- Merged `main` (206c3f3, contract 0.4.0): clean. All web tests pass (305 after the size fix), as do the typecheck and build. D's Python tests: 33 passed, now including the A-side notebook parity test, which runs because A's converter is in `main` (7 of 7). Python suite: 10 failures, the same 10 as on a clean `main` (A's sandbox, overhead and pip-install tests, which need a non-root, networked runner).
- `d.notebook` added to `flags.json`, off (Chiya's yes).
- Dial order frozen for Nov 2 (Chiya): the held-out run is its test.
- C's Upload tab now sends a lone `.ipynb` as kind `notebook`, without D's preview, output stripping or name cleaning; note to C with two options in `docs/requests/from-d.md`.
- Found while reading C's upload: the API refuses a check-in body over 6 MiB, and C holds files to 5 MB as JSON. D5 only checked A's 8 MB and 25 MB caps, so a notebook D5 accepted could be refused by the API. D5 now reads C's `MAX_TOTAL_BYTES` and checks what it sends the same way, with copy that names the limit.

**Waiting**

- Notebook tab check-in: C's source hand-off (not on `main` or `ws-c` yet).
- D2 share cards: C's share route (not on `main` or `ws-c` yet).
- A's surrogate crash stays with A (noted in `from-d.md`).

**Basis for every decision in this commit**

| Decision | Basis | Test that fails if it's wrong |
| --- | --- | --- |
| `d.notebook` off by default | Chiya, Oct 28; flags start off (contract) | `notebook/index.test.tsx` (flag name); flags.json is the shared list |
| Dial order and top 2 frozen until Nov 2 | Chiya, Oct 28 | `dial.zoo.test.ts` keeps the configured policy |
| One notebook path, (a) or (b), left to C and Chiya | measured: C's `precheck` refuses `train (1).ipynb` and A's `fetch` refuses `train (1).py`; outputs count against C's 5 MB | `test_d_notebook_parity.py::test_a_accepts_every_name_d_sends` |
| D5 holds what it sends to C's 5 MB of files as JSON | cited: `app/upload.ts` MAX_TOTAL_BYTES, `routes/jobs.py` MAX_BODY_BYTES 6 MiB | `convert.test.ts` "won't fit a check-in" and "just under"; `NotebookIntake.test.tsx` "says the check-in limit" |

**Pushed** at `6d61463`.

## Wed Oct 28, early (D5 notebook intake; dial counted per record)

**Done** (committed on `ws-d`, push requested below; `901332e` pushed). No tokens or GPU spent, no Modal app: D5 runs in the browser. Not on today's critical path (one zoo script from upload to map): zoo scripts are `.py` and come in as samples or repos; D5 lets notebooks in once C's desk takes an intake source.

- Dial test counts per record (Chiya): each planted problem once, under the record's own goal; safety checks still run for every goal. On A's fixtures: 3 of 3 in the dock on screen for both records, 100% [44, 100]. The order comparison can't decide on 6 problems (a perfect 6 of 6 can't clear today's 3 of 6), and needs at least 12, so 4 zoo records; B's grids alone won't change that. Policy unchanged and frozen for Nov 2.
- D5 notebook intake (`web/src/addons/notebook/`, flag `d.notebook`, key requested): a Notebook tab on the check-in desk. Choose or drop a `.ipynb`; D shows exactly what A will run (script, requirements, cells kept and skipped, shell lines commented), sends it as kind `notebook` without saved outputs, and cleans the file name to A's path rule. Errors say what happened and what to do. Checked in Chromium at 1440 and 390 px: axe (WCAG 2.2 AA) no violations, no overflow, focus ring, no console errors.
- Parity with A's converter, checked on A's code (ws-a ffdd1cf): 4 hand-made notebooks, 400 seeded random ones (every Python space and line break, NaN, odd cells, A's refusals), 11 names, 245 sent notebooks; 7 of 7 A-side tests pass there; the A-side test skips on `ws-d` until A's converter is in `main`.
- Independent review (two passes): fixed a crash on non-object metadata, a race where an older file could replace a newer one, wrong fallback advice (A's repo fetch skips `.ipynb`), NaN notebooks D rejected, unbounded size of what is sent, unpaired surrogates, a tokenizer that took 4.4 s on 20 MB (now 0.36 s), and weak tests.

**Basis for every decision in this commit**

| Decision | Basis | Test that fails if it's wrong |
| --- | --- | --- |
| Count the dial per record, under the record's goal | Chiya, Oct 28 | `dial.zoo.test.ts` report and safety tests (checked failing with the failure rule removed) |
| Default goal min_time when a record has none | derived: the app's default (GridProvider, GuidePanel) | `dial.zoo.test.ts` (both zoo records state a goal today) |
| Comparison needs at least 12 problems to decide | derived: Wilson 95%, 12 of 12 lower bound 76 vs 6 of 12 upper 75 | `dial.zoo.test.ts` prints "can't decide"; `wilson.test.ts` |
| Send kind notebook; A converts | cited: A's upload-format.md | `convert.test.ts` "sends kind notebook" |
| D's preview equals A's conversion | measured: fixtures generated by A's code | `convert.test.ts` parity (fails on drift, checked by breaking line splitting and quote stripping); `test_d_notebook_parity.py` on A's side |
| Leave outputs and other cells' text out of what is sent | derived: A reads only cell type and code source | `convert.test.ts` "sends a notebook that converts to the same script"; A-side `test_a_runs_what_d_sends_as_it_would_the_original` |
| Clean file names | measured: A's fetch rejects `train (1)`, `résumé`, `.hidden` | `convert.test.ts` names; A-side `test_a_accepts_every_name_d_sends` |
| NaN and Infinity read as Python's json does | cited: Python json accepts them; Jupyter writes them | `convert.test.ts` NaN, -NaN and 20 MB timing tests |
| Block notebooks with no code, unpaired surrogates, over A's caps | derived: no code costs a probe for nothing; A can't encode surrogates (A raises UnicodeEncodeError, reported to A); caps cited from A | `convert.test.ts` "can't run" and "stays small and valid" |
| 8 MB per file, 25 MB in all | cited: A's upload-format.md and probes/local.py | `convert.test.ts` "too_big" |
| 100 MB read cap | measured: JSON.parse of 100 MB in Chromium 78 to 88 ms; a 79 MB NaN-filled notebook previews in 1.0 to 1.5 s end to end | none automated (Chromium timing, recorded here) |
| 200 preview lines | guess: enough to recognise a script; replace with what visitors expand, from telemetry after Nov 4 | none (display only) |
| Newest file wins; nothing after unmount | derived: the desk must hold the notebook on screen | `NotebookIntake.test.tsx` race and unmount tests (checked failing with the guard removed) |

Checks: 266 web tests, typecheck, build; D Python tests 26 passed, 1 skipped (A-side parity, until A's converter is in `main`); ruff; no em dashes; gitleaks. Cloud runs: none.

**Pushed** at `ea2f578` (Chiya's yes, Oct 28).

## Tue Oct 27, late night (Chiya's census and margin decisions)

**Done** (committed on `ws-d`, push requested below; `aee591f` pushed; merged `main` at 6a3eca2)

- Unseen scripts stay out of the census entirely (Chiya), recorded in the decision note.
- Order of the five and the top count: today's version stays unless a challenger's Wilson 95% interval sits wholly above it (`beatsClearly` in `_stats/wilson.ts`). `dial.zoo.test.ts` reports all nine versions and fails only on a clear win. On A's fixtures all tie at 9 of 18, so today's version stays. The decision note keeps a table for each run.
- Checks: 221 web tests, typecheck, build; 26 D Python tests, ruff; no em dashes; gitleaks. Python suite 658 passed; the 4 failures are A's `test_probe_sandbox.py` (root container), as before.

**Basis for every decision in this commit**

| Decision | Basis | Test that fails if it's wrong |
| --- | --- | --- |
| Keep the incumbent order unless beaten by a clear margin | Chiya, Oct 27 (tuning and testing on the same two records) | `dial.zoo.test.ts` "keeps the configured policy unless another beats it by a clear margin" |
| Clear margin = challenger's Wilson 95% lower bound above the incumbent's upper bound | Chiya, Oct 27; Wilson interval derived (Wilson 1927, z = 1.959964) | `wilson.test.ts` margin cases (equal, close, far apart, reversed) |
| Unseen scripts out of the census | Chiya, Oct 27 (held-out score must describe the judged version) | `tiers.test.ts` "takes each training lens's baseline from the census ranking" and `test_d_knob_census.py` source checks: the census names only its 5 pinned sources |
| Top count stays 2 | guess; replaced by B's grids only on a clear margin, else tested by the held-out run on Nov 2 | same `dial.zoo.test.ts` test |

**Pushed** at `901332e` (Chiya's yes, Oct 28).

## Tue Oct 27, late (Chiya's four conditions for turning `d.expertise` on)

**Done** (committed on `ws-d`, push requested below). No tokens or GPU spent: the census reads
public GitHub checkouts only.

1. Warnings stay in the dock at "New to this", never dropped, until C's dock lists them and sets `findingsShown` on the knob-view context; only then do they leave the five and come back when opened. Today nothing sets it, so a newcomer sees every warning's knob.
2. The 9-of-9 is labelled for what it is: on screen today, in-sample, 2 records (9 of 9 in the dock on both); and a design check with the list, in-sample, 2 records, counting a list not yet on screen. The held-out run on the unseen scripts (Nov 2) is the result, per record with Wilson intervals, policy frozen beforehand.
3. The order of the five and the top count are a parameter (`POLICY`); `dial.zoo.test.ts` compares three orders with top 1, 2 and 3 and fails unless the configured one shows the planted knobs most often. On A's fixtures all nine tie at 9 of 18, so it decides when B's grids land.
4. Newcomer basics measured: a census of 55 examples (bug zoo 2, TRL 22, transformers 20, PEFT 8, nanoGPT 3) at pinned commits, built by `python -m preflight.addons.knob_census`, reproducible byte for byte. Fine-tuning: batch size 28 of 34, epochs 20, learning rate 20, sequence length 18. Counts and sources in `docs/decisions/expertise-new-level.md`. Pretrain and vision have 3 examples each.
- Fixed on the way: the rl baseline named `max_completion_length`, which isn't a knob id (`max_completion_len`); a test now checks every baseline id has a label.

**Basis for every decision in this commit**

| Decision | Basis | Test that fails if it's wrong |
| --- | --- | --- |
| At most 5 knobs picked at "New to this" | cited: Cowan 2001 (4 ± 1 chunks) | `tiers.test.ts` "picks at most five"; `dial.zoo.test.ts` "at most five" (both dock states) |
| Failure knobs first, never dropped | Chiya and the Oct 27 review | `tiers.test.ts` "never drops a failure's knob"; `dial.zoo.test.ts` failure knobs in the dock for every goal |
| Warning knobs stay until C's list is on screen | Chiya, Oct 27 | `tiers.test.ts` and `knobView.test.ts` gate tests; `dial.zoo.test.ts` on-screen 9 of 9; checked failing with the gate removed |
| A warning's knob shows once opened, and stays | Chiya, Oct 27 | `knobView.test.ts` reveal; `dial.zoo.test.ts` reveal (checked failing with the reveal removed) |
| Goal's top before baseline; top count 2 | guess; replaced by the comparison on B's grids, Wed Oct 28 | `dial.zoo.test.ts` "keeps the configured policy among those that show the planted knobs most often" (ties on fixtures, so it can only fail on real grids) |
| Baseline for finetune, pretrain, rl, vision | measured: census, 55 examples, pinned commits | `tiers.test.ts` "takes each training lens's baseline from the census ranking" (checked failing when unwired); `test_d_knob_census.py` consistency and engine ids |
| What counts in the census (unit = example; A's vocabulary; full configs excluded) | derived: a full config lists every field, so it can't show a choice; A's vocabulary keeps ids the same as the harness | `test_d_knob_census.py` parser, mapping, override-config and ranking tests |
| Baseline for agent, eval, serving | guess; replaced by first-opened knobs from session telemetry after Nov 4 | `tiers.test.ts` "names only knob ids the app knows" (ids only) |
| "Things worth checking" = `warn` findings only | derived: the contract's severities are info, warn, error, critical; errors and critical already hold knob places, info is not a problem | `worthChecking.test.ts` on all three golden records |
| Plain labels for `optimizer` and `lr_schedule` | derived from what the setting does; ux-writing rules (plain, short, no jargon) | `tiers.test.ts` "names only knob ids the app knows" |
| Dial claim on the golden fine-tune (failures, packing, batch size, sequence length) | measured on A's fixture | `knobView.test.ts` "on the golden fine-tune today ..." |

Checks: 220 web tests, typecheck, build; 25 D Python tests, ruff; no em dashes; gitleaks clean. Python suite: 657 passed; 4 of A's `test_probe_sandbox.py` fail here because this container runs as root (same on clean `main`).

**Pushed** at `aee591f` (Chiya's yes, Oct 27).

## Tue Oct 27, night (after Chiya's decision on warnings)

**Done** (pushed at 3e5ba2d)

- "New to this" picks at most 5 knobs by itself: failures first (never dropped), then the goal's top two, then the lens baseline. Warnings no longer take a place; their knob shows once the visitor opens it and stays for the visit. Checked in Chromium on the golden fine-tune: 4 knobs plus "Show 5 more settings"; opening "Pack short examples together" and closing it leaves it in the dock.
- "Things worth checking": `worthChecking.ts` lists the record's warnings with "1 thing / {n} things worth checking". It renders once C offers a spot under the dock in C's findings display (request in `docs/requests/from-d.md`); D doesn't build it in C's code.
- `dial.zoo.test.ts` now asserts, for every goal: failure knobs in the dock, at most 5 knobs, every planted problem visible (dock or list), and a listed warning's knob kept after opening. Fails when the list is emptied or the reveal removed (checked). On A's fixtures: finetune and pretrain each 9 of 9 visible, 100% [70, 100]; in the dock 6 of 9 and 3 of 9.
- Order of the five compared on the fixtures: four failures-first orders tie (the goal ranking is empty for most goals there), so the order is tagged a guess to settle on B's grids.
- `d.freetier` added to `flags.json`, off (approved by Chiya).
- Checks: 209 web tests, typecheck, build, no em dashes.

**Next**

- Rerun the zoo test on B's grids (Wed) and settle the order and `NEW_TOP_SENSITIVE`; score once on the unseen scripts (Mon Nov 2) as a held-out check.
- Wire the worth-checking list to C's spot when it lands. D5 once A documents uploads; D2 once C's share route lands (Thu).

## Tue Oct 27, evening (merged `main` at 7faf676)

- Merged `main`. Requests moved to `docs/requests/from-d.md`, brought up to date: C's three replies (loading fix, knob-view hook, provider and ranking) moved to Done; the dial rerun on B's real grids added as a request to B.
- Checks on the merged tree: 197 web tests, typecheck, build, ruff, no em dashes. Python 647 passed; 4 of A's `test_probe_sandbox.py` tests fail here and 6 fail on a clean `main`, because the sandbox expects to run as a non-root user and this container runs as root. Not D's code.

## Tue Oct 27, afternoon (after the Oct 27 review; merged `ws-c` at a2b21af)

**Done** (branch `ws-d`, committed locally, not pushed)

- Dial (D1), per the review: the detail level moved to the cockpit header; "New to this" now filters C's knob dock through the knob-view hook, ranking by the visitor's current goal (C's `ranked`), always showing the knobs of error and critical findings even off the grid (`eval_no_grad`), and keeping knobs the visitor touched. Checked in Chromium on the golden fine-tune: 9 knobs become 4 plus "Show 5 more settings", both failure knobs among them.
- New dial test (`dial.zoo.test.ts`): ground truth from the zoo scripts' planted problems, matched to golden records by entrypoint, through C's `applyKnobViews`. Asserts every planted-failure knob shows for every goal (it fails if the failure rule is removed). Reports, Wilson 95%, on A's fixture predictions: finetune 6 of 9 (67% [35, 88], packing hidden); pretrain 3 of 9 (33% [12, 65], precision and checkpoint interval hidden). The earlier simulation is withdrawn.
- Glossary fixes from the review: conformal intervals, verify, repetition exposure, seed power analysis and the cost frontier marked planned; measured, predicted and extrapolated per the contract; fp16 with loss scaling on T4 and P100; lower batch with more gradient accumulation; MFU 6N full, 4N with LoRA; fan-out product; the rate-limit formula; Hugging Face eval logit accumulation; ZeRO and FSDP added. 95 terms.
- `whenReady` removed now that C fixed loading.
- D4: Modal's Starter credit added as the free L4 ($30 a month, 24 h per run; Modal pricing and timeouts pages), a monthly-credit check, and fp16 notes on T4 and P100 tiers. The one-session simulation is relabelled calibration: simulated, with the real check named.
- Every D constant tagged cited, derived, simulated or guess; guesses name what replaces them.

**Open with Chiya**

- Settled Oct 27 night: warnings are listed, not shown as knobs; `d.freetier` approved.

**Blocked**

- D5 notebook intake: A hasn't documented what an upload hands the harness.
- D2 share cards: C's share route (Thu).
- Dial coverage on real grids: B's `api.py` and `make_golden.py` (B, Wed).

## Tue Oct 27, midday (rebased on `ws-c`)

**Done** (branch `ws-d`, rebased on `ws-c`)

- Flag keys `d.expertise` and `d.currency` added to `flags.json` with Chiya's approval, both off.
- D1 expertise dial, behind flag `d.expertise` (off by default), through C's existing extension points only:
	- Detail level (New to this, Practitioner, Expert) in the Logbook's pilot card (`logbook-settings` slot). A radio group with arrow keys and one tab stop, kept per browser in localStorage, guarded for blocked storage.
	- Glossary card in every knob's control (knob decoration): a native disclosure worded for the level, a "What to change" line where there is a clear fix, and "More detail" one level deeper without changing the setting. 63 terms; every knob in A's golden records and C's dev record is covered (test).
	- Tier rules ready for C's knob-view hook: `tiers.ts` (`visibleKnobs`, `labelKeyFor`) with plain new-level labels and expert code names for every record knob. New level = lens baseline + top 2 knobs by `sensitivity()` + pinned + guide-highlighted, chosen by Monte Carlo on the sample records (docs/decisions/expertise-new-level.md): the failure fix is visible in 100% of 303 cases, the best knob for any goal in 95.4% of 963, at most five knobs on screen. The simulation runs in CI as a regression test.
- Found and reproduced a bug in C's `registry.ts`: the eager add-on glob is hoisted above the registry's state, so an add-on that registers while loading blanks the page. D works around it (`addons/_shared/whenReady.ts`); fix requested from C.
- Re-filed D's requests against what A, B and C built; five settled by their work.
- Checks: 47 add-on tests (89 in web with C's 42), typecheck, production build; Chromium at 1440 and 390 px, dusk and day, no horizontal overflow, no console errors from D's code.

- D1 finished: every knob A declares (61 labels) now has a glossary card and plain and expert labels; the test covers C's full knob label list, not only the samples.
- D3 local currency, behind `d.currency`: costs in the visitor's currency from the browser region, or picked in the Logbook; converted at the ECB rates of 26 Oct 2026 (30 currencies), dated and linked on screen. Checked in Chromium: rupees from an en-IN browser, euros after switching, kept across reloads. Regional rows for B's table in `python/preflight/addons/regional_prices.py` (Modal 1.15x broad, 1.75x narrow; 15 tests).
- D4 free-tier presets, behind `d.freetier` (key requested): a band under the cockpit's whole run for Colab, Kaggle and Lightning AI's free plan saying one session, N sessions with how often to save, won't fit, or measure on the free GPU. Rule "slow end + 15 min fits the cap" chosen by Monte Carlo (docs/decisions/freetier-one-session.md), kept as a regression test.

**Blocked**

- Fewer knobs and plainer labels at "new" need C's knob-view hook (docs/requests.md).
- D2 share cards need C's share-token read route, R2 store and `workers/` in the deploy (C's Thu work); picked back up when it lands.
- `d.freetier` flag key (Chiya). D4's time check for most records needs T4 or P100 on the `gpu_type` grid (B) or an intake hook (C).

**Next**

- D2 once C's share route lands (Thu). Then D5 notebook intake once A answers what an upload hands the harness.

## Dependency map

| # | Add-on | Needs | From | Status |
|---|---|---|---|---|
| D1 | Expertise dial | Knob-view hook for filtering and labels | C | Setting and glossary cards done; filter and labels wait on the hook |
| D2 | Share cards and badge | Share-token read route, R2 store, `workers/` in the deploy, headline numbers | C, B | Waiting (C, Thu) |
| D3 | Local currency, regional prices | `setCurrency` (done, C), B's dated USD table (done, B) | C, B | Done; regional rows wait on B reading them and `Constraints.region` |
| D4 | Free-tier presets | T4, P100 in B's table (done); `d.freetier` key; T4/P100 on grids or an intake hook | B, C, Chiya | Done; time check widens when those land |
| D5 | Notebook intake | The exact input A's harness takes for an upload | A | Waiting on A's answer |
| D6 | "Jobs like yours" prior | | | Dropped (review, Oct 27) |
| D7 | Localisation | | C | Deferred beyond the guide's language field (review, Oct 27) |
| D8 | Hugging Face quick start | `template` job source in A's harness | A | After G1 |
| D9 | Energy and carbon | | | Dropped (review, Oct 27) |
| D10 | Command palette | | C | Deferred (review, Oct 27) |
| D11 | Phone view | Read-only share route | C | Dark set done by C; phone view waits on the route |
| D12 | Alerts outside the app | | | Dropped (review, Oct 27) |
