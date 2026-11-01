# Status: D (usability add-ons)

## Cloud runs

Every run through `remote-run.yml` or `amd-run.yml`: date, GPU, minutes, dollars.

| Date | GPU | Minutes | Dollars | What |
|---|---|---|---|---|
| | | | | None yet. D's checks run locally in CI; no GPU needed so far. |

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
