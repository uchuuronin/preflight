# Status: D (usability add-ons)

## Cloud runs

Every run through `remote-run.yml` or `amd-run.yml`: date, GPU, minutes, dollars.

| Date | GPU | Minutes | Dollars | What |
|---|---|---|---|---|
| | | | | None yet. D's checks run locally in CI; no GPU needed so far. |

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
| "Done means" for D1 to D5: a README line each and a nightly check that each still renders | not started | 2 | flags on in production are Chiya's call |
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
