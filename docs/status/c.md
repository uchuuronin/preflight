# Status: C (workspace, guide, fix loop, control plane)

## Tue Oct 27, late (decisions applied, cockpit as a dashboard)

**Done** (branch `ws-c`, merged `origin/main` first)

- Cockpit reads as a work dashboard, after D's glass prototype: one status line above the
  readouts from a template over the readouts and their intervals (review item 8); the stage
  switches Map and Fit (with Fit the orbit steps aside into the side column and the fit map
  follows every knob move); "Measured" is a capsule opening a drawer of probe chips; every
  explanation sits behind an "i". Checked at 1440 px and 390 px, dusk and day, no sideways
  scroll; the stranger test now has 18 checks, all passing.
- `c.fit_map` in flags.json, off by default.
- Verify checks both sides (review item 3): target improved; effective batch, tokens seen,
  LR schedule, throughput floor and fixed-seed loss unchanged; one stdlib module used by the
  API and by `.github/actions/verify`, which CI runs on example records.
- `/v1/jobs` pins GitHub links to a full 40-character commit at check-in; check-in takes
  GitHub links only.
- CI fails on GPL, AGPL or SSPL dependencies in root, API and web (gap 22): none today.
- Muse Code in a sandbox: install, auth and headless run written up with three options for
  keeping the key from visitor code (docs/decisions/muse-code-sandbox.md); smoke test waits
  for a yes.
- CI green again on f1ac48d: the secrets job had failed since 3d03ec8 on a test
  Idempotency-Key value gitleaks read as an API key. Renamed the value and listed the old
  commit's fingerprint in `.gitleaksignore`; full-history and working-tree scans are clean.
- Merged `main` (its `.gitleaksignore` kept). The secrets scan now reads only the history of
  the commit under test (`--log-opts=HEAD`), so one branch's false alarm can't turn others red.
- `guide_tour.py` ends in Modal before the workflow does: function timeout 20 min, startup
  timeout 2 min, no retries, and the runner cancels the call at a deadline under the
  workflow's `timeout` input. A crash on start now fails in 2 minutes, not 30.
- Meta spend measured on the Usage page: $3.01 for 659 requests against $2.91 estimated (3%
  low); recorded in `guide-limits.md`, caps unchanged.

**Cloud runs:** none since the evening's table (new rule: ask before anything that spends).

**Waiting**

- First deploy and the ledger's hourly cost: after the Cloudflare Pages project exists. Modal
  list prices put the ledger container (0.125 core, 128 MiB, always on) near $0.007 an hour.
- Retention notice: when the 30-day expiry rule on the record bucket exists.
- `/v1/jobs` live: A's job service in `main` and `runner.queue` on.
- Muse Code: Chiya's pick of the three key options, then a yes for the smoke test.

**Next**

- Coverage page (review item 2) once real runs exist to fill it; autopilot scored over the bug
  zoo with Wilson intervals (item 7, Nov 3); deploy for the Saturday builders (item 10).

## Tue Oct 27, night (after the merge into main)

**Done** (branch `ws-c`, merged `origin/main` 7faf676 first)

- Muse Spark verified for real: `muse-spark-1.3` on `https://api.meta.ai/v1`, Responses format,
  tool calls and usage fields as `muse_spark.py` expects. Five rounds of whole tours through the
  real guide route (`api/scripts/guide_tour.py`, 76 tours, 595 rounds) set the timeout (12 s),
  output cap (2,000 tokens), per-run and per-visitor caps, and found and fixed three failure
  modes: re-ranking loops, highlight loops while a move waits, and announcing a move without
  making it (a chosen instruction plus a one-time server reminder). Last run: the guide moved on
  9 of 10 tours, no loops, $0.034 per tour. docs/decisions/guide-limits.md.
- `deploy.yml` runs only by hand, and every deploy is gated on the stranger test.
- Stranger test (`web/e2e/stranger.mjs`, 16 checks: sample to map, status word on every dial,
  knob p95 under 400 ms at 4x CPU slowdown, the guide's first move, data-use notice before
  check-in, phone width). Passes locally and in CI (16/16).
- 4x CPU slowdown: knob to readout p95 135 to 175 ms (target 400). The slowest interactions fell
  from about 1 s to 0.5 s at p98 once map nodes stopped being re-measured on every knob change
  and the decorative loops pause while a hand is on a control. A transition-based variant was
  tried and dropped (A/B, 5 runs each: no gain, knob latency 149 to 259 ms). The rest is
  software compositing of the glass blur in headless Chromium, which a GPU does off the main
  thread.
- Data-use notice at check-in, every line switched by the flag for that behaviour; the old copy
  promised deletion after 24 hours and a private mode, neither of which exists.
- Fit map ("Where it fits", flag `c.fit_map`, key requested from Chiya): two knobs, status per
  square as the dials show it, boundary placed from interpolated memory. Between grid points the
  provider now derives P(OOM) from memory with B's formula.
- `/v1/jobs` (POST, GET, events as SSE, cancel) on A's JobStoreService, off until `runner.queue`.

**Cloud runs** (all through `remote-run.yml`; no GPU used)

| Date | Run | Where | Minutes | USD |
| --- | --- | --- | --- | --- |
| Oct 28 00:56 UTC | c-smoke-models | GitHub runner; 43 Meta + 20 Baseten tiny calls | 1 | < 0.01 |
| Oct 28 00:56 | c-guide-tour | Modal CPU, crashed on start until the timeout (path bug) | 30 | Modal CPU only, no model calls |
| Oct 28 01:27 | c-guide-tour | Modal 1 CPU core + Meta, 91 rounds | 3 | 0.40 Meta |
| Oct 28 01:34 | c-guide-tour-2 | Modal 1 CPU core + Meta, 48 rounds | 3 | 0.15 Meta |
| Oct 28 01:38 | c-guide-prompts | Modal 1 CPU core + Meta, 264 rounds | 13 | 1.37 Meta |
| Oct 28 01:52 | c-guide-tour-4 | Modal 1 CPU core + Meta, 113 rounds | 6 | 0.65 Meta |
| Oct 28 01:59 | c-guide-tour-5 | Modal 1 CPU core + Meta, 79 rounds | 6 | 0.34 Meta |
| Oct 28 01:34 | guide-smoke (push) | GitHub runner + Meta, 6 calls | 1 | 0.02 Meta |

Meta dollars per run use the guide's price table ($1.25 / $4.25 / $0.15 per million in, out,
cached). Measured total on Meta's Usage page: $3.01 for 659 requests, 3% above the table's sum. Modal CPU charges are on Modal's
usage page; nothing here used a GPU.

**Blocked**

- The ledger container's hourly cost: measured on the first deploy, which waits for the
  Cloudflare Pages project.
- Retention: the data-use notice says nothing is deleted automatically, because nothing is.
  Needs a decision (asked in chat) and then deletion built before the notice promises it.

**Next**

- Remaining review items: none of C's are open except the ledger's hourly cost (first deploy).
- Watch: a guide reply leaked a line about its own tools once in 79 rounds.

## Tue Oct 27, evening (review: security, live tour, honest readouts)

**Done** (branch `ws-c`, pushed a72a724, a138e3f, 96cce42)

- Same-site API: a Cloudflare Pages Function (`web/functions/v1/[[path]].ts`) forwards `/v1/*`
  to Modal with a shared edge secret and the visitor's IP; with `PREFLIGHT_REQUIRE_EDGE=1` the
  API refuses anything that didn't come through it. The web app calls `/v1` on its own origin.
- Signed sessions: `__Host-pf_session=v1.<id>.<HMAC-SHA256>`, keys from
  `PREFLIGHT_SESSION_KEYS` (first signs, all verify, rotation re-signs); production refuses to
  start without a key.
- Spend ledger: before every model call the API reserves the worst case (max output tokens x
  Meta's price) atomically, against caps per day, per run, per visitor and per IP plus an IP
  token bucket, then settles to the real usage or releases on failure. Production holds it in
  one always-on Modal container (`SpendLedger`, snapshot in a Modal Dict), so four API
  containers can't overspend between them.
- `/v1/guide/turns` takes only typed input (record id, setting, signed conversation state, a
  500-character message, a decision, typed tool results, approvals); tool definitions are built
  on the server from the record, and `set_knob` disappears while a move waits for the visitor.
  Any number in the guide's reply that this turn's tool results don't support withholds the
  reply.
- Autonomous live tour: the browser runs sensitivity, highlight, predict and a guarded
  `set_knob` (refuses pinned knobs, values outside the knob, settings that don't fit or that
  memory can't vouch for, and a second move in one round), then stops for the visitor's Apply,
  Revert or Pin and re-plans. Approving a spending tool runs it on the server and the guide hears
  the real result ("measuring isn't open yet" until A's job store service exists).
- Honest readouts: every dial says measured, predicted, extrapolated or not trusted, and the
  risk dial always says fits, tight or likely to fail. "Measured" needs a probe at exactly these
  settings that observed that target itself (never time.total, cost or risk). Dials keep one
  fixed scale per record (log over the grid's range; linear to a budget or cap), chosen by Monte
  Carlo: docs/decisions/dial-scale.md. The memory stack and ring draw each part at its own size
  and hatch any disagreement with the peak as untrusted, naming terms that abstain.
- Muse Spark smoke test (`api/scripts/muse_spark_smoke.py`, workflow `guide-smoke`): one plain
  call, a guide round trip and a fallback probe, reported as annotations.
- Checks: 77 web tests, 33 API tests, ruff, typecheck, build; knob change 83 ms p95.

**Blocked**

- Real Muse Spark call: needs the repository secrets `META_API_KEY` and `META_API_BASE`.
  Timeout and caps stay tagged guesses until it reports latency and tokens.
- Guide-approved probes: A's job store service and the `JobRequest` fields in docs/requests.md.

**Next**

- Fit map with the OOM frontier, Playwright stranger test gating the nightly deploy, data-use
  notice, `/v1/jobs` on A's job store, INP at 4x CPU throttle.

## Tue Oct 27, afternoon (D's requests, sample_more)

**Done** (branch `ws-c`, merged `ws-b` 39a7904)

- D's blank-page bug fixed at the root: add-ons load from `web/src/loadAddons.ts` after the
  registry exists, each as its own import (one that throws is logged and skipped; the app renders
  within 1.5 s regardless), and every add-on component renders inside an error boundary.
  Checked in Chromium with add-ons that register while loading, throw while loading and throw
  while rendering: the page stays up each time.
- Knob-view hook to D's spec (`registerKnobView`, filter + labelKey), with the dock never hiding
  pinned, guided, open or moved knobs and a "show all" capsule; the guide's moving knob stays in
  `highlight`. Decorations get `provider` and `ranked`; `useCockpit()` gives add-ons a read-only
  cockpit. D's expertise add-on and its 47 tests pass against it.
- `sample_more` guide tool (contract): agent runs only, refuses unless the cost range straddles
  the budget, waits for the click; the rule tour offers it first on the golden agent sample
  ($46 to $159 against a $150 budget).
- Agent runs: a tokens dial and B's runaway-margin sentence on the cost dial.
- Checks: 62 web tests, typecheck, build; knob change 97 ms p95.


## Tue Oct 27, midday (merged A's b59334b)

**Done** (branch `ws-c`, local merge of `ws-a`, not pushed yet)

- A's contract update (kernel tables, model shape, typed constraints, bounded GPU names) passes
  the web typecheck and tests unchanged.
- Probe errors: copy for all 20 of A's error-code families with a hint each, falling back to the
  family and then a generic line (`web/src/lib/errors.ts`); shown on boarding and in a new
  "What was measured" panel listing every probe with its settings, peak memory, GPU time and
  cost, and a "Fly at these settings" button.
- Run plan counts steps when the script gives none: a `max_steps` knob first (it moves with the
  knob), then the script, then epochs x examples / samples per step. The golden fine-tune now
  shows 3,235 steps with its eval and save marks.
- Labels for all 57 of A's knob ids and their values; the guide maps the new knobs (quantisation,
  attention kernel, LoRA alpha and dropout, optimiser, RL knobs) to map nodes and knows which
  change results.
- Checks: 48 web tests, 12 API tests, 523 Python tests, typecheck, ruff, build.

**Next**: `POST /v1/jobs` and the event stream on A's job store (QueueFull -> 429 with
Retry-After, StoreUnavailable -> 503), once where the store runs is settled.


## Tue Oct 27, early (rebased on B)

**Done** (branch `ws-c`, rebased on `ws-b`)

- Speaks B's vocabulary: all twelve `memory.<term>` targets grouped for the ring and the stack
  (older `memory.term.<t>` ids still read), B's fits / tight / fails cut-offs (1% and 80%)
  everywhere, the `gpu_type` knob, and GPU memory from `engine/gpus.py` through a generated
  `gpus.json` (47 GPUs, aliases and Modal names; CI checks it is current).
- Guide move rule chosen by Monte Carlo (docs/decisions/guide-moves.md): it applies only
  settings that fit, and offers "measure it first" when a tight one would save 10% or more.
  Same crash rate as fits-only (0.1%), about 5% extra cost instead of 42%.
- Failure signals are red at dusk and maroon by day; memory colours follow D's mockup, plus
  a lavender sixth slot for generation memory (KV cache) on the RL lens.
- CI: an `api` job (FastAPI tests) and the GPU table check.
- Checks: 42 web tests, 12 API tests, 468 Python tests, typecheck, ruff, build. Knob change to
  painted frame 82 ms p95 during the tour.


## Mon Oct 26, late (session 2)

**Done** (branch `ws-c`, rebased on `ws-a`, not pushed)

- Web app rebuilt on D's "Preflight glass workspace" design: one sky that climbs through five
  screens (Home scroll story, New flight check-in desk and departures board, Boarding, Cockpit,
  Logbook), glass nav capsule, dusk and day token sets, hash routes, reduced motion.
- Cockpit on A's real contract (generated `contract.ts`, `encoding.ts`, golden records):
  dials with intervals (time and cost press to set the aim), the orbit job map inside React
  Flow (phase arcs sized by step share, overlap-free nodes, eval diamond and save hexagon on
  tethers, memory ring where a full circle is the GPU and the overflow runs on a dashed ring),
  node detail with "measured at ..." provenance, knob dock with fits / tight / fails / not
  trusted ticks under each slider, flight plan with the predicted stop flag and playhead, and
  the memory stack.
- GridProvider reads A's grids: held-at-current rule for off-axis knobs (abstains with
  `off_grid`), log-axis interpolation, estimates win at the record's own settings.
- Dev record `web/src/workspace/fixtures/dev-finetune.record.json`, built and validated by
  `web/scripts/make_dev_record.py` against the Pydantic schema (labelled not measured).
- Guide: real tool calls appear in a visible trace; tour steps are Apply / Revert / Pin with
  "Why?"; "Measure it" queues an approval and never spends.
- Registry gains app-level slots for D: `intake`, `logbook-settings`, `nav-end`, `home-beats`,
  plus `cockpit-head` for the cockpit.
- Checks: 28 web tests, 12 API tests, typecheck, ruff, production build, screenshots of every
  screen in dusk and day at desktop and phone width. Knob change to painted frame: 116 ms p95
  in headless Chromium during the guided tour (target 400 ms).

**Blocked**

- `POST /v1/jobs` and the job event stream need A's runner queue; the check-in desk shows an
  honest "not open yet" message meanwhile.
- gitleaks can't be installed in the build sandbox (GitHub and the Go proxy are refused); a
  pattern scan found nothing. Chiya, please run gitleaks before the push.

**Next**

- Jobs endpoint and event relay, R2 record store, cached samples, nightly Playwright test.
- Cost fields when A adds them: run cost breakdown, "+ $X at risk", Preflight's own cost.
- Agent and pretrain tours and orbit tweaks (agent turns as the loop, context fill ring).


## Mon Oct 26, night

**Done** (branch `ws-c`, not pushed)

- Web app: Vite, React 19, TypeScript strict, tabs. Extension points in place for D:
  `t()` with auto-loaded `web/src/i18n/<lang>.json`; theme tokens with `registerTokenSet`
  (light set, palettes checked with the dataviz validator); `fmt()` with `setCurrency`;
  `web/src/registry.ts` for panels, readouts, knob decorations and map overlays;
  `web/src/addons/*/index.ts` imported automatically; `flags.json`.
- `PredictionProvider` with `GridProvider`: float16 grids, multilinear interpolation,
  measured / predicted / extrapolated / abstain, and `sensitivity()` that never trades into
  an OOM and ranks OOM fixes first.
- Workspace shell on a hand-made fine-tune record (labelled as not measured): four readouts
  with intervals and delta badges; knob rail with pins, constraints and the OOM frontier and
  abstention drawn on each control; React Flow job map with the fixed encodings (edge speed =
  throughput, outline = share of step time, dashed = predicted, magenta = guide focus) inside a
  GPU container; memory stack; node drawer; flight plan with evals, checkpoints, predicted
  failure and a draggable playhead.
- Guide: tool schema rebuilt per turn with pinned knobs removed from `set_knob`; browser tool
  executor; money-spending tools only queue an approval; rule-based tour that recomputes the
  next step after every Apply, Revert or Pin, names the pin in the way, and speaks up when the
  person crosses the memory cliff. Live Muse Spark loop wired behind `c.guide.live`.
- API skeleton (FastAPI, Modal entry point not deployed): error envelope, request ids,
  anonymous session cookie, `Idempotency-Key` helpers, `/livez`, `/readyz`, `/v1/status`,
  `GET /v1/records/{share_token}`, `POST /v1/guide/turns` with per-visitor turn caps,
  per-run and daily LLM caps and provenance, `api/addons/` auto-mount.
- Checks: 22 web tests, 12 API tests, typecheck, ruff, production build. Knob change to
  painted readouts measured at 60 to 71 ms p95 in headless Chromium (target 400 ms).

**Blocked**

- Repo was empty: no schema, golden records, CI or deploy from A's chat yet. C builds against
  provisional types; see `docs/requests.md`.
- Muse Spark wire format unverified: dev.meta.ai isn't reachable from the build sandbox, so
  the guide proxy sends an OpenAI-style Responses request from `api/preflight_api/guide/muse_spark.py`
  and `c.guide.live` stays off until one real call succeeds. Need: `META_API_BASE`.
- Modal access pending, so the API isn't deployed.

**Next (toward G1, Thu Oct 29)**

- Swap provisional types for generated ones; render A's golden records; auto-layout for graphs
  without `pos`.
- Fit map with the OOM frontier; interval bands and HOPs toggle; snapshots with ghost overlay.
- Record store on R2, cached samples, data-use notice on upload, nightly Playwright stranger test.
- Guide tours for pretraining and agent lenses; streaming guide turns.

**Contract requests:** 10, all additive, in `docs/requests.md`.
