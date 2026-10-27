# Status: C (workspace, guide, fix loop, control plane)

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
