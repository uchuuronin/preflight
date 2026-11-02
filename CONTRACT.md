# Preflight contract v0.6.3

Four agents build Preflight in parallel against this contract. The code in
`python/preflight/schema/` is the source of truth; this file is the map. Every
agent rereads this file and the Contracts tab at the start of every session.

Frozen at v0.1 on **Tue Oct 27, noon ET**. After that, changes are additive
only (new optional fields, new endpoints, new tools) and bump the minor
version. A breaking change needs Chiya's sign-off and a note in every agent's
chat. Ask for changes in your own `docs/requests/from-<you>.md` with a `TODO(contract)` line.

## Generated files

| File | From | Regenerate |
| --- | --- | --- |
| `schema/flight-record.schema.json`, `job-event`, `job-request`, `script-map` | Pydantic models | `uv run python -m preflight.schema.generate` |
| `schema/contract.schema.json` | every model the TypeScript side imports | same |
| `schema/ts/contract.ts` | `contract.schema.json` via json-schema-to-typescript | same |
| `schema/examples/*.json` | `scripts/make_golden.py` | `uv run python scripts/make_golden.py` |

CI fails when any generated file is stale. `schema/ts/encoding.ts` is
hand-written and kept equal to the Python encoder by a parity test.

## Ownership

| Path | Owner |
| --- | --- |
| `python/preflight/schema/`, `schema/`, `CONTRACT.md`, `flags.json` keys | Shared; changes through Chiya |
| `python/preflight/{probes,recorder,harness,runner,cli}/` | A |
| `python/preflight/{engine,corpus,campaign}/` | B |
| `api/`, `web/src/{workspace,guide,fix}/`, `.github/` | C (A set up `.github/` and hands it over Tue) |
| `web/src/addons/`, `web/src/i18n/` (non-English), `api/addons/`, `python/preflight/addons/`, `workers/`, `web/src/theme/` (dark set) | D |
| `zoo/` | A writes and runs for ground truth; B validates against it |

## The flight record

One record per job, under 1 MB gzipped (`dump_record` enforces it). Raw traces,
logs and diffs go to the object store as `blob://` keys, never inline.
Unknown fields are a validation error everywhere (`extra="forbid"`), so typos
fail loudly. Read stored records with `load_record()`, which runs `upgrade()`.

| Field | Holds | Written by |
| --- | --- | --- |
| `schema_version`, `record_id`, `share_token`, `parent_id`, `created_at` | semver; ULID (internal only); random read-link token (ULIDs leak creation time); derived-from record | A |
| `job` | lens, source, framework, `script_map`, objective, constraints | A |
| `env` | backend (cuda, rocm, api, cpu, mps), provider, GPU, libraries, seed, image | A |
| `graph` | job-map nodes and edges; `parent` for hierarchy, `repeat` for ×N, `metrics` for what animates; `skeleton` until measured | A builds, C reads |
| `knobs` | id, `label_key`, unit, kind, domain, current, `probed_range`, gridded, `needs_code`, `patch_template` | A declares, B fills `probed_range` |
| `probes` | per probe: knob values, steps, status, memory by term, step time by phase, throughput and MFU, stability series, data stats, RL signals, power, counts in `extras`, the kernel table (current-settings probe), cost, cache key | A |
| `agent_sample` | per task: turns, tokens (input, output, reasoning, cached, cache creation), tool calls, errors, latency, success, context by turn | A |
| `predictions` | `targets`, `estimates` at current knobs, `grids` | B |
| `models` | model cards, pinned by id from every estimate and grid | B |
| `findings` | severity, stable `code`, map node, evidence, source, fix template, knob, citations | A and B write, C shows |
| `patches` | template or open-ended, diff blob, verify result, session log | C |
| `campaign` | cells, lanes, calendar, cut list, completion quantiles | B |
| `provenance` | every model call: provider, model, purpose, tokens, cost, request id, time | whoever calls a model |
| `costs` | What Preflight itself spent making this record (probes, verify, model calls); never part of the run cost | A (GPU) and C (LLM) |
| `cost_basis` | Where the run cost's rates come from: price-table version, as-of date, provider, region, billing (on demand, spot, own cluster), what the total includes; USD | B |
| `job_cost` | The run cost: lines (GPU-hours, tokens, storage) as quantity × rate with intervals, the total (lows and highs added line by line), GPU-hours and market equivalent for an own cluster, and money at risk shown beside the total, never in it | B |

The record write token is never stored in the record. Golden records for the
fine-tune, pretrain and agent lenses live in `schema/examples/`; their
predictions are fixtures (`golden-fixture@0`) for UI work until B's engine
replaces them.

### Grids

Row-major over `axes` (at most 4), at most 16,384 cells inline; larger grids
set `sidecar` to a blob key. `point`, `lo`, `hi` are base64 little-endian
float16 of **value / scale**, where `scale` is a power of two
(`encoding.choose_scale`; above 1 for large values such as bytes, below 1 for
tiny ones); readers multiply back (`Grid.values()` in Python,
`decodeF16(data, n, grid.scale)` in TypeScript). Float16 tops out at 65,504,
so bytes and long run times need it. `abstain` is a base64 LSB-first bit
mask. Non-finite cells are rejected. `fixed` holds the values every non-axis
knob is held at.

## Prediction API (B)

```python
# python/preflight/engine/api.py: pure, no I/O, imports only numpy, scipy, scikit-learn, lightgbm
def predict(record: FlightRecord, knobs: dict[str, KnobValue]) -> Prediction: ...
def build_grids(record: FlightRecord, knob_ids: list[str]) -> list[Grid]: ...
def sensitivity(record: FlightRecord, objective: Objective) -> list[KnobImpact]: ...
```

`Prediction` is `{target_id: Estimate}`. An `Estimate` has `point`, `lo`,
`hi`, `unit`, `status` (measured, predicted, extrapolated, abstain),
`model_id`, `source` (grid, exact), `reason` (required when abstaining) and
`level` (nominal coverage, default 0.9). The interval must bracket the point.
The TypeScript `PredictionProvider` (GridProvider, PyodideProvider) exposes
the same three calls; the UI never asks a model directly.

Target ids are dotted: `memory.peak`, `time.total`, `cost.total`, `risk.oom`,
and lens-specific ones. `Target.readout` says which of the four readouts shows it.

## Script map (A)

`ScriptMap` is Muse Spark's structured reading of a script (or an adapter's,
with `produced_by: "adapter"`): lens, framework, entrypoint, model, data,
training, loop location, knobs with code locations, blocks for the map
skeleton, agent and RL details, and risks noticed while reading. The harness
generator turns it into a probe harness, the skeleton graph and declared knobs.

## Job events (A emits, C relays)

`GET /v1/jobs/{id}/events` streams `JobEvent`s as server-sent events with
`seq` as the SSE id. Stages in order: queued, admitted, fetching, mapping,
skeleton, harnessing, probing, triage, predicting, done (or failed,
cancelled). `record_rev` bumps when the stored record changed. `POST /v1/jobs`
takes a `JobRequest`.

## Guide tools, HTTP API, UI extension points

As in the Contracts tab, unchanged: pinned knobs are removed from `set_knob`'s
allowed ids every turn; `run_probe`, `propose_patch` and `run_verify` run on
the server and need the visitor's click. REST under `/v1`, one error envelope
`{error: {code, message, request_id, retryable}}`, 429 with Retry-After, and
an `Idempotency-Key` on every write that spends money. UI strings through
`t()`, colours through theme tokens, numbers through `fmt()`, features through
`web/src/registry.ts`, flags from `flags.json`, server add-ons in `api/addons/`.

## Rules

- One branch per agent (`ws-a` to `ws-d`); only Chiya merges into main, nightly.
- No push without Chiya's explicit consent in that chat, every time. Run gitleaks first.
- Touch only the paths you own. Need something else? `TODO(contract)` in your own `docs/requests/from-<you>.md`; never edit another agent's file.
- Every feature behind a flag in `flags.json`, off by default.
- Nightly: update `docs/status/<agent>.md` with done, blocked, next.
- Secrets only in Modal secrets, Cloudflare env and GitHub Actions secrets. Agents never hold keys: cloud runs go through `remote-run.yml` and `amd-run.yml` (docs/setup/cloud.md). Canonical names: `META_API_KEY`, `META_API_BASE`, `BASETEN_API_KEY`, `BASETEN_BASE_URL`, `BASETEN_MODEL`; Modal secrets `preflight-meta` and `preflight-baseten`, never mounted in probe containers.
- Start every session with `git merge origin/main` (merge, never rebase once pushed).
- Python 3.11, uv, ruff; TypeScript strict; tabs in both; British spelling and no em dashes in user-facing copy.

## Changelog

- 0.6.3 (Nov 2, additive; B's, after the validation freeze, from Chiya's Nov 2 item 3): `Grid.independent_of`
  (optional, absent by default): the knobs a grid's target doesn't depend on (the learning rate and its peers for
  memory, time and cost), so a reader answers at any value of them instead of holding them at `fixed`. Emitted only
  with `engine.carry_free_knobs` on (off). A reader that doesn't know it can ignore it; 0.6.2 records load unchanged.
- 0.6.2 (Nov 1, additive; C's request, approved by Chiya): `JobRequest.private` (default false). The
  visitor chose private mode: with `harness.glimmer_private` on, triage runs on Muse Glimmer on
  Preflight's own GPU (B's Modal host) instead of Muse Spark Open on Baseten. 0.6.1 records load unchanged.
- 0.6.1 (Nov 1, additive; A's and B's requests, approved by Chiya): `StepSample.reserved_bytes` (the
  allocator's reserved memory read at the same instant as `device_used_bytes`, so their gap is the
  context plus other processes) and `AgentSample.concurrency` (tasks run at once; when `ended` is
  "stalled", at most this many tasks with no stop reason can be stuck). 0.6.0 records load unchanged.
- 0.6.0 (Nov 1, additive; B's, C's and A's requests, approved by Chiya): a shared record draws the same
  campaign calendar as B's live plan. `Campaign` gains `start` (when the campaign clock starts; `start_s`
  and `finish_s` count from it), `p_by_deadline`, `checkpoint_interval` (`Estimate`, beside the
  `checkpoint_interval_s` point) and `lane_slots`; `CampaignCell` gains `slot` (0-based, within its
  lane), `reason` (why a run is cut or undated) and `gpu_hours_estimate` and `usd_estimate`
  (`Estimate`, beside the `gpu_hours` and `usd` points). `Estimate.rate` (`EstimateRate`: `quantity`,
  `usd_per_unit`, `unit`, `price_row`) splits a cost estimate into quantity × rate, so the browser
  re-prices it at the visitor's own rate. `AgentTask.stop_reason` gains "cut" (stopped with the whole
  run, never a runaway), and `AgentSample.ended` says how the sampled run ended (finished, stalled,
  cancelled, timeout). 0.5.0 records load unchanged.
- 0.5.0 (Oct 31, additive): `Campaign` gains the planner's intervals (B's request): `gpu_hours`, `usd`
  and `finish_s` as `Estimate` for the runs that go ahead, `grid_gpu_hours` and `grid_usd` for the whole
  grid before budget cuts, `quota` (`CampaignQuota`), `asha` (`CampaignAsha` with `AshaRung`s) and
  `notes`; `CampaignCell` gains `time` and `start_s` (`Estimate`, for C's calendar). `JobRequest.patch`
  (a unified diff, at most 256 KiB, probe and verify only) carries C's fix-loop patches to A's runner.
  0.4.0 records load unchanged.
- Rules (Oct 30, Chiya): everything about a job (upload, record, cached probe results with their error
  text, job index) is deleted 30 days after the job; a repo's code is deleted as soon as it's read.
  Every push needs Chiya's yes in chat, batch or not; reviewers are read-only.
- 0.4.0 (Oct 28, additive; A's fields, asked for by C and approved through the PM): `JobRequest.mode` (plan, probe, verify, sample_more; default plan), `knobs`,
  `tasks` and `purpose`, with `Source.kind` "record" and `Source.record_ref` (a share token or
  `sample:<id>`); a job on a record writes a new record whose `parent_id` is it. `Knob.affects`
  (map node ids, checked against the graph) and `Knob.changes_results`. `ModelCard.fixture`.
  `JobEvent.error_code` (Probe.error_code's families plus `submission.*`, `internal`,
  `internal.worker_lost`, `unsupported.agent_sampling`). `Source.commit` is the full hash the code
  was read at. Golden records: knobs carry `affects` and `changes_results`, model cards
  `fixture: true`, and sample URIs name the real zoo directories. 0.3.0 records load unchanged.
- 0.3.0 (Oct 28, additive): `JobCost.at_risk_gpu_hours`, so an own cluster shows money at risk in GPU-hours
  first (B's request). Golden records use the L4 capacity PyTorch reports on Modal: 23,689,363,456 bytes
  (22.06 GiB), measured Oct 28 by the PM smoke test.
- Rule (Oct 28): a generated file is regenerated by whoever changes its source; nobody edits one by hand.
- 0.2.0 (Oct 27, after the freeze, additive): `cost_basis` and `job_cost` (the run cost agreed Oct 26,
  gap 21); `costs` keeps its name and now says it holds Preflight's own spend only (renaming it would
  break every plane). The golden fine-tune carries a fixture run cost with its first-eval OOM as money
  at risk. 0.1.0 records load unchanged through `upgrade()`.
- 0.1.0 (Oct 26–27): first contract. Pre-freeze additions over the Contracts tab, all A's:
  `Grid.scale` (float16 overflow), `Estimate.level`, `Target.readout`, `Finding.code` and
  `Finding.knob_id`, `Graph.skeleton`, `JobEvent` and `JobRequest`, `env.backend` gains `cpu`
  and `mps` for local probes, probe status `cached` for probe-cache hits.
  Also: `Probe.error_code` (stable failure codes), `Probe.eval` (`EvalStats`), `Probe.checkpoint`
  (`CheckpointStats`), `Probe.generation` (`GenerationStats`); `Grid.scale` may be below 1 for very
  small values; `load_record` refuses records over 20 MB once decompressed.
  For B's engine (Oct 27, Chiya approved A adding them): `Probe.kernels` (`KernelTable` of
  `KernelStat` rows: name, `kernel_class`, calls, device_s, flops, bytes, dtype, per step);
  `ModelInfo` gains `head_dim`, `n_experts`, `experts_per_token`, `max_positions`, `gated_mlp`,
  `tie_embeddings`, `norm`, `attn_impl` and `shape_source`, and `quantization` uses
  `engine.memory`'s format keys; `Probe.extras` carries `params_total`, `params_trainable`,
  `tokens_per_step`, `samples_per_step`, `micro_steps_per_step`; GPU names in
  `JobRequest.gpu` and `Constraints.gpu_types` are 1–64 characters; `JobRequest.constraints` is
  a typed `Constraints`.
- 0.1.0, after the freeze (Oct 27 review, additive): `Probe.step_log` (`StepSample` per optimizer
  step), `EvalStats.loader_batches` and `memory_by_batch`, `JobEvent.share_token`; descriptions
  pin `AgentTask.output_tokens` as including reasoning tokens.
