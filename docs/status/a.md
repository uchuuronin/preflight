# Status: A (probes, recorders, execution plane)

Updated Tue Oct 27, evening ET.

## Done

- Contract v0.1: Pydantic schema, generated JSON Schema and TS types, TS grid decoder with a parity
  test, golden records (fine-tune, pretrain, agent), `upgrade()` migrations, record size and
  decompression caps.
- Repo skeleton, CI (lint, format, schema drift, tests, CPU torch, gitleaks), deploy and zoo
  workflows with actions pinned to commit SHAs. Ready to hand to C.
- Job queue (gap 2, skeleton): idempotent submit, 60 s leases renewed every 20 s, fencing tokens,
  reaper, transactional saves with rollback, per-visitor and global backpressure, fair dispatch
  and honest queue positions, capped and batched events, archived finished jobs.
- Worker loop: heartbeats, stops itself when it cannot renew, cancellation reaches the probe,
  permanent vs internal errors, no internals shown to visitors.
- Harness: static script reader covering every lens in the plan, Muse Spark client (structured
  output, retries that can't double-bill, overall deadline, one re-ask, token caps, provenance per
  attempt, no `-contributor` models for visitor code), mapper that reconciles Spark with the static
  reader, skeleton record with a job map per lens, knob domains and read-time findings.
- Recorder: global hooks for optimizer steps, backward, data loader, eval phases (capped at 2
  batches), output head, SDPA attention logits, checkpoint saves, GradScaler, HF generate and vLLM,
  TRL/HF trainer logs; host RSS and GPU power sampling; failures classified into stable codes.
- Probe runner: validated submissions, scrubbed env and throwaway HOME, non-dumpable parent,
  unprivileged per-probe uid when root, nothing survives the probe, capped logs, hard timeout,
  cancellation, untrusted child output parsed defensively.
- Probe plan from a Monte Carlo (docs/decisions/probe-plan.md): warmup 2, timed 5, median; three
  memory probes; eval and checkpoint pulled into the current-settings probe.
- B's requests: per-kernel table from the profiled probe (top 64 plus per-class tail), canonical
  GPU names and peak FLOP/s from B's table, the five counts in `Probe.extras`, model shape from the
  running model, a `gpu_type` knob, and the knobs B maps (LoRA alpha and dropout, quantization,
  attention kernel, context limit). Checked on a scratch merge with ws-b: 561 tests pass and the
  kernel table feeds `predict_cross_gpu`. The schema carries them (`Probe.kernels`, the
  `ModelInfo` shape fields), added before the freeze with Chiya's approval.
- Size-aware probe GPU (L4, L40S, A100-80GB, H200; after a GPU OOM one step up, or the whole
  ladder when the size was unknown), from a Monte Carlo (K, G, L, U in docs/decisions/probe-plan.md).
- Kernel tables and model shapes from the probe child are checked in the parent before use
  (`probes/untrusted.py`); visitor-supplied names are bounded before any regex sees them.
- Bug zoo: LoRA fine-tune (HF) and nanoGPT pretrain scripts.
- Independent code review: all Critical and High findings fixed with tests; Medium and Low fixed
  except the two noted below.

- Execution plane, as code: the pipeline job function (fetch, map, skeleton before Muse Spark,
  plan, probes, merge; a point that can't be set is skipped, eval OOM re-runs with eval off, a
  per-job time and dollar budget, the content-hash probe cache), the Modal app (job-store
  service, workers, one GPU sandbox per job with no secrets, an egress allowlist and read-only
  weights), and a working `preflight map|run` CLI. One script runs end to end through the queue
  on CPU in the tests.
- Review fixes (Oct 27): `.eval()` before training no longer cuts the data loader; overrides
  handle annotated fields, config classes and argparse flags (default and command line); timed
  steps have no recorder sync points; every step is logged; eval memory per batch; chunked
  attention-logit check; grad norms read after the step; visitor code runs under rlimits.
- Second review pass (Oct 27): uploads live on a volume and the job state keeps a reference; a
  retried job keeps its record id and share token; events are fenced by the lease token; each
  sandbox runs as one pinned uid with its own HF cache; the store's owner is claimed before it
  restores; workers stop leasing before their own timeout; probe time and cost are measured by
  the worker, not reported by the sandbox; flags follow the script's own spelling and argparse
  abbreviations; repos are read from git's object store with no checkout, big blobs never
  downloaded and the clone deleted; malformed sources are visitor errors, not crashes.

## Blocked

- No GPU and no Modal token in this environment, so probes are tested on CPU only. The Modal app
  (`runner/modal_app.py`) builds with Modal 1.6.3 but needs someone with Modal access to deploy it
  and run `prefetch` and `smoke` for the first real zoo run.
- Hugging Face is not reachable from this sandbox, so the zoo scripts run only on Modal.

## Next

- First real run: deploy the Modal app, `prefetch` the zoo weights, `smoke` one zoo script end to
  end (needs someone with Modal access; commands in the hand-off).
- Ground truth: both zoo scripts on a GPU, step time with and without the recorder's sync points,
  to replace the simulated probe-plan figures.
- Measured-rule findings (loader wait, padding, MFU, eval OOM, truncation, retry loops).
- Agent recorder: OTel GenAI proxy around OpenAI-compatible clients; RL phase timer.
- Remaining zoo scripts.

## Known limits, on purpose

- Timed steps carry no recorder sync points, but `by_phase` comes from two synchronised phase
  steps after them, so phase splits lose CPU/GPU overlap. B scales the fractions to `p50_s`.
- The queue saves under its lock. At 8 GPUs and a few visitors that is tens of saves a minute;
  revisit if the store moves off a single container.
- Repos need a full 40-character commit: GitHub only serves a fetch by the whole hash.

## Contract requests

See docs/requests.md.
