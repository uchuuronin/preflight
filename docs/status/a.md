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
- Oct 27 evening: guide jobs on a record (`run_probe`, `run_verify`, `sample_more`) write a new
  record derived from it; `Knob.affects` and `changes_results`, `ModelCard.fixture` and
  `JobEvent.error_code` for C; the Spark client reads `META_API_BASE`; the upload format for D
  (docs/decisions/upload-format.md) with requirements.txt installed in probes and notebook pip
  lines turned into one; probes never above A100-80GB on the demo, and the job budget counts the
  sandbox's CPU and memory; `python -m preflight.runner.stop_app` stops the deployed app through
  remote-run; the smoke run writes its events and record to out/; Apache-2.0 LICENSE.
- Found and fixed before any GPU run: Debian images keep /root at 0700, so the probe's
  unprivileged uid couldn't reach the package there. The Modal image now keeps it in
  /opt/preflight; elsewhere (the AMD checkout) probes import a read-only copy, and nothing on the
  host has its permissions changed.
- Independent review of the evening's work, all fixed with tests: the CLI as root gave probes a
  cache shared across runs that root then chowned recursively (a planted link would have handed
  over /etc), now a fresh per-run directory root never walks; requirements used pip --target,
  which reinstalls torch under anything that needs it, now --prefix with public-version pins and
  one install per job; verify passed a run whose loss went NaN; a stale volume view could feed a
  follow-up an old parent; a visitor's own upload uri and changed code are refused; moving GPU
  drops the parent's fits along with its probes; a mistyped GPU ceiling still caps.

## Blocked

- The first GPU work (deploy, `prefetch`, `smoke` on L4) waits for Chiya's yes on the costed plan;
  runs go through `remote-run.yml`, so nothing here holds a key.
- Hugging Face is not reachable from this sandbox, so the zoo scripts run only on Modal or AMD.

## Next

- First real run, once approved: deploy the Modal app, `prefetch` the zoo weights, `smoke` the LoRA
  zoo script end to end on L4, then `stop_app`; log GPU, minutes and dollars below.
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

## AMD session (Thu Oct 29): what the droplet needs

`amd-run.yml` checks for ROCm PyTorch on the host and installs only this repo and psutil, so:

- The ROCm PyTorch image (torch built for HIP: `torch.version.hip` set). Everything else the zoo
  script imports comes from `zoo/finetune_lora_hf/requirements.txt`, installed in each probe by
  `--install-requirements` (torch itself is never replaced).
- Optional, for power readings: AMD's `amdsmi` Python package (on ROCm 6: `pip install
  /opt/rocm/share/amd_smi`); without it probes record no power, nothing else changes.
- The `video` and `render` groups (stock on ROCm images): probes run as an unprivileged uid and
  need them to open /dev/kfd and /dev/dri.
- Internet to Hugging Face and PyPI, and about 15 GB of disk (Qwen2.5-1.5B, Alpaca, wheels).

The command, once Chiya approves the session:
`python -m preflight.cli.main run zoo/finetune_lora_hf --out out/records --install-requirements --require-gpu`
(as root the CLI pins one probe uid and shares one Hugging Face and pip cache across the probes;
`--require-gpu` fails the run if the probes saw no GPU).

## Cloud runs

Every run's GPU, minutes and dollars. Runs go through GitHub Actions (docs/setup/cloud.md) on T4
or L4 unless Chiya approves bigger; `modal_app.py::smoke` holds its probes to L4.

| Date (UTC) | Run | GPU | Minutes | USD |
| --- | --- | --- | --- | --- |
| Oct 28 02:39 | `a-deploy-main` (run 40622175703): `modal deploy --env main`, image builds | none | 2.4 (build); then 26 deployed | under $0.05 estimated, all CPU; Modal's bill for the window replaces it |
| Oct 28 02:42 | `a-prefetch-main` (40622466016): failed before starting (`list[str]` argument) | none | 0.5 | $0 |
| Oct 28 02:43 | `a-smoke-l4-main` (40622569222): submit failed, the job store never started | none (no sandbox) | 24 | $0 Modal GPU; the CPU is in the deploy row |
| Oct 28 03:07 | `a-stop-main` (40624905511): Chiya had already stopped the app at 03:07:19; out/stop.json lists it stopped | none | 0.4 | $0 |

Failed sitting, Oct 28 02:39 to 03:07 UTC: `JobStoreService` and `tick` crashed on every start
(Modal crash-loop alerts) with `IndexError` at `ROOT = Path(__file__).resolve().parents[3]`.
Modal loads a deployed file on its own as `/root/modal_app.py`, not from the repo, so the lookup
had no fourth parent; moving the package to /opt/preflight that evening didn't cause it, and the
app had never started on Modal before. The same assumption made `image_tag()` hash no files in
the container. Fixed (the repo path is read only where the repo exists, the tag hashes the
installed package and refuses to hash nothing), and `modal_app.py::startup_check` now runs both
images' startup code with no GPU before any deploy. No GPU or sandbox ran, so the $0.95/h sandbox
CPU and memory figure is still a guess until the smoke run's bill.

## Contract requests

See docs/requests/from-a.md.
