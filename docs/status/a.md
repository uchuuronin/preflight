# Status: A (probes, recorders, execution plane)

Updated Tue Oct 27, early morning ET.

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
- Bug zoo: LoRA fine-tune (HF) and nanoGPT pretrain scripts.
- Independent code review: all Critical and High findings fixed with tests; Medium and Low fixed
  except the two noted below.

## Blocked

- No GPU and no Modal token in this environment, so probes are tested on CPU only. Modal wiring
  (`runner/modal_app.py`) waits on Modal access (fallback decision due Wed Oct 28).
- Hugging Face is not reachable from this sandbox, so the zoo scripts run only on Modal.

## Next

- Modal app: probe function with no secrets mounted and an egress allowlist, one container per
  job, the queue store in one container saving to a Modal Dict, the reaper on a schedule.
- Pipeline job function: fetch source, map, skeleton event within 60 s, plan, probes, merge into
  the record, read-time plus measured findings, costs.
- Measured-rule findings (loader wait, padding, MFU, eval OOM, truncation, retry loops).
- Agent recorder: OTel GenAI proxy around OpenAI-compatible clients; RL phase timer.
- Remaining zoo scripts; ground-truth runs once Modal lands.

## Known limits, on purpose

- Step timings synchronise at phase boundaries, so CPU/GPU overlap is lost and small models read
  slightly slow. B's calibration absorbs it; we keep honest phase splits.
- The queue saves under its lock. At 8 GPUs and a few visitors that is tens of saves a minute;
  revisit if the store moves off a single container.

## Contract requests

See docs/requests.md.
