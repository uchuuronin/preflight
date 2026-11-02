# Status: A (probes, recorders, execution plane)

Updated Mon Nov 2, 03:30 ET.

**Production runner: deployed from main 1ea2dd6** (`a-deploy-main-21`, GitHub 41412727746, Nov 2 08:24 UTC / 03:24 ET), after both startup checks passed (`a-startup-check-21` 41412104839, `a-startup-check-serving-7` 41412457816). It carries Glimmer's secret on the worker, private jobs never on Baseten, the Modal capacity fix, the runner-held probe limit, the GPU-wait events, the serving cap (1,024 tokens) and C's fix-loop verify path. The limit check passed after it (`a-limit-check-2`, 41412867356): three probe jobs accepted, the fourth refused with `limit.visitor_daily`. The previous deploy was ws-a 530785e (03:09 UTC); rollback is `modal deploy --env main` from 530785e.

## Decisions overnight (Chiya's autonomy, Nov 2 12:16 AM ET)

| When (ET) | Decision | Why | Cost | How to undo |
| --- | --- | --- | --- | --- |
| 00:20 | Formatted with CI's own ruff (0.17.3, from uv.lock) instead of the local 0.16.9 | CI's format check failed on a notebook the local ruff left alone (ed85b1b); fixed in 1e2067c | none | none needed |
| 00:25 | GPU waits shown as `probing` events ("waiting for a free L4 GPU …", "… ready; probe running") plus `waiting_for_gpu: {gpu, since}` in GET /v1/jobs/{id}, not a new stage | a new stage is a contract change; the status passthrough and the free-text message need none | none | revert the commit; the key is additive, C may ignore it |
| 00:25 | A private job is never triaged on Baseten: with `harness.glimmer_private` off it gets no triage (it used to fall back to Spark Open) | the contract's privacy promise (0.6.2), found by review | none | `pipeline._triage_client`: return `deps.triage` for private jobs again |
| 00:25 | The runner holds the probe's own time limit (past it the sandbox ends, the probe reads `timeout`) | two probes ran to the sandbox's 30 min with no log (serving, eval) | none | `sandbox_runner.py` watcher: drop the limit check |
| 00:30 | Ran the private-triage check on the production LoRA record (zoo, not derived) through the ops app with the worker's own clients and secrets, `harness.glimmer_private` on in that process only | item 3: proves the worker path before the PM's flag | $0.163 (Glimmer approval: $1.46 of $3 used) | nothing to undo; production's flag untouched |
| 00:35 | Specific failure lines for the submission, patch and demo-limit codes; the limit lines reach visitors only once C's API adds them (asked in from-a.md) | item 43: every code names its step and a way past it | none | revert 2637ee6 and its fix |
| 00:40 | `warm_pool_dry_run` in the ops app, read only (hydrates the deployed worker, reads the tick's last setting) | item 45: a dry run without touching production; the pool stays off until Nov 5 | none (CPU, about a minute when run) | delete the function |
| 00:45 | Wrote zoo/notebook_tiny/tiny_train.ipynb (a 200-step MLP with AdamW) for the notebook check-in | item 5: D's colab_lora fixture has no optimizer, so it can't pass as a training check-in | none | delete the folder |
| 03:20 | Redeployed the runner from 1ea2dd6 itself, not main's head 89ed12e | the PM's note names 1ea2dd6; 89ed12e only adds that note (docs), and its CI's one failure is a flaky agent-proxy test, fixed on ws-a in 6c73978 | none (CPU checks, the serving check's L4 minutes) | `modal deploy --env main` from 530785e |
| 03:25 | Agent proxy listen backlog 5 → 128 (6c73978) | main's CI (89ed12e) saw one connection reset in the 20-call cap test; not reproduced locally in 60 runs under load, so the backlog is the likely cause, not a proven one; a visitor's agent can burst that many calls too | none | revert 6c73978 |
| 03:27 | Ran `warm_pool_dry_run` against production (`a-warm-pool-dry-1`, 41413074651), read only | item 45's dry run before the pool turns itself on | CPU, about a minute (cents) | nothing to undo |
| 01:25 | **The "ended without a result" cause, found by review in Modal's SDK (1.6.3):** a GPU sandbox's create itself waits up to 21 min (1,260 s) for scheduling and then raises "Insufficient capacity"; the runner caught it as a broken sandbox, so no fallback, no NoGPU and no wait recorded. Now create runs inside the runner's wait: told each minute, bounded by the job's wait cap, an L4 out of capacity falls back to the A10, the last GPU gives `gpu.unavailable` naming Modal's reason, and a create still waiting when the runner gives up is ended when it returns | three probes (serving, eval, a training notebook) lost this way; a queue wait read as a hang | none | `sandbox_runner.py`: create back in the loop |
| 01:05 | `READY_TIMEOUT_S` 30 → 25 min, and a failed sandbox start check is named on the probe that lost its result | the training-notebook check-in (`a-notebook-checkin-2`, 41394704676) ended like the serving and eval probes: "without a result" 1,284 s in, no GPU wait recorded; the same script ran in 5 s under the recorder on an A10 outside the sandbox (`a-gt-notebook-tiny`, 41397365737), so the sandbox path is at fault; the likely cause, a start check as long as the sandbox's own 30-min life failing at once, isn't proven | $0.64 (the check-in), about $0.02 (the ground truth) | `sandbox_runner.py`: READY_TIMEOUT_S back to 30 min |

**Monday's unseen runs, ready (item 8; the list not opened):** from the `validation-freeze` tag, per script:
(1) `ops_app.py::smoke --dir <dir> --entrypoint <file> --args=--a=1,--b=2 --visitor a-unseen-<n> --gpus L4,A10` checks it in through production as an upload (the record, the failures naming their step);
(2) `ops_app.py::ground_truth --dir <dir> --entrypoint <file> --args=... --bare-steps <n> --first-gpu A10 --install-requirements` for its full run (bare and recorder, step times, memory; serving and torchtitan scripts pick their own image).
Arguments go comma-separated (remote-run allows no quotes), `,,` for a comma inside one. Cost per script about $0.20 to $0.60 (check-in) plus the full run's own minutes on an A10 (about $1.10/h), under Chiya's $30 for Monday.

**Checks passed overnight (for the PM's flags):**
- **`harness.glimmer_private`: passed.** `a-private-triage-1` (GitHub 41394153172, Nov 2 05:33 UTC): the worker's private-job triage (`pipeline._triage`, private=True) with the worker's own clients and the preflight-glimmer secret, now mounted on `run_worker` too, live since the 08:24 UTC redeploy from 1ea2dd6: Glimmer triaged the LoRA record's p1-eval as `triage.oom` ("The eval pass ran out of GPU memory after 3 optimizer steps", quoting the recorded line), one call, 279 s (cold), $0.163; no call to Baseten. With the flag off, a private job now gets no triage at all, never Baseten.
- **Warm pool (item 45), dry run passed** (`a-warm-pool-dry-1`, Nov 2 08:27 UTC, against the deployed runner, changing nothing): the deployed `run_worker` is found and scalable; the tick last set 0 warm workers and wants 0 now; from Nov 5 00:00 ET it wants 1 warm worker and L4 jobs probe on the A10 (a job allowed only the L4 stays on the L4), back to 0 and the L4 at Nov 9 00:00 ET. The pool turns itself on at Nov 5 00:00 ET from the deployed code, with no other step.
- **`d.notebook`: not yet.** The runner accepts a notebook end to end (`a-notebook-checkin-1`, Nov 2 03:11 UTC), but the one that trains (`a-notebook-checkin-2`) lost its probe in the sandbox as above; to rerun after the next redeploy.

- **Now:** the Pareto sweep (Muse Spark arm, then Spark Open, on its own $1 ledger); the redeploy gated on the new probe-user startup check (a tiny torch.compile and the vLLM and datasets imports as uid 61000), then the serving probe again on the fixed image.
- **Done since Oct 27:** every lens's recorder and probe path, the agent proxy and sampler, run hygiene and measured findings, `JobRequest.patch`, the HF Trainer/TRL/Lightning/torchtitan adapter checks, the Muse Spark script-map gate; LoRA ground truth on L4, GRPO 50-step ground truth on A10, DDP and FSDP on 2 × A10; the production runner deployed and left running.
- **Next, in order (Chiya, Nov 1):** (1) Spark diagnosis, slim (item 31, behind `harness.spark_diagnosis`, off): one Muse Spark call per finished record with the rendered memory timeline and loss plot and the raw probe numbers, findings are disagreements between the two (or "consistent"), each citing the plot region and the number; no search grounding. (2) Spark Open checks each diagnosis claim against the probe numbers through a tool call (item 30, slim); no supporting number is "abstain, or run a probe"; every call logged. (3) pip-installable CLI and a 60-second quickstart (item 42). (4) Prebuilt images and a warm pool for judging, Nov 5–8 (item 45); the pool's cost is an ask before it's on. (5) Every failure names its step and offers a manual entry point (item 43, harness side). (6) Eval-sweep record (item 19), RL entropy and KL (item 14). Also: one real probe through the production runner once round five turns `harness.spark_open_triage` on, its share link, triage calls and spend posted.
- **Planned, for Monday's cut sheet (not started):** the LR range test, the FakeTensor cross-check, Chakra export, DDP/FSDP beyond the Nov 1 2 × check, the rate-limit wall clock, A's half of data repetition, checkpoint resume exact-match.
- **Standing yes (Chiya, Nov 1):** runs under $1 on a T4, L4 or A10, up to $5 a day for A, without asking; pushes to ws-a once gitleaks is clean and one read-only review has checked the change, each push's contents listed under Pushes below. Still asked first: over $1 or past $5 a day, A100/H100, a new kind of run, any deploy or anything that keeps billing, main, production flags, contract changes (posted before the PM's noon and 8 PM ET batches).
- **Unseen scripts:** A has not opened `unseen-scripts.md` in the Project, and won't before the list arrives on Monday.

## Pushes to ws-a (what each sends)

| When (ET) | Range | What it sends | Review | gitleaks |
|---|---|---|---|---|
| Nov 1 10:57 | 9e46c3c..4cc6f57 | dataset size from the probes (0bdff6f, abb93ad, fb84348); the probe uid's user name and passwd/group entry (e5b0713, 4cc6f57); the A10 GRPO record, run log and B handoff (03d1420) | 3 read-only rounds (dataset size 2, probe user 1) | clean |
| Nov 1 11:06 | 4cc6f57..9afbf4c | the probe-user startup step (47f68ed, 88cd732); the C note on the redeploy (cf96ea5); status and B notes (ead63bb, 17582b7, 9afbf4c) | 1 read-only round | clean |
| Nov 1 11:20 | 9afbf4c..20ebb7a | the torch.compile startup step and gcc/g++ in the sandbox images (b86b72a); pyarrow 17 in the serving image and its datasets import check, the wrong-version message (0771c5f, 20ebb7a); a comment order (bdcebd6); status (53c392c, 107ce3e) | 1 read-only round each | clean |
| Nov 1 12:10 | 20ebb7a..998704f | main merged (0.6.1, f017010); Spark diagnosis (485df08, ace3a06, 97a458a, 7a55eb7) and Spark Open's check (202666f), both behind harness.spark_diagnosis; 0.6.1 concurrency and the review fix (fde2e68, 549a5f5); engine flags and N_TRAIN_* (913e2b6); `preflight demo` and the root/venv probe fix (eb5e5e5, 060ec5f); failures name their step (ab74a70); triage's eval node (30b0224); the serving probe's sampled generate (16d5cd6, 998704f); the CI fix, the deploy gate's rule out of ops_app (26b134b); requests and status (3f58945, 579cd23, 179c8ad, 390f778, 40fed56) | 2 rounds for diagnosis, 0.6.1 and the engine/N_TRAIN change; 1 round for the rest | clean |
| Nov 1 12:30 | 998704f..f46fad2 | RL entropy estimate and its limit (d9f969c, f2f2c5e); the warm-pool ask and status (e851890, b4480d3); main merged (round five, 0.6.2) | 1 read-only round each, 2 for the entropy change; the merge checked for damage in A's files | clean |
| Nov 1 12:49 | f46fad2..8154268 | main's production flags only (3c6699e: runner.queue, c.guide.live), merged onto f46fad2 so the deploy carried nothing unreviewed | main's own (PM); the merge changed flags.production.json only | clean |
| Nov 1 13:06 | 8154268..HEAD | the ops diagnosis run and its cap (1e40d5e, ec0b39a, aaaefa8, 16ee36f); Glimmer's client and private-job routing (ea1626a, bdd3c6f); the demo limits (9a2bb82, 5a78a60); requests and status (67dd9fc, eb8adbb, b54d854, this row) | 2 rounds each for the cap and the limits, 1 for Glimmer | clean |
| Nov 1 13:12–13:43 | ..b300684..8da594b | the CI fix for the diagnosis-run test (ba2d9f5); triage reads the probe's recorded error (b300684); diagnosis at medium effort, 8,192 tokens, a 180 s timeout (79db248, 8da594b) | 1 read-only round each | clean |
| Nov 1 14:00 | 8da594b..9566d23 | main merged (6757667); ground truth for any GPU script (7be9039, 0fba764); the judging warm pool, (a) and (c) (963bb9e, 9566d23); status (fbcac17) | 1 read-only round each, findings fixed | clean |
| Nov 1 14:09 | 9566d23..243ace5 | eval probes measure two generate calls (fda1762) and count only the outermost (243ace5); status (aa84cd6) | 1 read-only round, its finding fixed | clean; CI green in every job |
| Nov 1 16:05 | 243ace5..d73b4e1 | status for the 13:06 pushes (8040297); the retrieval, GNN and vision zoo records for C, share tokens as placeholders (576f6b6); the AMD startup check `python -m preflight.probes.user_check` and the Nov 1 AMD plan (62fac50); its review fixes: psutil reported if the probe Python lacks it, a ROCm torch required where /dev/kfd exists, a launch error still prints its line, gate tests for a missing passwd/group entry | 1 read-only round, its 4 findings fixed | clean (tree and commits); CI: python job failed once on a flaky agent-proxy test (connection refused), green on rerun |
| Nov 1 16:45 | d73b4e1..(this) | the agent-proxy test flake fix (22d9a79); the GRPO record names its A10 (env.gpu and the gpu_type knob), ground truths carry env, dataset size from cached and eval-OOM probes, Glimmer 330 s timeout and `ops_app.py::triage` (0874c0f); its review fixes (6190ec7); the verify eval pull (88c8387) and its revert (3be3ed4, the review found it unsafe); status | 2 read-only rounds (one per change); the first's findings fixed, the second's change reverted | clean (tree and commits) |

## Known limits, on purpose

- Under the L40S ceiling (Nov 1), a probe or verify follow-up on a record measured on an A100
  before that night runs on the L40S with the cross-GPU warning, and, the GPU being different,
  starts afresh: the parent's probes, predictions and models aren't carried (review, Nov 1).
- The Glimmer arm prices each call by its own wall time at the A100-40GB rate: calls sharing the
  container are each charged in full and its cold start and idle time aren't, so the proxy's count
  isn't Modal's bill. Each call reserves its longest possible time, so 4 calls in flight hold
  about $0.93 at a 200 s timeout (review, Nov 1).
- The serving probe's 1,024-token cap (a-serving-cap, 305b745) is held back from the Nov 1 night
  deploy: B's serving simulation reads the probe's reply lengths as the real ones and doesn't read
  `extras.serving_probe_max_tokens` yet, so it would under-predict long replies as measured.
- Structured replies (Muse Spark, Spark Open, Glimmer) are read leniently: the JSON object after
  any reasoning, the last fenced one, else the outermost braces, validated in full. A reply whose
  only object is one it quoted from the visitor's log is accepted as the answer; it can only
  mislead that visitor's own job, within the schema's fixed categories (review, Nov 1).
- A probe sandbox that outlives the probe's own time limit (the 18:06 serving run on an L4: 1,284 s
  against 600 s) is ended from outside and its log is lost, so its record says only "ended without
  a result". Not fixed yet; the serving cause itself is known (`a-serving-debug-1`, Nov 1).
- Verify jobs don't run the eval inside their window, so eval fixes can't be verified yet (the
  first attempt was reverted after review; the design is with C, Nov 1).
- Timed steps carry no recorder sync points, but `by_phase` comes from two synchronised phase
  steps after them, so phase splits lose CPU/GPU overlap. B scales the fractions to `p50_s`.
- The queue saves under its lock. At 8 GPUs and a few visitors that is tens of saves a minute;
  revisit if the store moves off a single container.
- Repos need a full 40-character commit: GitHub only serves a fetch by the whole hash.
- RL phases: weight sync misses the ZeRO-3 parameter gather (it lands in forward) and vLLM V1
  (only the V0 engine path is wrapped); with PEFT and `num_iterations` > 1 the reference forward
  isn't told apart from the policy's old log-probs, so it counts as forward.
- Agent sampler: a task's key hashes its first call without the system prompt, so a timestamp or
  id in the first user message still defeats the top-up's skip; tasks that all open with the same
  call can't be topped up. A run with no model call for 2 minutes (and none in flight) is stopped as stalled, so a coding agent's tool step longer than that is cut. Which task is stuck in its tool can't be told at the proxy from one that finished on a tool call (a submit, a passing test run), so none is singled out: the run's stall is the signal, and selection says how many tasks ended on a tool call and that up to the script's workers of them may be stuck; those tasks keep the proxy's own reading (done).
  The proxy's refusals carry `x-should-retry: false`, which the OpenAI SDKs honour; other
  clients may back off on them. Side calls (judges, sub-agents) start tasks of their own.
- In-flight mode: an eval or checkpoint pause inside a window can raise a false step-time alert.
- GPU fallback (L4 to A10 to L40S after 2 min, Nov 1): a job's budget pre-check prices the next
  point at the GPU asked for, so after a fallback a job can overspend by about one point; an A10
  run is never stepped up from (the ladder starts at L4), and an A10 that fit doesn't prove an L4
  would; a follow-up whose parent ran on an A10 asks for an A10, which has no fallback; an image
  change waits the 2 minutes for an L4 again; up to two waiting threads per job can linger for
  30 min; an L4 scheduled just after the cutoff runs a few uncounted seconds before it's ended.
- Spark Open triage (item 29, behind harness.spark_open_triage): calls run one after another
  after the probes, up to 30 s each, outside the job's probe time budget and with no cancel check;
  the quote check only asks that the quoted line (8 characters or more) appears in the log tail.
- GNN probes read graph sizes from loader batches (PyG's loaders, a DataLoader of Data objects): a
  script that samples subgraphs by hand, as zoo/gnn_pyg does, gets no graph counts, and with no
  loader the forward can't be told from the sampling, so both land in "other" (Nov 1 L4 probe:
  60% other). On the cut sheet (Chiya, Nov 1): count each conv layer's edges at SAGEConv/MessagePassing forward, about 1 hour.
- FSDP (Nov 1, 2 × A10): `params_total` and `params_trainable` count a rank's shard (67M of the
  fixture's 134M), since FSDP's flat parameters hold only that rank's part; per-rank peaks are right.
  The sharded run's tokens and samples per step read 0 there and in torchtitan (no batch shape seen).
- Probe-user startup check (Nov 1): runs in the probe's HOME, not a workdir; a timeout kills only
  its direct child; the `user` it reports is the USER the probe sets, so only the passwd and group
  lookups test the image. The gate is the manual `startup_check` before a deploy, not CI.
- Judging window (Nov 5–8, judging.py): a job starts on the A10 only when its GPU wasn't asked for
  or inherited; its first point's eval-off rerun may still fall back to the L4; if point 0 is
  skipped, the first point that runs has no L4 fallback; with the GPU ceiling at L4 the A10 is
  capped back with a misleading cross-GPU warning.
- Ground truth for any script (Nov 1): `--bare-steps` means nothing for a script with no optimizer
  (serving, eval, retrieval): its bare run is the whole script, capped at 20 min; two `--dir`s with
  the same last folder name write the same out file.
- Demo limits (Nov 1): the spend cap counts what jobs report (GPU at measured probe time plus the
  sandbox estimate), so sandbox start-up and idle time aren't in it, and one job can pass its $0.60
  between probe checks; a retried job is charged for its first attempt and reserves again. A worker
  on new code calling a store on old code (only mid-deploy) fails its report.
- RL entropy (Nov 1, item 14): rl.kl is recorded from TRL's logs; rl.entropy is TRL's own where it
  logs one, else a Monte Carlo estimate (mean -log p of sampled completion tokens, biased by the
  temperature, top-p/top-k and the EOS cut, flagged by extras.rl_entropy_estimated) on TRL versions
  with a `_get_per_token_logps` method. **TRL 0.14, the probe image's, gets no entropy**: its
  log-probs are a closure inside compute_loss. On the cut sheet: exact entropy from the policy's
  logits in a chunked forward hook (about 1.5 h; extra vocab-wide memory on a GRPO run already at
  22 of 24 GB, so it needs a GPU check).
- Serving probe sample (Nov 1): the first generate call is the measurement, so a warmup call
  before the real one becomes it; prompts and per-prompt lists are cut only when passed as the first
  argument or by keyword, not prompt_token_ids, lora_request or priority by position.
- Dataset size (`extras.train_dataset_len`, Nov 1): the training loader's sampler length, else its
  dataset's. With `batch_size=None` a split sampler isn't seen, so validation rows are counted; a
  Lightning `DistributedSamplerWrapper` over a split falls back to the whole dataset. A GNN script
  that samples by hand (zoo/gnn_pyg) has no loader, so no size; D is to expose its training-node
  count as a plain setting (Chiya, Nov 1), which A reads once it lands. The recorder side hasn't run on
  torch in this workspace; the first GPU probe is its check.

## AMD session (Nov 1 plan, for today's code): A's amd-run.yml calls

Each is one `amd-run.yml` dispatch with `-f host=<droplet IP> -f port=22 -f ref=REF`, run in this
order, one at a time. REF is the ws-a commit named in the reply to the PM (the head that passed
CI). Nothing starts before the PM posts the droplet's IP. Only the last call passes
`destroy_droplet_id`, and only if B has said B is done.

| # | name | `cmd` | timeout | Expected | What it gives |
| --- | --- | --- | --- | --- | --- |
| 0 | `a-amd-usercheck` | `python -m preflight.probes.user_check --uid 61000 --gpu` | 15 | 5 min | the startup check: uid 61000 as a probe launches it, the video/render groups, /dev/kfd opened, a GPU matmul, torch.compile on the CPU and on the GPU (ROCm Triton), the probe child's imports |
| 1 | `a-amd-lora-record` | `python -m preflight.cli.main run zoo/finetune_lora_hf --out out/records --install-requirements --require-gpu` | 45 | 15 min | the LoRA zoo record on ROCm |
| 2 | `a-amd-lora-gt` | `python -m preflight.recorder.ground_truth zoo/finetune_lora_hf out/gt_lora_mi300x.json --bare-steps 120 --recorder-steps 40 --install-requirements --budget-min 30` | 40 | 15 min | the LoRA ground truth: bare vs recorder step times |
| 3 | `a-amd-nanogpt-record` | `python -m preflight.cli.main run zoo/pretrain_nanogpt --out out/records --install-requirements --require-gpu` | 45 | 15 min | the nanoGPT record |
| 4 | `a-amd-nanogpt-gt` | `python -m preflight.recorder.ground_truth zoo/pretrain_nanogpt out/gt_nanogpt_mi300x.json --bare-steps 300 --recorder-steps 100 --install-requirements --budget-min 30` | 40 | 20 min | the nanoGPT ground truth |

About 70 minutes of runs plus about 2 minutes of setup per call (rsync, venv, `pip install -e .`):
**about 80 minutes expected [guess], about 1 h 40 min with one retry, 2 h at most; $3.50 to $5.20
of droplet time at $2.59/h** (the Oct 29 rate; B's hours are B's own).

What changed since Oct 29:
- Step 0 is new. On the droplet the checkout and `.venv-amd` sit under /root, which a fresh uid
  can't enter, so `preflight.cli.main run` there runs its probes as root (the Nov 1 quickstart
  fix, `uid_can_run_this_python`). Steps 1 to 4 therefore don't exercise the probe user; step 0
  pins uid 61000 and launches exactly as a probe does. Measurements don't depend on the uid.
- A failed step 0 doesn't block steps 1 to 4: it means a root `preflight run` on ROCm with a
  reachable Python would fail, which goes in known limits with its error.
- The droplet has no passwd entry for 61000: the check reports the name as missing rather than
  crashing (the Modal gate still fails an image without one).

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

Commands for A's part, in order, each one `amd-run.yml` call (`-f host=<ip> -f ref=ws-a`); only
the last passes `destroy_droplet_id`, and only if B is done too. Times are guesses from the L4
smoke run (Oct 28) and the MI300X being faster per step; downloads and installs dominate:

| # | `cmd` | timeout | Expected | What it gives |
| --- | --- | --- | --- | --- |
| 1 | `python -m preflight.cli.main run zoo/finetune_lora_hf --out out/records --install-requirements --require-gpu` | 45 | 15 min | the LoRA zoo record on ROCm (memory, step time, kernels, power if `amdsmi` is there) |
| 2 | `python -m preflight.recorder.ground_truth zoo/finetune_lora_hf out/gt_lora_mi300x.json --bare-steps 120 --recorder-steps 40 --install-requirements --budget-min 30` | 40 | 15 min | bare vs recorder step times, the recorder's overhead on ROCm |
| 3 | `python -m preflight.cli.main run zoo/pretrain_nanogpt --out out/records --install-requirements --require-gpu` | 45 | 15 min | the nanoGPT record (TinyStories download, the planted slow loader) |
| 4 | `python -m preflight.recorder.ground_truth zoo/pretrain_nanogpt out/gt_nanogpt_mi300x.json --bare-steps 300 --recorder-steps 100 --install-requirements --budget-min 30` | 40 | 20 min | the same for a from-scratch pretrain |

About 65 minutes of runs plus 2 minutes of setup per call: **1.25 hours expected, 2 hours with one
retry, so $3.25 to $5.20 of droplet time at $2.59/h** (B's corpus slice is B's own hours on the
same droplet). Unlike `modal run`, the droplet's output reaches the workflow log as it comes, so
a stall shows where it is. Things that could cost a retry: the venv is made without `--seed`, so
installing requirements relies on the image's own pip being importable (step 1 shows it in its
first minute); a version clash between the image's packages and the zoo's pins (`datasets` wants
`fsspec<=2024.9.0`). The planted eval OOM may not fire on 192 GB, or may fire later: the eval keeps
every batch's graph, so its memory grows with the batch count; either way the record says which.

## Every plan item A owns (Oct 31 morning ET; updated 08:45 ET)

From the Contracts tab (Contract A, the gaps table, the Oct 27 review, the competitive review) and
the main plan's feature set and validation targets. "main" = merged; "ws-a" = pushed to the
branch, not yet merged (it carries round three: agent recorder, retention, longest-row extras,
ground-truth module, log reader); hours are A's remaining work, honest guesses, GPU time not
included. Hero lenses first.

| # | Item (source) | Where | Left, h |
| --- | --- | --- | --- |
| **Fine-tuning (hero)** | | | |
| 1 | Probes, recorder, memory terms by term, eval-OOM re-run, P(OOM) inputs (lens) | main; L4 run Oct 28 | 0 |
| 2 | Packing and padding-waste measurement (lens, contract) | main (`padding_fraction`, `tokens_per_sample_mean`) | 0 |
| 3 | Longest dataset row for B's memory rule (B, Oct 30) | ws-a | 0 |
| 4 | FakeTensor cross-check of the analytical memory terms (contract) | handed to B (PM, Nov 1 9:47 PM) | 0 |
| 5 | L4 ground truth: recorder overhead, where the OOM lands (review Oct 27) | bare run measured Oct 31; recorder half failed on a cache permission (fixed locally); rerun with 210 steps needs a yes | 0.5 + paid run |
| **Agent and API runs (hero)** | | | |
| 6 | OTel GenAI proxy around any OpenAI-compatible client (lens, contract) | ws-a (`recorder/agent.py`) | 0 |
| 7 | Sampler: 30 tasks, 80% runaway bound, 20 more when the range straddles the budget (lens, D3) | ws-a: proxy run mode, pipeline, `sample_more`, CLI; local: the hosted path (proxy behind a tunnel, script in a sandbox with no secrets), first run waits for a yes; the bound is B's `agent_totals` | 1 + API spend |
| 8 | Rate-limit wall-clock (lens) | not built by Nov 4 (PM, Nov 1 9:47 PM) | 0 |
| 9 | Pareto model sweep across Muse Spark and Spark Open (lens, contract) | not started | 4 |
| 10 | Agent zoo script and an agent record on the public app (G1) | not started (D writes the script) | 2 |
| **RL post-training (hero)** | | | |
| 11 | GRPO zoo script, public model and data, 512-token completion cap planted (competitive #1) | local (`zoo/grpo_trl`); review: at 512 the truncation rate may be only a few percent on Qwen2.5-0.5B, ground truth will say | 0.5 |
| 12 | RL phase timer: rollout, reference, reward, update, weight sync (lens, contract) | local: all five; weight sync misses the ZeRO-3 gather and vLLM V1 (V0 engine path only) | 0.5 |
| 13 | Pass rate per prompt for B's zero-variance fraction (lens) | local (`recorder/rl_groups.py`, `rl.pass_rate_per_prompt`, `rl.zero_spread_fraction`) | 0 |
| 14 | Entropy, KL, length and truncation signals (lens) | partial on main (TRL logs become RL signals; entropy estimate from the policy's log-probs); the rest handed to B (PM, Nov 1 9:47 PM) | 0 |
| **Campaign (hero; B owns the model)** | | | |
| 15 | Probes for a grid's points (a campaign is many records from one script) | not built by Nov 4 (PM, Nov 1 9:47 PM) | 0 |
| **Other lenses** | | | |
| 16 | LM pretraining: throughput, MFU, loader probe, stability signals (lens) | main | 0 |
| 17 | LR range test (lens, contract) | not built by Nov 4 (PM, Nov 1 9:47 PM) | 0 |
| 18 | Checkpoint resume exact-match (lens, contract, hygiene) | local: what the script's torch.save is given (weights with or without optimizer state, LR schedule, RNG, step; HF Trainer's optimizer.pt / scheduler.pt / rng_state by name) becomes `hygiene.resume_mismatch`. A content check, not a resumed run replayed step for step | 3 |
| 19 | Evaluation sweeps: HF generate vs vLLM throughput, thinking-cap truncation, activation-dump storage (lens) | hooks on main, eval probe stops after two generate calls; the eval-sweep record handed to D (PM, Nov 1 9:47 PM) | 0 |
| 20 | Serving probes on vLLM for B's model (contract) | hooks on main; local: the serving image (vLLM 0.6.4.post1, its own torch) and routing; startup check and first probe wait for a yes | 3 + paid run |
| 21 | Retrieval probes through an adapter (lens) | local: encode, faiss add/train/search timed with items a second; a stage after training isn't reached by a probe | 3 |
| 22 | GNN probes through an adapter (lens) | local: nodes, edges, seeds, nodes per hop, largest batch; sampling time is the data phase; needs a PyG script on a GPU | 3 |
| 23 | Vision probes through an adapter (lens) | local: images per batch, resolution, channels-last; images/s from samples; needs a vision script on a GPU | 2 |
| 24 | DDP and FSDP on one multi-GPU Modal node (lens, contract) | not started | 6 + paid run |
| 25 | Run hygiene: stale checkpoints and caches, identical outputs from different weights, truncation rate (lens) | local: findings `hygiene.stale_checkpoint` (checkpoint dirs sent with the code), `hygiene.cache_not_keyed` (a fixed-path `if exists: load`), `generation.truncated` (over 10%, Wilson interval), `hygiene.identical_outputs` (two weight states, same tokens on every compared input). Not yet on a GPU run | 3 |
| 26 | Data repetition and quality exposure, A's half: read per-document quality scores (lens) | static detection only; per-document scores not built by Nov 4 (PM, Nov 1 9:47 PM) | 0 |
| 27 | In-flight collector (feature, contract) | local (`recorder/inflight.py`, `preflight watch`); getting the live record to the guide waits on C | 1 |
| 28 | Adapters: HF Trainer, TRL on main; Lightning and torchtitan (contract) | Nov 1, CPU probes of each framework's own loop (tests/test_adapters.py; tiny random models and synthetic data, shaped on each project's public example, since this workspace can't download from the Hub): **HF Trainer working** (transformers 5.21; also on L4 in the LoRA ground truth), **TRL working** (SFTTrainer, trl 1.17; GRPOTrainer on L4 in `a-gt-grpo-l4-3`), **Lightning working** (2.6.6; its planned run length now read from Trainer.fit). All three: steps counted, phase split, planned steps 40 of 40. **torchtitan working** (0.1.0 on an A10, Nov 1 `a-torchtitan-check-2`: steps and phase split; tokens and samples per step not read) | 4 of 4 |
| **Muse Spark** | | | |
| 29 | Spark Open triage of every probe log on Baseten, every call logged (contract) | client on main, triage not wired | 3 |
| 30 | Spark Open checks each claim against the raw probe numbers (competitive #5) | not started | 5 |
| 31 | Spark diagnosis: traces and plots as images, search grounding, cited findings (contract) | not started | 5 |
| 32 | Glimmer host proposal with B (competitive #6) | not started | 2 |
| **AMD** | | | |
| 33 | ROCm probes on the MI300X, AMD kernel names (contract) | main (prepared); blocked: the AMD Developer Cloud account is on a verification hold (Oct 31 evening). Session Sunday 9 AM if it clears, else to the noon cut with AMD answers from datasheets and the cross-GPU model, labelled not measured | 2 + droplet |
| 34 | Droplet idle watchdog (contract) | not built by Nov 4 (PM, Nov 1 9:47 PM) | 0 |
| **Platform and gaps** | | | |
| 35 | Job store, queue, leases, reaper, probe cache, sandbox policy, migrations, token caps, AMD offline (gaps 1, 2, 3, 6, 8, 9, 11) | main | 0 |
| 36 | Hostile-repo fixture and sandbox tests (gap 17) | main | 0 |
| 37 | LICENSE (gap 22) | main | 0 |
| 38 | 30-day retention of uploads, probe cache, job index (Chiya, Oct 30) | ws-a | 0 |
| 39 | Runner deploy with `engine.memory` on (G1) | waits for round three on main | 1 |
| 40 | Execution-plane logs and metrics (gap 14) | partial (backlog alarm, cleanup log) | 2 |
| 41 | GPU power sampling (contract) | main | 0 |
| 42 | pip-installable CLI, one-command install, 60-second quickstart (contract) | CLI on main, quickstart checked (Nov 1); packaging handed to D (PM, Nov 1 9:47 PM) | 0 |
| 43 | 8 of 9 unseen scripts harnessed, failures name the step (contract; PM picks them) | not started | 6 |
| 44 | Hash of the unseen-script list, published before Nov 2 (competitive #3) | not started (needs the PM's list) | 0.5 |
| 45 | Prebuilt images and a warm pool for judging, Nov 5–8 (contract) | built: warm pool for Nov 5–8 (runner/judging.py), probe images prebuilt at deploy; A keeps it | 3 |
| 46 | Chakra trace export (contract) | not built by Nov 4 (PM, Nov 1 9:47 PM) | 0 |
| 47 | Any OpenAI-compatible endpoint in the CLI's model sweep (competitive #11, after G2) | not started | 2 |
| 48 | First public number, "one script" (competitive #2) | ws-a (status file) | 0 |
| **Validation rows A feeds** | | | |
| 49 | Peak memory within 10%, no missed OOMs: full runs of the zoo and unseen scripts | 1 real run so far | 3 + paid runs |
| 50 | Step time within 15%, same GPU: full runs | L4 ground truth pending | 2 + paid runs |
| 51 | Rollout time within 20%: GRPO zoo script and an unseen TRL script | blocked on 11, 12 | 2 + paid runs |
| 52 | Agent run total cost inside the 90% interval: a full benchmark vs the 30-task sample | blocked on 7 | 2 + API spend |
| 53 | Planted problems caught (bug zoo) | LoRA eval OOM caught; nanoGPT not yet on a GPU | 1 + paid run |
| 54 | Cost of one Preflight run under $1 and under 5 minutes | **miss**: $0.10 GPU, but 6 min 45 s upload to record on Oct 28 (`a-smoke-l4-main-4`, measured), and later runs waited 14 to 26 min for an L4; the GPU fallback (next) addresses the wait | 3 |

**Total left: about 113 hours of A's work (was 138 in this morning's first version) (sum of the column, before paid runs)**, against roughly 30 working hours to noon Sunday. The
order the PM set (hero lenses, then the rest) is the order I work in; Chiya's cuts at noon Sunday
can read the hours column. Paid runs are asked for one at a time.

## Cloud runs

Every run's GPU, minutes and dollars. Runs go through GitHub Actions (docs/setup/cloud.md) on T4
or L4 unless Chiya approves bigger; `ops_app.py::smoke` holds its probes to L4.

| Date (UTC) | Run | GPU | Minutes | USD |
| --- | --- | --- | --- | --- |
| Oct 28 02:39 | `a-deploy-main` (run 40622175703): `modal deploy --env main`, image builds | none | 2.4 (build); then 26 deployed | under $0.05 estimated, all CPU; Modal's bill for the window replaces it |
| Oct 28 02:42 | `a-prefetch-main` (40622466016): failed before starting (`list[str]` argument) | none | 0.5 | $0 |
| Oct 28 02:43 | `a-smoke-l4-main` (40622569222): submit failed, the job store never started | none (no sandbox) | 24 | $0 Modal GPU; the CPU is in the deploy row |
| Oct 28 03:07 | `a-stop-main` (40624905511): Chiya had already stopped the app at 03:07:19; out/stop.json lists it stopped | none | 0.4 | $0 |
| Oct 28 04:33 | `a-startup-check-main` (40632951774): both images start; a tiny CPU probe ran 9 steps as uid 61000 | none | 1.2 | under $0.01 (estimate) |
| Oct 28 04:35 | `a-deploy-main-2` (40633100843): deployed 04:35:12, stopped about 04:39 | none | 4 deployed | under $0.01 (estimate) |
| Oct 28 04:35 | `a-prefetch-main-2` (40633194148): Qwen2.5-1.5B and gpt2 in `preflight-weights` | none (CPU) | 1.4 | under $0.01 (estimate) |
| Oct 28 04:37 | `a-smoke-l4-main-2` (40633354342): submit refused (`StoreUnavailable`), no job ran | none | 0.6 | $0 GPU |
| Oct 28 04:38 | `a-stop-main-2` (40633463121): out/stop.json `stopped: true` | none | 0.3 | $0 |
| Oct 28 05:09 | `a-startup-check-3` (40636359093): ok; container environment `main` seen; guard says "not deployed" from the ops app | none | 1 | under $0.01 (estimate) |
| Oct 28 05:11 | `a-deploy-main-3` (40636474757): deployed 05:11:36, stopped 05:14 | none | 3 deployed | under $0.01 (estimate) |
| Oct 28 05:11 | `a-smoke-l4-main-3` (40636560937): the job ran end to end (fetch, map, Muse Spark map, plan, 3 probes, record), but every probe failed at sandbox creation in 0.01 s | none (no sandbox started) | 1.2 | Muse Spark $0.020047 (measured, 4,937 in and 3,265 out tokens); $0 GPU |
| Oct 28 05:13 | `a-stop-main-3` (40636720429): out/stop.json `stopped: true` | none | 0.4 | $0 |
| Oct 28 05:49 | `a-startup-check-4` (40640178518): ok, a real CPU sandbox from the worker's factory ran a 9-step probe | none (CPU sandbox) | 1.2 | under $0.01 (estimate) |
| Oct 28 05:51 | `a-deploy-main-4` (40640317696): deployed 05:51:03, stopped about 05:59 | none | 8 deployed | under $0.01 (estimate) |
| Oct 28 05:51 | `a-smoke-l4-main-4` (40640376141): LoRA zoo script, upload to record in 6 min 45 s (05:51:32 to 05:58:17) | L4 | 6.4 of probes (386 s measured) | GPU $0.0858 (measured probe time x L4 list $0.000222/s); Muse Spark $0.017833 (measured tokens); sandbox CPU and memory about $0.10 (guess rate, see below) |
| Oct 28 05:59 | `a-stop-main-4` (40641112810): out/stop.json `stopped: true` | none | 0.4 | $0 |
| Oct 28 18:52 | `a-startup-check-5` (40749121792): ok; the worker sees flags env `production`, `engine.memory` false | none (CPU sandbox) | 1.5 | under $0.01 (estimate) |
| Oct 28 18:54 | `a-gt-lora-l4` (40749433567): ground truth, LoRA script, 210 bare + 60 recorder steps. No result: the workflow's 45-minute limit ended `modal run` (exit 124) with nothing printed, and the L4 function kept running until `a-stop-ops-main` | L4 | about 47 (18:54 to 19:41) | about $0.63 GPU at list (47 min x $0.80/h, estimate); CPU and memory on Modal's bill |
| Oct 28 18:54 | `a-gt-nanogpt-l4` (40749437887): ground truth, nanoGPT script, 1000 bare + 200 recorder steps. Same: no result, ran until stopped | L4 | about 47 | about $0.63 GPU at list (estimate) |
| Oct 28 19:41 | `a-stop-ops-main` (40756468356): both ground-truth apps were still `ephemeral` after the workflow ended; stopped by id, out/stop.json `stopped: true` | none | 0.5 | $0 |
| Oct 28 22:49 | `a-startup-check-6` (40782894652): ok | none (CPU sandbox) | 1.5 | under $0.01 (estimate) |
| Oct 28 22:51 | `a-gt-lora-l4-2` (40783074654): ground truth again, LoRA 120 bare + 40 recorder steps in a 30-minute budget. No result: nothing printed in 40 minutes and the workflow limit ended `modal run` (exit 124); not even the measurement's first stage line arrived, so the function may never have started. App `ap-smlQhBII0aKV0BnMwULRzO` | L4 (if it started) | at most 50 (Modal's own limit on the function) | at most $0.67 GPU at list (estimate); Modal's bill says whether it ran |
| Oct 28 23:32 | `a-stop-ops-main-2` and `-3` (40788021207, 40788111609): never started. GitHub: "recent account payments have failed or your spending limit needs to be increased" | none | 0 | $0 |
| Oct 31 00:52 | `a-app-logs-gt-lora-2` (41152518808): log read of `ap-smlQhBII0aKV0BnMwULRzO`, refused: Modal no longer lists the app | none | 1 (runner) | $0 Modal; GitHub Actions minutes |
| Oct 31 02:33 | `a-app-logs-gt-lora-2b` (41160545127): read allowed through its remote-runs record; 0 lines (Modal fetches only the last hour by default) | none | 1 (runner) | $0 Modal; Actions minutes |
| Oct 31 03:3x | `a-app-logs-gt-lora-2c`: read over the run's own window (22:46 to 00:31 UTC, Oct 28): 0 lines, no error | none | 1 (runner) | $0 Modal; Actions minutes |
| Oct 31 03:4x | `a-app-logs-calibrate`: the same read on `a-startup-check-6`'s app (same day, known to have run containers): also 0 lines. So the empty reads say nothing about the rerun: Modal returns no logs for these stopped temporary apps two days on. Where the 45 minutes went stays unknown; round two's unbuffered `remote-run` logs and the ground truth's own progress lines are how the next run will show it | none | 1 (runner) | $0 Modal; Actions minutes |
| Oct 31 09:59 | `a-startup-check-7` (41190958484): ok | none (CPU sandbox) | 1.5 | under $0.01 (estimate) |
| Oct 31 10:01 | `a-gt-lora-l4-3` (41191057139): LoRA ground truth, 120 + 40 steps, 30-min budget. Modal queued the L4 function for 25 minutes ("waiting to be scheduled on a GPU_L4 worker", 10:03 to 10:28), then it failed at once: "/root/zoo/finetune_lora_hf has no train.py" (the probe image has no zoo; fixed: the files now come from the checkout) | L4, about 10 s | about 25 waiting, 27 in all | under $0.01 GPU (seconds on the L4; no container while queued) |
| Oct 31 10:30 | `a-app-logs-gt-lora-3`: the read above, with timestamps (the app was still listed) | none | 1 (runner) | $0 Modal; Actions minutes |
| Oct 31 11:30 | `a-gt-lora-l4-4` (41197146925): 120 bare steps ran (median 1.79 s, peak 17.78 GB allocated, 23.21 GB reserved); the recorder half failed at once on root's dataset locks in the shared HF cache (fixed, 82d6404) | L4 | about 26 queued, 6.5 on the L4 | about $0.09 GPU (estimate) |
| Oct 31 20:51 | `a-startup-check-8` (41240812675): ok, both images; CPU probe 9 steps as uid 61000 | none (CPU sandbox) | 2.3 | under $0.01 (estimate) |
| Oct 31 20:54 | `a-gt-lora-l4-5` (41241033277): bare run OOM at step 200, the first eval, at 22.13 GB (20.61 GiB) allocated, 23.37 GB reserved; the Oct 28 probe caught it at 21.9 GB (20.4 GiB). Recorder 5 + 40 steps ok; hook overhead median 1.010 (95% CI 1.004–1.017, 20 pairs), synced phase steps 1.084 (0.99–1.20, 19 pairs) [measured] | L4 | about 20 queued, 11 on the L4 | about $0.15 GPU at list (estimate) |
| Oct 31 21:26 | `a-gt-grpo-l4` (41243558415): GRPO smoke, OOM before the first step in both runs (20.91 GiB allocated, asked 2.32 GiB more); fp32 policy and reference with 16 completions x 152k vocab, likely [inference] | L4 | about 2.5 on the L4 | about $0.03 GPU (estimate) |
| Oct 31 23:02 | `a-agent-sample-1` (41250821790): D's agent eval, 30 of 500 tasks through the proxy (our key outside the sandbox), all 30 done in 1–4 turns, 0 cached input tokens, 470 calls refused past the sample, script wall 49 s | none (CPU sandbox + worker) | about 3 | **$0.4289 of Muse Spark calls** [measured by the proxy at the price table's rate]; CPU under $0.02 (estimate) |
| Nov 1 01:04 | `a-gt-grpo-l4-2` (41259481848): GRPO smoke at 1 prompt (8 completions), fp32 weights, Adafactor: OOM again before the first step in both runs: 16.36 GiB allocated plus 3.95 GiB reserved but unallocated (fragmentation), asking 2.00 GiB more; peak allocated 19.88 GB (18.5 GiB) [measured] | L4 | about 3.5 on the L4 | about $0.05 GPU (estimate) |
| Nov 1 01:19 | `a-gt-grpo-l4-3` (41260610897): GRPO smoke **fits** with expandable_segments: 4 bare + 5 recorder steps ok, peak 21.62 GB (20.14 GiB) allocated, 22.99 GB (21.41 GiB) reserved. Phase split rollout 95.3%, backward 2.4%, forward 1.2%, reference 0.8%, optimizer 0.2% [measured]. Per-prompt pass: 7 of 160 completions correct, 4.4% (Wilson 95% 2.1–8.8%); 5 of 20 prompts had any correct answer; 15 of 20 (75%, 53–89%) had zero reward spread, all of them all-wrong [measured]. Truncation not measured: TRL passes the 512 cap in generation_config, which the recorder didn't read (fixed, a test added); longest completion 512, p95 426. No overhead figure (3 steps) | L4 | about 17 queued, 9 on the L4 | about $0.12 GPU at list (estimate) |
| Nov 1 03:32 | `a-agent-builds-1` (41269611387): D's agent eval, 100 more of 500 MBPP tasks (the Oct 31 record's 30 skipped), all 100 done in 1–4 turns (1: 28, 2: 29, 3: 38, 4: 5), 188 tool calls, 0 errors, 0 cached of 659,276 input tokens, longest context 10,383 tokens; per-task wall (whole task, tools included) median 4.1 s, p95 7.6, max 9.2; script wall 127 s. **Planted tool loop never fired**: no stall, so no sampled build failed run_tests; runaway 0 of 130 sampled so far, one-sided 95% upper bound 2.3% (0/100 alone: 3.0%) [measured, Clopper-Pearson]. Per task median $0.0096, mean $0.0124 ($0.0052–0.0353); all 130 tasks' mean $0.0128 × 500 ≈ $6.4 for the whole run [derived] | none (CPU sandbox + worker) | about 3 | **$1.2371 of Muse Spark calls** [measured by the proxy]; CPU under $0.02 (estimate) |
| Nov 1 04:28 | `a-startup-check-5` (41273207191) ok; `a-deploy-main-5` (41273278185) deployed the runner from ws-a 5344a9b at 04:30 | none | about 2 | under $0.01 (estimate) |
| Nov 1 04:30 | `a-smoke-eval-sweep-l4` (41273326647): D's eval sweep through the runner, record 01M6ANZ653VDK1TM5W9CDHFAP7. **No measurement**: its one probe ended with "The probe sandbox ended without a result" after 1,280 s (04:30:44 to 04:52:05). Static finding read.eval.activation_dump only; no truncation figure. Likely [inference]: about 14 min waiting for an L4, then the sandbox exec's 450 s limit (the 360 s probe timeout + 90 s slack) killed it before a result line was printed; the 1.5B model isn't prefetched, so its 3.5 GB download sits inside the probe's 360 s. Not confirmed: the runner's logs can't be published | L4 | 21 counted, of which about 7.5 likely on the GPU [inference] | record: $0.2845 GPU (counts the whole 1,280 s, queue included) + $0.3379 sandbox CPU and memory (estimate at the $0.95/h guess rate) + $0.0173 Muse Spark (pre-gate runner, our zoo script) = $0.64 by the record, over the $0.40 stated; GPU billed is likely about $0.10 (estimate) |
| Nov 1 04:52 | `a-stop-main-5` (41274735821): runner stopped | none | 1 | under $0.01 (estimate) |
| Nov 1 05:40 | `a-startup-check-5`, `a-deploy-main-5` (41277823019, 41277897291) and an **unintended** `a-smoke-eval-sweep-l4` rerun (41277947615): my batch script reused the head of the earlier one. Cancelled at 05:47 and the runner stopped (`a-stop-main-6`, 41278250717); its job had reached probing at 05:42:44, most likely still waiting for an L4. It showed the Muse Spark gate holding: no map refinement | L4 (if one came) | at most 5 | at most $0.15 (estimate), counted against the night's $3 |
| Nov 1 05:48 | `a-prefetch-main-3` (41278305703): DeepSeek-R1-Distill-Qwen-1.5B, all-MiniLM-L6-v2, timm resnet50.a1_in1k, and the gsm8k, natural-questions and beans datasets into `preflight-weights`; `a-startup-check-7/8`, `a-deploy-main-7/8` ok | none (CPU) | about 6 | under $0.05 (estimate) |
| Nov 1 05:51 | `a-smoke-retrieval_st-l4` (41278502598): never reached Modal, the workflow refused the name (underscores); runner stopped by `a-stop-main-7` | none | 0 | $0 |
| Nov 1 05:54 | `a-smoke-retrieval-st-l4` (41278712794): D's retrieval script. **No measurement**: requirements.txt didn't install. Under the probe's uid, pip walked an unreadable PYTHONPATH entry (/root) and raised PermissionError (fixed: b60af69, pip -I). Points 2 and 3 skipped by the rule. Static finding read.retrieval.small_batch only. Runner stopped (`a-stop-main-8`) | L4 | about 0.1 (6 s) | $0.0031 by the record (GPU + sandbox estimate) |
| Nov 1 06:04 | `a-startup-check-10`, `a-deploy-main-10` ok (runner from ws-a da80524, the pip fix in); `a-smoke-retrieval-st-l4` (41279387340): **no measurement**, job cancelled at its 15-minute limit while still in its first probe (record 01M6AVDKBTR3BN0PFTKSXQJ73A has no probe and $0 of cost); most likely the whole 15 minutes in Modal's queue for an L4 [inference], as the earlier runs waited 14 to 26 min. Fix: fall back to A10, then L40S, after 2 min (66bb070). Runner stopped (`a-stop-main-10`) | L4 (if one came) | 15 in the job, likely all queued | $0 by the record; at most about $0.44 had it run the whole time (estimate) |
| Nov 1 10:53 | `a-startup-check-11` (stray, from a stopped script; ok) and `a-startup-check-12`, `a-deploy-main-12` ok: the runner from ws-a f8d6d14, GPU fallback in | none | about 3 | under $0.01 (estimate) |
| Nov 1 10:56 | `a-smoke-retrieval-st-l4` (41299583623): D's retrieval, record 01M6BC0YKBCMC4GHNG33NQT111, all 3 points ok on an L4: batch 8 / 4 / 2 peak 0.59 / 0.49 / 0.48 GB, p50 41 / 40 / 40 ms [measured]; static read.retrieval.small_batch | L4 | 0.9 of probes | **$0.0276** by the record ($0.0128 GPU + $0.0148 sandbox estimate) |
| Nov 1 11:01 | `a-smoke-gnn-pyg-l4` (41299933798): D's GNN, record 01M6BC9ZSFCS72Y4AQ11EBFYTK, all 3 points ok on an L4: batch 1,024 / 512 / 256 peak 2.62 / 1.96 / 1.88 GB, p50 98 / 82 / 72 ms; step 60% other (the hand-written sampler), 38% backward [measured]. No graph counts: the script has no loader (known limit) | L4 | 1.0 of probes | **$0.0302** ($0.0140 GPU + $0.0162 sandbox estimate) |
| Nov 1 11:03 | `a-smoke-vision-timm-l4` (41300131134): D's vision, record 01M6BCEJ68ED0F4AP011M4FWSP, p1 ok on an L4: batch 64 at 224 px, peak 3.32 GB, p50 175 ms, data 52% of the step, channels_last 0 [measured]; static read.vision.cpu_augment. p2 and p3 skipped: the job's 15-minute wall, which counts the GPU wait and the timm install, was spent | L4 | 0.6 of probes | **$0.0173** ($0.0080 GPU + $0.0093 sandbox estimate) |
| Nov 1 11:19 | `a-stop-main-12` ok; `a-agent-resample-30` (41301359530): D's agent eval on its plant-3 fix (run.py matching ws-d, sha256 8278748b), the first 30 MBPP tasks again. **Stall detection measured**: the script stopped 120 s after the last model call (script wall 257 s); 24 tasks ended on a tool call, up to 4 of them stuck (the script's 4 workers): 4 of 30 is 13% (Clopper-Pearson 95% 3.8–30.7%) if all four are, as the plant intends [measured / derived]. Turns 1–4, per-task wall median 4.5 s, max 10.0 s; the stuck tasks read `done` (no cut stop reason until schema 0.5.1). Sample in zoo/agent_eval_spark/sample_nov1_resample.json | none (CPU sandbox + worker) | about 5 | **$0.4155 of Muse Spark calls** [measured by the proxy]; per task $0.0053–0.0394 |
| Nov 1 11:24 | `a-gt-grpo-50` (41301761572): the 50-step GRPO ground truth. **No measurement**: Modal queued the L4 function for 59 min (container up 12:24), and the workflow's 60-minute limit ended `modal run` a minute into the bare run (exit 137). `a-stop-ops-main-2` found the ops app already stopped. The queue's second half stopped here (torchtitan and 2×L4 not run) | L4, about 1 min | 60 (59 queued) | about $0.01 GPU (estimate); Actions minutes |
| Nov 1 12:53 | `a-startup-check-14` (41308614408) ok; `a-deploy-main-14` (41308698051): **the production runner, deployed and left running** (ws-a 9e46c3c, main merged in, production flags; the job store about $0.19/day, C's approved deploy plan) | none | — | about $0.19/day (C's estimate) |
| Nov 1 12:44 | `a-gt-grpo-50-b` (41307865340): **the 50-step GRPO ground truth, on an A10** (GPU list L4, A10; Modal gave the A10 within about 20 s). Bare 3 steps in 99 s; recorder 2 warmup + 50 timed steps ok in 1,415 s, p50 27.6 s a step. Phase split rollout 93.5%, backward 3.1%, forward 1.6%, reference 1.3%, optimizer 0.4%. Peak 22.29 GB allocated, 23.18 GB reserved, of the A10's 24 GB. Zero spread: 167 of 208 prompt groups (80%, Wilson 95% 74–85%), the `rl.zero_spread` finding on the record. Correct completions 63 of 1,664 (3.8%, 3.0–4.8%); 40 of 208 prompts had any. Truncation at the 512 cap: 22 of 1,664 (1.3%, 0.9–2.0%); p95 363 tokens. Recorder overhead: sync 1.05 (0.83–1.25, 24 pairs), hooks 1.06 (1 pair) [measured]. Record `zoo/grpo_trl/record_nov1_a10.json` (01M6BKQTWZQ8HSDE2CAV9JTWFP) | A10 | about 26 | **about $0.52 GPU at list** (estimate) |
| Nov 1 13:10 | `a-torchtitan-check` (41309914515): startup check ok (torch 2.7.1, debug model), then the check **failed before the first step**: `KeyError: getpwuid(): uid not found: 61000`. The probe's uid has no passwd entry and the script asks for its user name. Fixed (the probe sets USER and LOGNAME, and the images get a passwd entry); not rerun, since the fix needs a push. Modal gave the A10 (the log's "on the L4" was a fixed label, fixed). The queue stopped here, so the 2× check didn't run | A10 | about 1 | about $0.01 GPU (estimate) |
| Nov 1 15:58 | `a-startup-check-15` (41323850105) ok; `a-deploy-main-15` (41323942603): **the production runner redeployed** from ws-a 4cc6f57 (main merged in, production flags), with the probe-user fix; the 12:55 deploy (9e46c3c) would have crashed visitor scripts that ask for their user name (torch.compile, vLLM). Left running | none | — | the job store, about $0.19/day (C's estimate) |
| Nov 1 16:02 | `a-torchtitan-check-2` (41324243638): **torchtitan working under the recorder**, on an A10 (GPU list L4, A10, L40S). torchtitan 0.1.0 on torch 2.7.1, debug model (6.27M params): 15 steps ok, p50 26 ms; phase split backward 52%, forward 32%, data 11%, optimizer 4%; peak 0.42 GB [measured]. Not read: tokens and samples per step came out 0 (the recorder doesn't see torchtitan's batch shape) | A10 | about 1.5 | about $0.02 GPU (estimate) |
| Nov 1 16:04 | `a-multi-gpu-2x` (41324369861): **DDP and FSDP on 2 × A10** (GPU list 2 × L4, 2 × A10), the 134M-parameter MLP fixture, both ranks reporting: DDP 15 steps ok, p50 72 ms, peak 3.24 GB a rank; FSDP 15 steps ok, p50 87 ms, peak 1.36 GB a rank [measured]. Under FSDP `params_total` reads 67M, one rank's shard of the same 134M model (known limit, told B) | 2 × A10 | about 1 | about $0.04 GPU (estimate) |
| Nov 1 16:05 | `a-prefetch-serving` (41324593868): Qwen2.5-7B-Instruct and dolly-15k into the weights volume, CPU; `a-startup-check-serving` (41324754966) ok (vLLM 0.6.4.post1 on torch 2.5.1, and vLLM imports as the probe user) | none (CPU) | about 4 | about $0.05 counted (CPU estimate) |
| Nov 1 16:10 | `a-smoke-serving-vllm` (41324883012): the vLLM serving probe on the production runner, **error before the first step**: `pyarrow requires NumPy 2.0 or newer, found 1.26.4`. datasets 3.1 pulled the newest pyarrow; the serving image pins numpy below 2 for vLLM. Fixed: pyarrow pinned to 17.0, and the serving startup check imports datasets. The error read "not installed", now it names both versions. The runner stayed up | not reached | about 2 | **$0.0253** by the record ($0.0026 GPU, $0.0013 sandbox, $0.0214 model calls) |
| Nov 1 16:12 | `a-pareto-muse-spark` (41325108987): the Pareto sweep's Muse Spark arm, the first 30 MBPP tasks of zoo/agent_eval_spark through the proxy, rows labelled from the datasets-server (30 of 30). Clean rows (0-based 0–25): **23 of 26 ended on a tool call, the proxy's reading (88%, Wilson 95% 71–96%)**, an upper bound on passes; the run stalled, so up to 2 of those may be stuck in a failing tool rather than passed (4 workers, 2 accounted for by planted rows): at least 21 of 26 (81%, 62–91%). $0.0131 a clean task, 22,147 output tokens over the clean rows. Planted rows 26–29 (planted: never passes): 2 of 4 ended on a tool call, the hang D planted [measured by the proxy / derived] | none (CPU sandbox + worker) | about 4 | **$0.3876 of Muse Spark calls** [measured by the proxy], cap $0.55 |
| Nov 1 16:16 | `a-pareto-spark-open` (41325436143): the Spark Open arm (Baseten, `meta-models/muse-spark-open-1026`), the same 30 rows. Clean rows: **14 of 26 ended on a tool call, the proxy's reading (54%, 35–71%)**, an upper bound on passes; up to 1 may be stuck: at least 13 of 26 (50%, 32–68%). $0.0110 a clean task, 26,965 output tokens over the clean rows. Planted rows 26–29: 3 of 4 ended on a tool call [measured by the proxy / derived] | none (CPU sandbox + worker) | about 4 | **$0.3252 of Spark Open calls** [measured by the proxy], cap $0.45 |
| Nov 1 16:20 | `a-startup-check-16` (41325841406) ok, with the new probe-user step: in the probe and serving images (rebuilt with gcc and g++), as uid 61000 `preflight-probe`, getpass, passwd and group resolve, a tiny torch.compile runs (7.274 = 8 sin 2), vLLM and datasets import; `a-startup-check-serving-2` (41326174732) ok; `a-deploy-main-16` (41326298472): **the production runner redeployed** from ws-a 20ebb7a, gated on both checks. Left running; C told it can deploy the API | none (CPU) | about 6 | about $0.05 CPU (estimate); the job store about $0.19/day |
| Nov 1 16:27 | `a-smoke-serving-vllm-2` (41326407788): the vLLM serving probe on the redeployed runner, record 01M6BZ0C74ZZYJ6579G8Z91SQW: **timeout at 600 s before any measurement**, on an L40S (by its price). Cause, from the code: the recorder timed vLLM's generate whole, and zoo/serving_vllm's one call is 1,024 prompts x 4,096 tokens (about half an hour), and a probe with no optimizer steps ended as too_few_steps anyway. Fixed in ws-a 16d5cd6 (a serving probe measures one sampled call of 32 prompts and ends there), checked through a fresh install with a stand-in vLLM; not rerun (needs the redeploy) | L40S | about 11 | **$0.4986** by the record ($0.3256 GPU, $0.1586 sandbox, $0.0145 model calls) |
| Nov 1 11:47 ET | Quickstart check (local, no GPU): wheel built, a fresh 3.11 venv, `pip install preflight[probe]` (torch from PyPI, 9 min), then `preflight demo` in an empty folder: 3 probes ok on the CPU in 31 s (batch 64 / 32 / 16, p50 14 / 12 / 11 ms) [measured]. Found and fixed: as root, probes dropped to a uid that couldn't reach the venv and ran without torch | none | — | $0 |
| Nov 1 17:36–18:04 | `a-startup-check-17` (41332365957) and `a-startup-check-serving-3` (41332579612) ok on f46fad2, deploy held for the flag decision; `a-startup-check-18` (41334281206), `a-startup-check-serving-4` (41334482001) ok on 8154268; `a-deploy-main-18` (41334637660): **the production runner redeployed** from ws-a 8154268 (round five, runner.queue and c.guide.live on, triage on, the serving-probe and CI fixes). Left running; C told | none (CPU) | about 12 | about $0.10 CPU (estimate); the job store about $0.19/day |
| Nov 1 18:06 | `a-smoke-serving-vllm-3` (41334716223): the serving rerun on the redeployed runner, record 01M6C4K2PZKQKPV894ZPHYN4Y7, L4 (GPU list L4, A10): **failed**, the probe sandbox ended without a result after 1,284 s (past the probe's 600 s, so the sandbox itself was ended); the record has no log, so the cause isn't known. Triage (Spark Open) called it unclear, $0.0024 | L4 | about 21 | **about $0.64** by the record ($0.2854 GPU, $0.3389 sandbox estimate, $0.0195 model calls) |
| Nov 1 18:28 | `a-triage-lora-prod` (41336726935): **the production triage run**, finetune_lora_hf, record 01M6C5WRVEM3RTRRBDRPAAPEGT (share jgk05gwUgV-Bo78sErzKFQ): p1, p2, p3 ok; p1-eval oom.eval as planted (530 MiB asked, 486 MiB free of 22.06 GiB). 4 Spark Open triage calls on Baseten, all ok, $0.0184 in all; the eval probe's finding names node eval. It said "unclear" instead of quoting the OOM line (the OOM is in the probe's recorded error, not stdout): fixed in b300684, live at the next deploy | L4 (228 s GPU wait) | about 10 | **$0.2077** by the record ($0.0788 GPU, $0.0926 sandbox estimate, $0.0363 model calls) |
| Nov 1 18:40 | `a-diagnose-lora` (41337801746): Spark diagnosis of that record, ops app: both Muse Spark calls cut off at 4,000 output tokens (high effort's reasoning used them up), no findings | none (CPU) | about 1 | **$0.0416** of Muse Spark |
| Nov 1 18:44 | `a-diagnose-lora-2` (41338042440): again at medium effort, 8,192 tokens: one finding, consistent (the loss plot's p1 peak at steps 8–10 is loss[9] = 1.563, the series' maximum, checked against the record); Spark Open's check fetched the path and agreed. The first call still ran to 8,192 tokens and the re-ask answered | none (CPU) | about 1 | **$0.0685** ($0.0593 Muse Spark, $0.0091 Spark Open) |
| Nov 1 18:53 | `a-smoke-eval-sweep-2` (41338293000): the eval-sweep probe, record 01M6C6YPNC7BZAZ82TAY4B9MSD: **timeout at 600 s, nothing kept** (492 s GPU wait first). The log shows batch after batch of generate set up: a generation-only probe had nothing to stop it and the recorder's stats are written at the end. Fixed in fda1762 (an eval probe measures two generate calls and ends as the third starts), checked with a stand-in; rerun after the next deploy | L4 or A10 | about 18 | **$0.3163** by the record ($0.1336 GPU, $0.1587 sandbox estimate, $0.0240 model calls) |
| Nov 1 20:16 | `a-startup-check-19` (41346004770), `a-startup-check-serving-5` (41346187791), then `a-deploy-main-19` (41346314893): **the gated redeploy, from ws-a d73b4e1 (main's 6757667 in it)**, after C's production deploy. Worker, probe, sandbox, the probe user in all three images (torch.compile and the vLLM import as uid 61000) ok; deployed 20:20 UTC with the demo limits, the eval and serving probe stops, the warm pool off until Nov 5 | none (CPU) | about 4 | about $0.03 CPU (estimate) |
| Nov 1 20:20 | `a-serving-debug-1` (41346358580): the serving debug run, `ops_app.py::ground_truth --script serving_vllm` (Chiya's yes), printing vLLM's log. Modal gave the A10. The sampled probe ran ok in 314 s: weights 14.25 GB loaded in 6 s, graphs 15 s, then **32 prompts x 4,096 new tokens each (all truncated, about 131k tokens) at about 480 tokens/s = 272 s**, with KV-cache preemption (the cache fits 2.19x of a 32k request) [measured]. On an L4 (about half the bandwidth) that decode is about 550 s [estimate], past the probe's 600 s with startup: the cause of the L4 failure at 18:06. Fix options (a) cap the probe's new tokens at 1,024, (b) 8 prompts: posted to Chiya, asked of B in from-a.md. The bare run (the full 1,024-prompt burst) stopped at its 420 s share, as planned | A10 | about 13 | about $0.25 (estimate) |
| Nov 1 20:45 | `a-triage-glimmer-1` (41348458748): the first Glimmer triage, `ops_app.py::triage` on the production LoRA record's p1-eval (oom.eval): **no finding and no call in provenance** after 150 s. The client logged nothing because a reply that isn't a completion (not JSON, or no choices) was raised before it was recorded; fixed (such a reply is now a logged, priced call, and the run reports why). Rerun next | Glimmer's A100-40GB (B's app) | about 2.5 | about $0.25 (estimate: B's cold start; the client recorded $0) |
| Nov 1 20:59 | `a-triage-glimmer-2` (41349583805): the rerun, with the call now logged: **HTTP 303 after 150 s**, empty body. Modal's web endpoints answer a request that runs past 150 s with a 303 to poll for its result, and the client didn't follow redirects. Fixed: the Glimmer client follows up to 3, on the same host only (its Modal-Key and Modal-Secret headers never go elsewhere) | Glimmer's A100-40GB (B's app) | about 2.5 | **$0.0876** recorded (150 s at the A100-40GB rate) |
| Nov 1 21:10 | `a-triage-glimmer-3` (41350601567): with redirects followed, **Glimmer answered** (371 output tokens after a 459 s cold start) but the reply didn't validate against the Triage schema, nor the re-ask's (6 s, warm): no finding. Next: take the JSON object out of the reply, keep the schema error in the report; one more try on Nov 2's budget | Glimmer's A100-40GB (B's app) | about 8 | **$0.2717** recorded (2 calls) |
| Nov 1 23:43 | `a-triage-glimmer-4` (41363453459), under Chiya's Glimmer $3 (6 PM): the reply was JSON now, but **not the schema**: category `oom.eval` (the probe's error code, copied) and fields missing, twice; Glimmer on vLLM has no guided decoding, so it never saw the response_format schema. Fixed: the triage prompt spells out the answer's JSON and its categories (7a9b68c) | Glimmer's A100-40GB (B's app) | about 6 | **$0.1955** recorded (2 calls: 330 s cold, 5 s warm) |
| Nov 1 23:49 | `a-pareto-glimmer` (41363963718): **the Pareto's third arm**, Muse Glimmer on B's Modal app through the proxy (Modal-Key headers, priced by GPU time), the same first 30 MBPP rows, labelled 30 of 30, scored as the other two arms. Clean rows: **15 of 26 ended on a tool call (58%, Wilson 95% 39–74%)**, an upper bound on passes; the run stalled with up to 2 clean rows stuck: at least 13 of 26 (50%, 32–68%). Planted rows: 2 of 4 on a tool call. 25,787 output tokens over the clean rows. Against Muse Spark 23/26 ($0.3876) and Spark Open 14/26 ($0.3252) on the same rows [measured by the proxy / derived] | Glimmer's A100-40GB (B's app), CPU sandbox | about 7 (script wall 376 s) | **$0.5383** by the proxy (each call's wall time at $2.10/h, summed: 4 calls share the GPU, so this overcounts); the GPU's own time for the 376 s run is about $0.22 [derived], plus any cold start |
| Nov 2 00:00 | `a-triage-glimmer-5` (41364641243): **the Glimmer triage works**, with the prompt that spells out the answer (7a9b68c): on the production LoRA record's p1-eval, finding `triage.oom`, "Muse Glimmer on probe p1-eval: oom", "The eval pass ran out of GPU memory after 3 optimizer steps", quoting the recorded line "Ran out of GPU memory during the eval pass." (checked by code to be in the text); one call, 394 in / 191 out tokens, 411 s of it the cold start [measured] | Glimmer's A100-40GB (B's app) | about 7 | **$0.2398** recorded; Glimmer under the 6 PM $3: $0.97 in all |
| Nov 2 01:20 | `a-triage-agreement` (41372018640): Glimmer vs Spark Open on **every distinct failed probe in the stored zoo records** (11: sample records only, none derived, no visitor arguments, no patched probe; same input, the probe's recorded error). **Same category on 9 of 11 (82%, Wilson 95% 52–95%)**; both answered all 11 and quoted a line all 11 times. The 2 disagreements: vision p2 `budget` (Glimmer timeout, Spark Open unclear) and an Oct 31 LoRA `recorder` error (Glimmer infrastructure, Spark Open unclear). Against the 3 stored Spark Open findings (all "unclear", made before triage read the recorded error): Glimmer 0 of 3, and Spark Open itself 0 of 3, so that gap is the input, not the model [measured] | Glimmer's A100-40GB, Baseten | about 10 | **$0.3652** ($0.3316 Glimmer, $0.0336 Spark Open); Glimmer under the 6 PM $3: $1.30 in all |
| Nov 2 02:57 | `a-startup-check-20` (41380688130), `a-startup-check-serving-6` (41381305538), `a-deploy-main-20` (41381442911): **the gated redeploy from ws-a 530785e** (main c084bee plus the L40S probe ceiling, Chiya 9:47 PM); deployed 03:09 UTC. Then `a-limit-check-1` (41381513249): one fresh visitor's jobs 1–3 accepted (a sample that doesn't exist: failed at fetch, no GPU), **job 4 refused with `limit.visitor_daily`**, retry_after_s 6605 (midnight ET) | none (CPU) | about 14 | about $0.03 CPU (estimate) |
| Nov 2 03:11 | `a-notebook-checkin-1` (41381671178): **D's notebook check-in through the deployed runner**: web/src/addons/notebook/fixtures/colab_lora.ipynb as kind `notebook` (visitor a-notebook-check, L4 or A10); record 01M6D3VC0R2Z7CVRWK818ZYJNB. Admitted, fetched (the notebook converted, its `!pip`/`%pip` lines as requirements), mapped (lens finetune, Muse Spark refined), 3 probes on an L4, predicted, done. Each probe ended `too_few_steps`, rightly: the fixture's loop has no optimizer (`loss = step`); the failure names its step and the manual command (`preflight run colab_lora.py`) | L4 | about 8 | **$0.0272** by the record ($0.0038 GPU and sandbox, $0.0234 model calls) |
| Nov 2 03:27 | `a-mbpp-full` (41383158427): **the full agent benchmark on Muse Spark** (Chiya's Meta credit, cap $14; PM: no overlap with the held-out set): zoo/agent_eval_spark, MBPP full test, the 470 tasks the Oct 31 record doesn't hold (rows 30–499; rows 0–29, the planted 26–29 among them, are that record's 30, since the planted rows retry forever and stall any run that reaches them). **All 470 finished; 339 ended on a tool call (72%, Wilson 95% 68–76%, 90% 69–75%)**, an upper bound on passes (no stall, so none stuck). Against the 30-task sample's clean rows: 23 of 26 (88%, 90% interval 74–95%), at least 21 (90% 65–90%): the full run sits below the upper reading's interval and inside the lower one's [measured by the proxy / derived]. **$0.01325 a task** (the sample's clean rows: $0.0131). Rows unlabelled (the datasets-server returned 502 at offset 300); counts only kept in zoo/agent_eval_spark/bench_nov2_full470.json | none (CPU sandbox + worker) | about 14 (script wall 794 s) | **$6.2288 of Muse Spark calls** [measured by the proxy]; **the 500-task benchmark's total, with the Oct 31 sample's $0.3876: $6.62**, against B's estimate $7.15 (90% $5.90–$13.71) |

**What the Oct 28 ground-truth runs most likely were (Oct 31):** today's run shows the same silence
is Modal queueing the L4 function, not a slow download or load: no container runs while it waits,
and the first line it prints comes only once it has a GPU. So the three Oct 28 runs probably billed
little or no L4 time, not the ~$1.90 estimated in the rows above (their logs are gone, so this is
an inference; Modal's bill for Oct 28 is the check). Had any of them got a GPU, it would have failed
within seconds on the missing zoo (the earliest code read an empty file set rather than raising).

### First public number (one script)

> **One script:** on our planted-bug LoRA fine-tune (Qwen2.5-1.5B on Alpaca, batch 4 x 4
> accumulation, one L4), Preflight caught the evaluation pass running out of memory in **189 s of
> probe time, $0.042 of GPU at list price** [measured], before any real run. Left alone, the run
> would have crashed at its first evaluation, step 200: about 6 minutes of training at the measured
> 1.68 s a step plus loading, about $0.09 of L4 [derived]. The saving is one failed run, not hours.

Sources: record `ICDYrfHQ2nBNbv3mlRHHMg` (probe `p1-eval`: status oom, code oom.eval, eval batch 1
of 33 at 21.9 GB (20.4 GiB) allocated on a 23.69 GB (22.06 GiB) card; probe `p1`: p50 1.68 s); `EVAL_EVERY = 200` in `zoo/finetune_lora_hf/train.py`;
L4 list price $0.80/h. One script and one run, so no interval: it is an example, not a rate.

First end-to-end run on a real GPU, Oct 28 05:51 UTC (record `ICDYrfHQ2nBNbv3mlRHHMg` on
`preflight-records`, env L4 with 23,689,363,456 bytes): the planted eval OOM was caught by the
first probe (`p1-eval`, status oom, code oom.eval, eval batch 1 of 33 at 21.9 GB, i.e. 20.4 GiB, allocated) and the
point re-ran with eval off. Measured on the L4 (step_time.p50 over the timed steps, peak allocated):

| Probe | Setting | p50 step | Peak | Probe time | GPU at list |
| --- | --- | --- | --- | --- | --- |
| p1-eval | batch 4, seq 1024, eval on | 1.36 s | 21.9 GB (20.4 GiB) in eval | 189 s | $0.0420 |
| p1 | batch 4, seq 1024, eval off | 1.68 s | 15.7 GB | 122 s | $0.0272 |
| p2 | batch 2, seq 1024 | 0.84 s | 9.6 GB | 22 s | $0.0049 |
| p3 | batch 4, seq 512 | 1.67 s | 15.5 GB | 53 s | $0.0117 |

p3 barely moved because Alpaca examples are mostly shorter than 512 tokens: the seq_len knob is a
truncation cap, so halving it changes little on this data (for B's memory fit). The read-time
findings also flagged eval.no_grad, eval.batch_large and data.no_packing. The record's gpu_usd
had folded the sandbox CPU and memory guess into the GPU figure; they are now kept apart
(breakdown `sandbox_cpu_memory_estimate`). The $0.95/h guess stays a guess until Chiya or the PM
reads Modal's bill for 05:51 to 05:59 UTC.

Third attempt, Oct 28 05:09 to 05:14 UTC: the deployed job store took its state (the guard works),
and the job went from submit to a stored record (share token `oZKPJ2pUoLBf5itdEpXWSA`) through
Muse Spark. All three probes failed as "The probe sandbox ended without a result" in 0.01 s:
`modal.Sandbox.create` or its first `exec` raised server-side (local validation passes; the text
is only in the worker's log). The startup check now also creates a real CPU sandbox with the
worker's own factory and runs the probe entry through SandboxRunner, returning the exception text,
so the next run names the cause before any GPU. Cause found by the independent review: the probe
image is resolved where `Sandbox.create` runs, inside the worker, and its `add_local_dir` source
still pointed into the repo (`/repo-not-here/python/preflight`), so the mount failed on the client
in 0.01 s. Image source paths now fall back to the installed package and the image's zoo, and the
startup check fails if any image source folder is missing in the container (it catches the
previous commit). Also found: the record's `costs.llm_usd` stayed 0
although the call carried its cost; the merge now sums the record's model calls.

Second attempt, Oct 28 04:33 to 04:39 UTC: the startup fix held, but every `modal run` of
`modal_app.py` (the startup check, prefetch, smoke) started a temporary copy of the app, and that
copy's job store (min_containers=1) claimed the shared job state, so the deployed store refused its
own saves and the smoke submit failed. The app list showed six `preflight-runner` entries. All
`modal run` entries now live in `runner/ops_app.py` ("preflight-runner-ops": no job store, no
schedule), the startup check also loads `modal_app.py` standalone from a shallow path as Modal
does (it fails on the Oct 27 file with IndexError: 3), and a test keeps `modal_app.py` free of
`modal run` entries. An independent review found the runner could still be started as a copy
(`modal run ...::tick`, `modal serve`, `modal shell`), so the job store now claims the shared
state only in the deployed app's own containers (`deployed_here`, fails closed) and a copy refuses
every call; `stop_app` also stops live temporary copies by id. The weights from prefetch are on
the volume and stay.

Failed sitting, Oct 28 02:39 to 03:07 UTC: `JobStoreService` and `tick` crashed on every start
(Modal crash-loop alerts) with `IndexError` at `ROOT = Path(__file__).resolve().parents[3]`.
Modal loads a deployed file on its own as `/root/modal_app.py`, not from the repo, so the lookup
had no fourth parent; moving the package to /opt/preflight that evening didn't cause it, and the
app had never started on Modal before. The same assumption made `image_tag()` hash no files in
the container. Fixed (the repo path is read only where the repo exists, the tag hashes the
installed package and refuses to hash nothing), and `ops_app.py::startup_check` now runs both
images' startup code with no GPU before any deploy. No GPU or sandbox ran, so the $0.95/h sandbox
CPU and memory figure is still a guess until the smoke run's bill.

## Contract requests

See docs/requests/from-a.md.
