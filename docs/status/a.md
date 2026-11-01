# Status: A (probes, recorders, execution plane)

Updated Sun Nov 1, 01:30 ET.

- **Now:** tonight's approved GPU queue: D's retrieval, GNN and vision probes; then the agent re-sample on D's plant-3 fix, 50-step GRPO, torchtitan and 2×L4. $3 ceiling, each run logged under Cloud runs.
- **Done since Oct 27:** every lens's recorder and probe path, the agent proxy and sampler, run hygiene and measured findings, `JobRequest.patch`, the HF Trainer/TRL/Lightning adapter checks, and the Muse Spark script-map gate; LoRA and GRPO ground truths on L4.
- **Next:** Spark Open log triage on Baseten, a GPU fallback when an L4 won't schedule, the vLLM serving probe, the Muse Spark / Spark Open Pareto sweep; plan table below.

## Known limits, on purpose

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
| 4 | FakeTensor cross-check of the analytical memory terms (contract) | not started | 3 |
| 5 | L4 ground truth: recorder overhead, where the OOM lands (review Oct 27) | bare run measured Oct 31; recorder half failed on a cache permission (fixed locally); rerun with 210 steps needs a yes | 0.5 + paid run |
| **Agent and API runs (hero)** | | | |
| 6 | OTel GenAI proxy around any OpenAI-compatible client (lens, contract) | ws-a (`recorder/agent.py`) | 0 |
| 7 | Sampler: 30 tasks, 80% runaway bound, 20 more when the range straddles the budget (lens, D3) | ws-a: proxy run mode, pipeline, `sample_more`, CLI; local: the hosted path (proxy behind a tunnel, script in a sandbox with no secrets), first run waits for a yes; the bound is B's `agent_totals` | 1 + API spend |
| 8 | Rate-limit wall-clock (lens) | not started | 2 |
| 9 | Pareto model sweep across Muse Spark and Spark Open (lens, contract) | not started | 4 |
| 10 | Agent zoo script and an agent record on the public app (G1) | not started (D writes the script) | 2 |
| **RL post-training (hero)** | | | |
| 11 | GRPO zoo script, public model and data, 512-token completion cap planted (competitive #1) | local (`zoo/grpo_trl`); review: at 512 the truncation rate may be only a few percent on Qwen2.5-0.5B, ground truth will say | 0.5 |
| 12 | RL phase timer: rollout, reference, reward, update, weight sync (lens, contract) | local: all five; weight sync misses the ZeRO-3 gather and vLLM V1 (V0 engine path only) | 0.5 |
| 13 | Pass rate per prompt for B's zero-variance fraction (lens) | local (`recorder/rl_groups.py`, `rl.pass_rate_per_prompt`, `rl.zero_spread_fraction`) | 0 |
| 14 | Entropy, KL, length and truncation signals (lens) | partial on main (TRL logs become RL signals) | 2 |
| **Campaign (hero; B owns the model)** | | | |
| 15 | Probes for a grid's points (a campaign is many records from one script) | not started; needs B's campaign spec | 3 |
| **Other lenses** | | | |
| 16 | LM pretraining: throughput, MFU, loader probe, stability signals (lens) | main | 0 |
| 17 | LR range test (lens, contract) | not started | 3 |
| 18 | Checkpoint resume exact-match (lens, contract, hygiene) | local: what the script's torch.save is given (weights with or without optimizer state, LR schedule, RNG, step; HF Trainer's optimizer.pt / scheduler.pt / rng_state by name) becomes `hygiene.resume_mismatch`. A content check, not a resumed run replayed step for step | 3 |
| 19 | Evaluation sweeps: HF generate vs vLLM throughput, thinking-cap truncation, activation-dump storage (lens) | hooks on main; probes on D's scripts not started | 4 |
| 20 | Serving probes on vLLM for B's model (contract) | hooks on main; local: the serving image (vLLM 0.6.4.post1, its own torch) and routing; startup check and first probe wait for a yes | 3 + paid run |
| 21 | Retrieval probes through an adapter (lens) | local: encode, faiss add/train/search timed with items a second; a stage after training isn't reached by a probe | 3 |
| 22 | GNN probes through an adapter (lens) | local: nodes, edges, seeds, nodes per hop, largest batch; sampling time is the data phase; needs a PyG script on a GPU | 3 |
| 23 | Vision probes through an adapter (lens) | local: images per batch, resolution, channels-last; images/s from samples; needs a vision script on a GPU | 2 |
| 24 | DDP and FSDP on one multi-GPU Modal node (lens, contract) | not started | 6 + paid run |
| 25 | Run hygiene: stale checkpoints and caches, identical outputs from different weights, truncation rate (lens) | local: findings `hygiene.stale_checkpoint` (checkpoint dirs sent with the code), `hygiene.cache_not_keyed` (a fixed-path `if exists: load`), `generation.truncated` (over 10%, Wilson interval), `hygiene.identical_outputs` (two weight states, same tokens on every compared input). Not yet on a GPU run | 3 |
| 26 | Data repetition and quality exposure, A's half: read per-document quality scores (lens) | static detection only | 4 |
| 27 | In-flight collector (feature, contract) | local (`recorder/inflight.py`, `preflight watch`); getting the live record to the guide waits on C | 1 |
| 28 | Adapters: HF Trainer, TRL on main; Lightning and torchtitan (contract) | Nov 1, CPU probes of each framework's own loop (tests/test_adapters.py; tiny random models and synthetic data, shaped on each project's public example, since this workspace can't download from the Hub): **HF Trainer working** (transformers 5.21; also on L4 in the LoRA ground truth), **TRL working** (SFTTrainer, trl 1.17; GRPOTrainer on L4 in `a-gt-grpo-l4-3`), **Lightning working** (2.6.6; its planned run length now read from Trainer.fit). All three: steps counted, phase split, planned steps 40 of 40. **torchtitan not confirmed**: 0.4.0's trainer defaults to CUDA and doesn't import in a CPU-only install, so it needs a GPU run (and a tokenizer and dataset the wheel doesn't ship) | 3 of 4 |
| **Muse Spark** | | | |
| 29 | Spark Open triage of every probe log on Baseten, every call logged (contract) | client on main, triage not wired | 3 |
| 30 | Spark Open checks each claim against the raw probe numbers (competitive #5) | not started | 5 |
| 31 | Spark diagnosis: traces and plots as images, search grounding, cited findings (contract) | not started | 5 |
| 32 | Glimmer host proposal with B (competitive #6) | not started | 2 |
| **AMD** | | | |
| 33 | ROCm probes on the MI300X, AMD kernel names (contract) | main (prepared); blocked: the AMD Developer Cloud account is on a verification hold (Oct 31 evening). Session Sunday 9 AM if it clears, else to the noon cut with AMD answers from datasheets and the cross-GPU model, labelled not measured | 2 + droplet |
| 34 | Droplet idle watchdog (contract) | not started (B's sweep has an uptime ledger) | 2 |
| **Platform and gaps** | | | |
| 35 | Job store, queue, leases, reaper, probe cache, sandbox policy, migrations, token caps, AMD offline (gaps 1, 2, 3, 6, 8, 9, 11) | main | 0 |
| 36 | Hostile-repo fixture and sandbox tests (gap 17) | main | 0 |
| 37 | LICENSE (gap 22) | main | 0 |
| 38 | 30-day retention of uploads, probe cache, job index (Chiya, Oct 30) | ws-a | 0 |
| 39 | Runner deploy with `engine.memory` on (G1) | waits for round three on main | 1 |
| 40 | Execution-plane logs and metrics (gap 14) | partial (backlog alarm, cleanup log) | 2 |
| 41 | GPU power sampling (contract) | main | 0 |
| 42 | pip-installable CLI, one-command install, 60-second quickstart (contract) | CLI on main, packaging not started | 3 |
| 43 | 8 of 9 unseen scripts harnessed, failures name the step (contract; PM picks them) | not started | 6 |
| 44 | Hash of the unseen-script list, published before Nov 2 (competitive #3) | not started (needs the PM's list) | 0.5 |
| 45 | Prebuilt images and a warm pool for judging, Nov 5–8 (contract) | not started | 3 |
| 46 | Chakra trace export (contract) | not started | 3 |
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
