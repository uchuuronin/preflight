# Status · B (prediction engine and corpus)

Updated Tue Oct 27, evening ET, on `ws-b` (merged with A's `ws-a` at b59334b; from now on merges, not rebases).

## Done
- **Settings chosen by Monte Carlo** (`docs/decisions/engine-settings.md`, `scripts/mc_engine_settings.py`): memory calibration on A's three-probe plan, the 90% band (including one or two probes) and the fits/tight/fails cut-offs, the divergence label (D9 catches early collapses, a case a CPU smoke run turned up), cross-GPU step time and the corpus budget rule.
- **GPU table** (`engine/gpus.py`): 47 devices (40 NVIDIA, 7 AMD) covering Modal, Unity, Colab, Kaggle and AMD Developer Cloud, one row per device as PyTorch sees it (MI250X per GCD). Every row carries a confidence label (datasheet, derived or secondary) and GPU-to-GPU link bandwidth. `resolve()` handles PyTorch, ROCm `gfx`, MIG and Modal names; `peak_flops()` is a drop-in for the recorder. Prices are dated and sourced.
- **Memory model** (`engine/memory.py`):
  - Weights by dtype and eleven quantisation formats; twenty optimizers.
  - DDP buckets; FSDP and ZeRO-1/2/3 sharding with gather buffers; tensor and pipeline parallel.
  - Gradient accumulation; optimizer offload.
  - Eval, train and generate moments (KV cache, colocated vLLM); RL reference and PPO value models; MoE.
  - Calibration (E3), band (R5), abstain past ±50%, P(OOM) with the chosen cut-offs.
- **Step-time model** (`engine/steptime.py`):
  - Phase taxonomy for every schema phase, converted from A's fractions of `p50_s`.
  - Per-kernel roofline replay (K1), with a kernel-name classifier for CUDA, HIP, Triton and aten as the K4 fallback.
  - Recompute cost; ring all-reduce for multi-GPU; MFU with MoE active parameters.
- **Knob map** (`engine/knobs.py`): every knob id A's harness can declare (checked by a test that reads A's harness), with what it moves and whether B models it yet; precision setups; model-family hints; `d_ff` solved from the parameter count (Qwen2.5-1.5B: 8960, exact); work units per lens; script map to engine inputs.
- **Corpus**: failures carry A's error codes; the D9 divergence label; the B5 budget rule in both drivers. The Modal app files rows under the GPU it really ran on and stops a wave that makes no progress; the local driver refuses a `--gpu` that doesn't match the machine. Op rows carry a kernel class.
- **Independent review** (find-bugs plus the system-design resilience and task-scheduling checklists): no critical issues. All seven important findings and twelve minor ones are fixed, each with a regression test:
  - each corpus config runs in its own process with a deadline, so a hang becomes a `timeout` row and a kernel fault can't poison the next config;
  - spend survives retries;
  - the AMD driver counts billing overhead;
  - "measured" only when the caller says every knob matches;
  - out-of-range queries abstain;
  - bf16 onto a T4 abstains;
  - capacity is what PyTorch reports (L4 21.45 GiB, not 24);
  - DeepSeek-R1 distills stay dense;
  - frozen quantised models are costed quantised;
  - "unknown" precision uses the load dtype;
  - TF32 is off by default.

  The review also asked for the band with one or two probes, now simulated (S2c).
- **A's Oct 27 requests, all closed:**
  - **Kernel classes:** the classifier covers every op A saw as "other", keeps the class of the roll-up rows, and adds a `host` class (views, allocations, host syncs) that doesn't scale with the GPU. A real profiled training step has no unclassed op.
  - **Bounded GPU names:** names over 128 characters are refused, and the size pattern is bounded.
  - **MIG:** slices resolve to the right card, and the scaled names round-trip.
  - **`group_by_length`:** declared, honestly marked not yet modelled.
- **A's new schema, read by the engine:**
  - `Probe.kernels` goes straight into cross-GPU step time (`kernel_rows`, `predict_cross_gpu_from_probe`). It abstains on CPU tables, and replay is exact on the probed GPU even when an estimated byte count overshoots. K1 is unchanged at 98.2%.
  - The measured model shape (gated MLP, tied embeddings, norm, head dim, experts) beats the name hints.
  - `attn_impl` sets the attention kernel when no knob does.
- **Agent and API totals** (`engine/agent_cost.py`): `cost.total`, `tokens.total` and `risk.runaway` from A's task sample. It uses A's bootstrap plus a Clopper-Pearson runaway margin, re-checked on A's generator (S6): 89.5% at 50 tasks, 90.3% at 30.
- **Independent review of this round:** no critical issues. All six important and six minor findings are fixed, each with a test:
  - many tiny launches count as latency per launch;
  - in-place and quantised matmuls and NCCL's `record_param_comms` get the right class;
  - Gemma's head dim feeds the `d_ff` solve (exact parameter count);
  - GH200 MIG slices pick the right card;
  - dict tables are validated;
  - the agent bootstrap runs in bounded memory (17 MB at 10,000 tasks).
- **Chiya's decisions applied:**
  - **Below the smallest probe:** an "extrapolated" estimate with a growing band and a probe offer. Memory widens 25% per halving (S2d); step time widens 0.1 per halving and abstains about 2.7 halvings down (S4b).
  - **Agent sample:** 30 tasks, plus a 20-task top-up offered when the cost range straddles the budget (S6b: 88.2% final coverage).
  - **Flags:** B's four flags are in `flags.json`, off.
- **`engine/api.py`:** `predict`, `build_grids`, `sensitivity`, plus `targets_for` and `fill_predictions` (the call `make_golden.py` makes).
  - Every lens gets its readouts: GPU lenses get memory, OOM risk, step and total time, cost, the wall-cap risk and preemption loss; API lenses get cost, tokens, wall clock under concurrency and rate limits, context fill and runaway rate.
  - Knob moves the models don't capture widen the band or abstain, and "measured" means a probe really saw the peak.
  - About 0.4 ms per `predict`; a 7x7 grid of four targets takes 20 ms.
- **Independent review of `api.py`:** one critical, seven important and seven minor findings, all fixed with tests:
  - an unknown GPU no longer reads as the probe's GPU, and an OOM probe only counts on its own GPU;
  - bad precision values and sizes abstain instead of crashing;
  - throughput sensitivity is per unit of work;
  - more GPUs add the all-reduce, with its own error;
  - the agent path honours turn caps, the sampled model and unmodelled knobs;
  - the context limit below the sample abstains, and the top-up is offered once.
- **C's requests:**
  - `memory.<term>` for all twelve terms, adding up to the peak, with eval as its own wedge;
  - `time.phase.<name>` adding up to the step;
  - `sensitivity(..., pinned=, knobs=)`;
  - eval settings held in every grid's `fixed`.
- **Review, Oct 27 (top priority):**
  - `api.py` runs on the three golden records, with a 1-D grid for every knob around its current value.
  - "Measured" only where a probe observed the target: P(OOM) never, unless a probe ran out of memory there.
  - "Extrapolated" whenever a knob other than batch or sequence sits off every probe.
  - Model cards say `calibration: simulated`, and engine and corpus constants carry source tags.
  - A Pyodide parity test (it needs the CDN, so CI runs it), an import allowlist, and an unlisted-GPU fallback from the probe's own memory.
- **K1's band (S4c):** sized to cover 90% even if per-kernel drift is twice what S4 assumed. Cross-vendor drops from a hand-set 0.25 to 0.22, so AMD gets numbers with wide bands instead of always abstaining.
- **Review, Oct 27 (the rest of B's list):**
  - split-conformal intervals on log ratios, grouped by architecture, per GPU and lens, with a leave-one-architecture-out coverage check and Wilson intervals; the engine reads the table when one is built and stays `calibration: simulated` until then;
  - P(OOM) from empirical residuals, shown in bins, with a reliability diagram that decides whether it may ever show as a number;
  - corpus: grouped probe-like configs, 10 real HF LoRA and 4-bit jobs (pinned library versions, which are now part of each config's key), and `corpus.calibrate` to turn them into the table;
  - memory refinements: optimizer-step moment, autocast weight cache, FSDP root unit, accumulated eval logits, LoRA frozen layers saving fewer activations, and flash attention no longer counted twice;
  - the AMD droplet's uptime ledger: `--billing uptime` counts the droplet's whole bill, idle time included, and logs each config with the spend so far.
- **Run cost (gap 21, contract 0.2.0):** `engine.api.run_cost(record, knob_values, provider=, billing=, region=)` returns `cost_basis` and `job_cost`, and `fill_predictions` writes both, replacing the golden fixture.
  - GPU-hours x a dated price-table row, with redone work as its own line when the interruption rate is known.
  - Own cluster: GPU-hours first, with a grey market equivalent.
  - Agent and API runs: a tokens line.
  - Money at risk sits beside the total, never in it; its range spans the P(OOM) bin's edges (S7).
  - `engine/gpus.py` now reads its prices from `engine/prices`, so there is one table.
  - Storage isn't billed yet: no paid storage rows.
  - On an own cluster, money at risk leads with GPU-hours and keeps dollars as the grey market figure (Chiya, Oct 27), until the contract has an at-risk GPU-hours field.
- **Measured L4 (PM's Modal smoke run):** usable memory is 23,689,363,456 bytes (22.06 GiB, was 21.45), and the engine prefers a probe's reported total over the table for the probe's own GPU.
- Tests live in root `tests/`; 797 pass across the repo (all agents' tests included), ruff clean with the repo config.

## Blocked
- The L4 runs that build the real calibration table (grouped and HF jobs) wait on Chiya's yes to the $45 corpus plan (ask before spending, Oct 27).

## Next (Wed Oct 28)
- NVIDIA corpus slice re-planned to $45 of Modal (corpus/README.md): about $24 estimated, $35.20 in hard caps, $9.80 reserve; waiting on Chiya's yes before phase 0.
- AMD: `data/amd_grouped.jsonl` (the same 300 grouped configs as the NVIDIA GPUs) and its `amd-run` command are ready for Thursday's droplet session.
- Real held-out calibration on the L4 once Modal access lands; model cards switch to `held_out` per target as the table covers it.
- Agent cost from tokens when tasks lack `cost_usd`, once D's price table lands.

## Cloud runs and spend
Every run through `remote-run.yml` or `amd-run.yml`, with its GPU, minutes and dollars.

| Date | Run | GPU | Minutes | Dollars |
| --- | --- | --- | --- | --- |
| none yet | | | | |

Total so far: $0 of the $45 corpus budget.

## Contract requests
In `docs/requests/from-b.md`.
- **Open from B to A:**
  - one golden-record GPU name;
  - FLOP counts for in-place matmuls;
  - the reasoning-token convention;
  - the agent top-up in the probe;
  - the `make_golden.py` call, which needs 30 agent tasks in the golden record.
- **Open from B to the others:**
  - Chiya: one more flag, `engine.agent_totals`;
  - C: every target id per lens, the extrapolated label, and the top-up offer;
  - D: the price table.
- **A to B:** every request is closed except the golden predictions, which now have a reply with the exact call.
