# Status · B (prediction engine and corpus)

Updated Tue Oct 27, evening ET, on `ws-b` (merged with A's `ws-a` at b59334b; from now on merges, not rebases).

## Plan items B owns (Sat Oct 31, 9 AM ET)

Every B item in the Contracts tab (Contract B, the gaps table, the adopted items) and the main plan (feature set, prediction engine, validation targets). "Branch" means on `ws-b` but not on `main`. Hours are my honest estimate of work left, review and fixes included. They don't include waiting on A's data or on a paid run.

**Hero lenses first**

| Item | Where | Hours left | Notes |
| --- | --- | --- | --- |
| Fine-tune lens: memory, P(OOM), step time, cost, money at risk | `main`; phase 2 table and fixes on branch | 1 | Today: step-time bands for vision, retrieval and pretrain, and P(OOM) shown "uncertain (5–95%)" in the middle |
| Agent lens: per-task bootstrap, context growth, cache prices, rate-limit wall clock, runaway rate | `main` | 0.5 | Tokens-only cost when tasks lack `cost_usd`, now that the price table is in |
| GRPO / RL: zero-variance fraction p^G + (1−p)^G from `pass_rate_per_prompt`, response-length bootstrap for rollout time | not started | 3 | A's fields already exist in the schema; calibrated on A's RL probes when they land |
| Campaign planner: grid pricing, Young/Daly intervals, Slurm wall-time segments, quota check, ASHA plan, Monte Carlo completion dates, team lanes | not started (schema only) | 10 | The card says its accuracy is the engine's, because its numbers come from the engine's intervals |

**The other lenses and engine features**

| Item | Where | Hours left | Notes |
| --- | --- | --- | --- |
| Serving: KV-cache formula, prefill and decode rooflines | not started (the KV term exists in the memory model) | 3 | |
| Serving: discrete-event scheduler simulation, calibrated on A's vLLM probes | not started | 6 | Calibration needs A's vLLM probe records |
| Multi-node closed forms, labelled analytical-only | not started (single-node multi-GPU all-reduce is on `main`) | 3 | |
| Loss forecast: power-law fit, plus LC-PFN posterior, abstaining on wide bands | not started | 5 | The power law alone is 2 of these hours |
| Data repetition and quality exposure (with A) | not started | 4 | |
| Proxy campaigns | not started | 3 | |
| Power analysis: seeds needed from seed-to-seed variance | not started | 2 | |
| GBDT residual on cross-GPU step time | not started | 4 | Corpus rows are in |
| Divergence classifier: 800+ short runs, AUROC target 0.85 | not started (the corpus `divergence` kind and the D9 label exist) | 4, plus a paid sweep | The sweep needs your yes; I'll post its cost first |
| GNN cost model, used only if it beats the GBDT held out | not started | 6 | |
| Muse Spark Open batch-labels the corpus | not started | 2 | |
| Conformal layer, abstention rules, float16 grids within the cell budget | `main`; real table on branch | 0 | |
| Pyodide parity test and import allowlist (gap 10) | `main` | 0 | |
| Sample records for the library (gap 4, with C) | `main` (`fill_predictions`) | 0 | C serves them |
| Run cost and its basis (gap 21) | `main` | 0 | |
| Corpus sweep: resumable, keyed, halts at budget (gap 13) | `main`; A100 board fix on branch | 0 | |
| Versioned model cards pinned in records (gap 15) | `main` (ids like `memory-analytical@0.2.0` on every estimate) | 1 | Needs a test that an old record keeps the model it was predicted with |
| Glimmer private-mode host proposal (with A, by Sun Nov 1) | not started | 1 | |
| Corpus published with its datasheet (gap 16) | not started | 2 | |

**Validation rows (README table)**

| Row | Where | Hours left | Notes |
| --- | --- | --- | --- |
| Interval coverage ≥ 88% | branch: memory 90.1% held out on the corpus | 2 | Still to do on full runs of unseen scripts (Nov 2) |
| Peak memory within 10% | partly measured: HF jobs and A's smoke record | 1 | Scored on A's full runs |
| OOM: no misses, ≤ 1 false alarm per 10 scripts, reliability diagram | branch: reliability bins | 1 | Scored on the bug zoo |
| Step time within 15%, same GPU | branch: the line puts 461 of 575 held-out queries within 15% | 1 | Full runs still to come |
| Step time within 20%, unmeasured GPU (4th type, incl. AMD) | waiting on the A100 and AMD runs | 2 | The A100 run is going now |
| GNN + GBDT beats GBDT alone | not started | with the GNN | |
| Divergence warning: AUROC 0.85+, 100+ steps early | not started | with the classifier | |
| RL rollout time within 20% | not started | 1 | Needs A's GRPO zoo script run |
| Serving tokens/sec within 15% | not started | 1 | Needs A's vLLM full runs |
| Agent total cost inside the 90% interval | simulated on `main` (S6); no full run yet | 1 | |
| Abstention: every query outside the probed range is flagged | `main` (automated tests) | 0 | |
| Coverage self-test page (with C) | not started | 2 | |

**Total: about 82 hours of B work left.** That is more than the time left before noon Sunday. I'll work through it in the order you gave, hero lenses first, so whatever you cut on Sunday comes off the end of the list.

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
  - Money at risk also comes as GPU-hours (`at_risk_gpu_hours`, contract 0.3.0), shown first on an own cluster with dollars as the grey market figure (Chiya, Oct 27); it needs no price, so an unpriced GPU still gets it.
- **Measured L4 (PM's Modal smoke run):** usable memory is 23,689,363,456 bytes (22.06 GiB, was 21.45), and the engine prefers a probe's reported total over the table for the probe's own GPU.
- Tests live in root `tests/`; 798 pass across the repo (all agents' tests included), ruff clean with the repo config.

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
| Oct 28 | `b-corpus-smoke-grouped` (GitHub run 40622057852): failed before start, the command needed `::main` | none | 0 | $0.00 |
| Oct 28 | `b-corpus-smoke-grouped` (GitHub run 40622148184): 10 grouped configs, all ok, 16 to 34 s each | L4 | 3.5 GPU-min (7.3 min wall, image build included) | $0.06 (ledger: GPU-seconds x $0.80/h x 1.25) |

| Oct 28 | `b-corpus-smoke-hf` (GitHub run 40622922548): all 50 HF LoRA and QLoRA jobs, 49 ok and 1 out of memory (a query), 14 to 38 s each | L4 | 16.1 GPU-min (18.9 min wall) | $0.27 (ledger) |

| Oct 28 | `b-corpus-l4` (GitHub run 40633186279, Modal app ap-CTVbvqPCmY8QdS4PdaXdSA): grouped n = 600, stopped by the workflow's 120-minute timeout after 340 of 590 configs, 332 ok and 8 out of memory; per-config median 8.0 s, p90 14.3 s | L4 x 4 containers | 53 config-min; containers up for most of 118 min | ledger $0.88; billed estimate $4.50 to $6.40 (4 L4 containers for most of 2 h), waiting on Modal's figure |

| Oct 30 | `b-corpus-startup` (GitHub run 41152541453): the T4 container started in the real image and ran its check; the reply failed to unpickle on the runner (a torch object), fixed with plain values | T4 | about 1 min | about $0.01 |
| Oct 30 | `b-corpus-export` (GitHub run 41152543575): 426 L4 kernel rows back as `out/corpus/kernels/L4.jsonl.gz` | CPU | under 1 min | about $0.01 |

| Oct 30 | `b-corpus-startup-2` (GitHub run 41156645584): the app starts in its real image, one small config through the child in 5.6 s, plain values back | T4 | about 1 min | about $0.01 |

| Oct 30 | `b-corpus-export-hf` (GitHub run 41157095457): the 50 HF rows back | CPU | under 1 min | about $0.01 |
| Oct 30 | `b-corpus-l4-2` (GitHub run 41159946133): phase 1 finished, 151 configs (150 ok, 1 out of memory), 6 waves, each settled under its worst case | L4 x 4 | 25 min wall, 5,588 container-s | $1.37 (container-time bound) |

| Oct 30 | `b-corpus-a10` (GitHub run 41161812522): phase 2, 577 grouped configs (566 ok, 11 out of memory), 19 waves | A10 x 4 | about 28 min wall, 7,669 container-s | $2.60 (container-time bound) |
| Oct 30 | `b-corpus-a10-hf` (GitHub run 41163888314): 4 HF jobs ok, then it stopped itself because one more worst-case wave wouldn't fit its 20 minutes (nothing measured yet, so the 900 s deadline); rerun queued | A10 | 178 container-s | $0.05 |
| Oct 30 | `b-corpus-t4` (GitHub run 41164044206): 75 fp32 and fp16 configs (71 ok, 4 out of memory), 8 waves | T4 | 3,528 container-s | $0.65 |
| Oct 30 | `b-corpus-l40s` (GitHub run 41165087592): 195 configs (194 ok, 1 out of memory), 10 waves | L40S | 2,986 container-s | $1.78 |
| Oct 30 | `b-corpus-a10-hf-2` (GitHub run 41166235925): the other 46 A10 HF jobs, 13 waves, all 50 done (49 ok, 1 out of memory) | A10 | 16 min wall, 1,214 container-s | $0.41 (container-time bound; $0.47 with the first run) |
| Oct 30 | `b-corpus-a100` (GitHub run 41165933860): **no configs**. The containers' ~150 s cold start ate a 60 s margin in the start check, so each quit before its only config; the no-progress guard stopped the run. Fixed in `a330f9b` | A100-80GB x 3 | 576 container-s | $0.44 |
| Oct 31 | `b-corpus-a100-2` (GitHub run 41191432272, cap $2.50 total): 2 configs ok, on an **A100-PCIe-80GB**. Modal's A100-80GB landed on the PCIe board, the rows were filed under it, and the no-progress guard stopped the run. Fixed in `20e0372` (sibling boards count as done) | A100-80GB x 1 | 63 container-s | $0.05 ($0.49 of the $2.50 with the first run) |
| Oct 31 | `b-corpus-export-2` and `b-corpus-export-hf-2` (GitHub runs 41191537650, 41191862052): every kernels and HF row back | CPU | about 7 min wall | about $0.02 |

**Measured:** phase 0 and phase 1's first run, plus A's sitting and both ground-truth attempts, cost **at most $1.55 combined**, from Modal's monthly total ($1.91 for October, all credits). Phase 1's resume adds $1.37 by the bound.

Total so far: ledger $1.21 plus about $0.03 today; billed estimate $4.80 to $6.70 until Modal's usage page gives the real figure. **The ledger under-counts container time.** It sums each config's own seconds, but containers are also billed for start-up, the per-batch volume reload, a volume commit after every config, and the waits between waves. That overhead was small in phase 0's single container and dominant in phase 1 (about 80 container-seconds per config against about 9 counted). So the $6 cap was enforced on the wrong measure. Phase 2 is on hold until the budget counts container wall time. Measured per config on the L4: grouped median 18.8 s, p90 25.7 s (n = 10); HF median 16.9 s, p90 25.2 s (n = 50). The HF jobs were far quicker than the 120 s guess, so phase 0 finished phase 1's HF set too. The ledger counts the configs' GPU-seconds; Modal's bill also counts container start-up between waves, so check it on Modal's usage page.

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
