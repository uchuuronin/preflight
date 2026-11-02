# Status · B (prediction engine and corpus)

Updated Mon Nov 2, about 1 AM ET, on `ws-b` (merges, not rebases, for pushed commits).

**Unseen scripts:** B has not opened `unseen-scripts.md` in the Project, and won't before Monday's scoring.

## Decisions today (Nov 2, from 11:15 AM ET; Chiya out until about 5 PM)

| # | What | Why | Cost | How to undo |
| --- | --- | --- | --- | --- |
| 1 | **Unseen scoring runs from a separate checkout of the frozen commit 56c2141** (its own environment); checked first: last night's manifest scores to the identical table there. | The freeze: every first attempt is predicted by the frozen engine and scored by the frozen scorer. | none | — |
| 2 | **CI on 3a1abf8 failed once in A's `test_agent_sampler.py::test_past_the_dollar_cap_new_tasks_are_refused_not_counted_empty`** (no 429 seen); it passed 3 of 3 locally and on a rerun of the failed job. Not B's code (that push was docs and the hash check). | Flaky timing in A's test; told here so A sees it. | none | — |
| 3 | **"Preflight vs asking a model": the prompt is committed first** (`corpus/ask_model.py`, `ask-model@1`), before A posts any unseen ground truth. Muse Spark gets the script map's files (same suffixes, entrypoint first, same 60,000-character budget), the command and the GPU, with no probes and no static-reader hint (that is Preflight's analysis, not the code). It answers peak memory (max reserved, GiB), median step time after warm-up and OOM yes/no with a probability, each number with a 90% interval, told how many optimizer steps the full run takes. Scored by `p_oom` against the scorer's 0.5 alarm level (Preflight's own rule); a malformed interval or probability counts as no answer. | The PM's item 2; the hint left out so the model gets the code, not Preflight's reading of it. | Meta credit, cap $2 | — |

## Decisions overnight (Nov 2, under Chiya's 12:16 AM autonomy note)

| # | What | Why | Cost | How to undo |
| --- | --- | --- | --- | --- |
| 1 | **Grid fix:** every 1-D grid axis (and every grid) now holds each probed value of its knob and the knob's current value, inside the 16,384-cell budget; when the extra stops push a grid over it, the unprobed stops farthest from the current value go first (0755724, review fixes 717312c). | The PM's bug: the pretrain sample probed batch 8 and 12, but its batch axis was [2, 3, 6, 12, 24, 48, 96], so the probed batch 8 read "not trusted". | none | revert 0755724 and 717312c, refill `schema/examples/pretrain.json` |
| 2 | **Records changed by the sweep:** only `schema/examples/pretrain.json` (refilled; its five 1-D batch grids gain 8, which is now trusted (memory and P(OOM) abstain only at 48 and 96; time and cost grids at 2, 48 and 96, and no longer at 3)). `campaign-grid.record.json` gains 8 too, but its stored grids already hold it, so it needed no refill. The other 14 of the 16 records (examples, web fixtures, zoo) have no off-grid probed value. | The PM asked which records changed. | none | as 1 |
| 3 | **`finetune.json` refilled once C's guide test stopped reading its numbers** (037b9e1, pushed with main's overnight merge 2 in feb780d; CI green). Its eval batch 4 now reads "fits". C asked whether the B200 move is really cheaper: the memory half is right (117 GiB at eval batch 32; the B200 and H200 clear it), the cost half isn't a real difference ($0.96, rough band $0.57 to $1.63, against the L4's $0.98), so I asked C to word overlapping cost bands as "about the same". | Held at first: the refill broke C's "says honestly that nothing safe can be moved" test; C rewrote the test (acf4cf9) and the refill waited until main had it, so `ws-b`'s CI stayed green. | none | revert 037b9e1 |
| 4 | **Validation readiness:** the scorer run on every full run we have, on table 50ea0fb14a; the table is in `docs/validation/table-nov2.md` and between the markers in `README.md` and `docs/readme-final.md`. A new **Status** column says measured, simulated or not yet run (ab6e310). | Overnight item 2. | none | revert ab6e310 |
| 5 | **The agent row, on A's 500-task MBPP run:** the 30-task sample's 90% interval, $5.90 to $13.71 (point $7.15), holds the whole run's **$6.66** (the 30 sampled, $0.43, plus the other 470, $6.23, run 41383158427; all 470 ended done). Hit, 1 of 1. The point is 7% high. | Overnight item 3: written either way. | none | drop `agent_total_usd` from the manifest |
| 6 | **Knob response row from C's stranger test**, run in B's container on a local build of main cfcf83c (18 of 18 checks passed): p95 198 ms at 4x CPU slowdown, 9 of 9 under 400 ms. The manifest's `knob_ms` now takes the test's own file, as C asked. | The deployed site isn't reachable from here; main's own build is the same web code. | none | drop `knob_ms` from the manifest |
| 7 | **The GRPO A10 run counts as calibration for every row** in the validation manifest. | Its record's only probe is that 50-step run, so predicting it from itself tests nothing, and it calibrated the rollout band. | none | change its `calibrates` |
| 8 | **`engine.serving` stays off.** | No vLLM full run has finished (A's serving debug run of Nov 1 hit its 420 s limit at 24 of 1,024 prompts, with 251 KV preemptions), so the serving rows have nothing to score and the simulation is checked against no measured throughput. | none | the PM flips it once a vLLM full run scores within 15% |
| 9 | **`engine.divergence` stays off.** | The classifier misses held out: AUROC 0.66 (95% 0.56 to 0.76) against 0.85; lead 138 steps (3 to 269). The threshold rules (D6) give the warning either way. | none | none needed |
| 10 | **LC-PFN is not wired; no loss forecast ships.** Held out on the divergence sweep's 676 runs that didn't diverge (seen for 60 or 150 of 600 steps; LC-PFN on Modal, CPU, run 41397869309): the power law's 90% band covers 24% and 23%; LC-PFN's 93% (Wilson 90 to 95%) at 60 steps but 74% (70 to 77%) at 150; the ensemble 92% and 89.6% of the curves it answers, abstaining on 380 and 129 (40% and 72% over all 676). Report: `corpus/reports/loss_coverage_nov2.json`; the card is in `engine/lossforecast.py`. | Overnight item 6: wired only where its held-out coverage meets the 90% band. It does at 60 steps alone, on one model family, and falls as the probe sees more, so I didn't wire that one case. The forecast was never in `predict`, so nothing shown changes. | about $0.01 | none needed; to revisit, rerun `scripts/loss_coverage.py` on other job types |
| 11 | **Datasheet labelling write-up:** already on `ws-b` (213ba35, pushed Nov 1): model-written labels, 62% memory agreement, divergence 89% against a 94% always-"stable" baseline, 39 of 41 failure kinds, nothing trains on them. | Overnight item 4. | none | — |
| 12 | **RL entropy hook memory probe:** done before midnight under the PM's approval (A10, about $0.15, 80b8684): no OOM; 21.46 GB allocated and 22.91 GB reserved of 23.68 GB. | Overnight item 5. | $0.15 (Nov 2's standing yes) | — |
| 13 | **Option (b) for the fits floor and table 50ea0fb14a** were landed before midnight on Chiya's and the PM's calls (Nov 1, 9:47 PM and 11:58 PM); listed so the morning sees them in one place. | — | — | the old table is saved; revert the install commit |

## Plan items B owns (updated Sun Nov 1, 1 AM ET)

Every B item in the Contracts tab (Contract B, the gaps table, the adopted items) and the main plan (feature set, prediction engine, validation targets). "Branch" means on `ws-b` but not on `main`. Hours are my honest estimate of work left, review and fixes included. They don't include waiting on A's data or on a paid run.

**Hero lenses first**

| Item | Where | Hours left | Notes |
| --- | --- | --- | --- |
| Fine-tune lens: memory, P(OOM), step time, cost, money at risk | `main`; phase 2 table and fixes on branch | 1 | Today: step-time bands for vision, retrieval and pretrain, and P(OOM) shown "uncertain (5–95%)" in the middle |
| Agent lens: per-task bootstrap, context growth, cache prices, rate-limit wall clock, runaway rate | `main` | 0.5 | Tokens-only cost when tasks lack `cost_usd`, now that the price table is in |
| GRPO / RL: zero-variance fraction p^G + (1−p)^G from `pass_rate_per_prompt`, response-length bootstrap for rollout time | branch (`engine/rl.py`, behind `engine.rl`) | 1 | Calibrated Nov 1 on A's GRPO zoo runs 2 and 3 (L4) and A's 50-step run (A10, a-gt-grpo-50-b): length drift 0.21; the zero-spread band spans the Jeffreys and the measured rates (from 8 to 80 of the 50-step run's prompts it covered the rest, 7 of 7; the Jeffreys band alone missed 5 of 7, reading 0.13 to 0.18 low); abstains below 8 prompts. All three runs are calibration, so the RL rows score only the unseen TRL script |
| Campaign planner: grid pricing, Young/Daly intervals, Slurm wall-time segments, quota check, ASHA plan, Monte Carlo completion dates, team lanes | branch (`campaign/planner.py`; the view is behind C's `c.campaign_calendar`) | 1 | The card says its accuracy is the engine's, because its numbers come from the engine's intervals |

**The other lenses and engine features**

| Item | Where | Hours left | Notes |
| --- | --- | --- | --- |
| Serving: KV-cache formula, prefill and decode rooflines | branch (`engine/serving.py`, behind `engine.serving`) | 0 | |
| Serving: discrete-event scheduler simulation, calibrated on A's vLLM probes | branch | 1 | Calibration code is in; it needs A's vLLM probe records |
| Multi-node closed forms, labelled analytical-only | branch (`nodes`, `inter_node_gbps` knobs; two-level all-reduce) | 0 | Needs A to declare `nodes` |
| Loss forecast: power-law fit, plus LC-PFN posterior, abstaining on wide bands | branch: the power-law half (`engine/lossforecast.py`; band covers 89/100 on synthetic curves) | 3 for LC-PFN | LC-PFN needs a pretrained network: a cut candidate for Sunday |
| Data repetition and quality exposure (with A) | branch (`engine/repetition.py`) | 1 | Needs A's data profile per bin |
| Proxy campaigns | branch (`engine/proxy.py`: rank agreement and proxy settings) | 1 | Proxy runs themselves go through the campaign planner |
| Power analysis: seeds needed from seed-to-seed variance | branch (`engine/power.py`) | 0.5 | Needs A's multi-seed probe |
| GBDT residual on cross-GPU step time | branch: fitted and scored, **fails held out** (51% vs K1's 60% within 20%), so not used | 0 | |
| Divergence classifier: 800+ short runs, AUROC target 0.85 | branch: fitted on 720 runs (L4 and A10), **misses**: held-out AUROC 0.66 (0.56 to 0.76), lead 138 steps (3 to 269: the point estimate clears 100, the interval doesn't); not used, the warning comes from the D6 threshold rules | 0 | No more divergence sweeps (Chiya, Nov 1). Why it misses: 37 of 44 divergences are fp16 blow-ups, 15 in the first 30 steps |
| GNN cost model, used only if it beats the GBDT held out | branch: paired gain +4.8 points (−1.4 to +11.6), so the GBDT is used and the table says so | 0 | |
| Muse Spark Open batch-labels the corpus | not started | 2 | Needs a Baseten run through Actions: a paid ask |
| Conformal layer, abstention rules, float16 grids within the cell budget | `main`; real table on branch | 0 | |
| Pyodide parity test and import allowlist (gap 10) | `main` | 0 | |
| Sample records for the library (gap 4, with C) | `main` (`fill_predictions`) | 0 | C serves them |
| Run cost and its basis (gap 21) | `main` | 0 | |
| Corpus sweep: resumable, keyed, halts at budget (gap 13) | `main`; A100 board fix on branch | 0 | |
| Versioned model cards pinned in records (gap 15) | branch: `stale_models`, `current_models` | 0 | C shows it |
| Glimmer private-mode host proposal (with A, by Sun Nov 1) | branch: hosting now (`python/preflight/glimmer/`, `docs/decisions/glimmer-host.md`) | 0.5 | Chiya, Nov 1: Apache 2.0 from Meta, hold lifted. FP8 copy of Meta's bf16 checkpoint on an A100-40GB behind Modal proxy auth; Meta's Q4_K_M GGUF on an L4 for the own-GPU claim. Startup check, smokes ($2 cap) and the deploy in progress |
| Corpus published with its datasheet (gap 16) | branch (`corpus/DATASHEET.md`) | 0.5 | Publishing the rows themselves |

**Validation rows (README table)**

| Row | Where | Hours left | Notes |
| --- | --- | --- | --- |
| Interval coverage ≥ 88% | branch: memory 90.1% held out on the corpus | 2 | Still to do on full runs of unseen scripts (Nov 2) |
| Peak memory within 10% | partly measured: HF jobs and A's smoke record | 1 | Scored on A's full runs |
| OOM: no misses, ≤ 1 false alarm per 10 scripts, reliability diagram | branch: reliability bins | 1 | Scored on the bug zoo |
| Step time within 15%, same GPU | branch: the line puts 461 of 575 held-out queries within 15% | 1 | Full runs still to come |
| Step time within 20%, unmeasured GPU (4th type, incl. AMD) | corpus: K1 60% within 20% leave-one-GPU-out; A100 holdout same-GPU time 93% in band | 2 | AMD rows still to come; K1's band needs Chiya's call |
| GNN + GBDT beats GBDT alone | miss: the paired interval includes zero, so the GBDT is used | 0 | |
| Divergence warning: AUROC 0.85+, 100+ steps early | miss: AUROC 0.66 (0.56 to 0.76), lead 138 (3 to 269: the point estimate clears 100, the interval doesn't), 720 runs; the scorer reads `corpus/reports/divergence_nov1.json` | 0 | |
| RL rollout time within 20% | point 5.8% off between GRPO zoo runs 2 and 3; the zoo script is now calibration, so the row scores the unseen TRL script | 0.5 | Needs the Nov 2 unseen TRL run |
| Serving tokens/sec within 15% | not started | 1 | Needs A's vLLM full runs |
| Agent total cost inside the 90% interval | simulated on `main` (S6); no full run yet | 1 | |
| Abstention: every query outside the probed range is flagged | `main` (automated tests) | 0 | |
| Coverage self-test page (with C) | branch: corpus half (`calibration_summary`) | 0.5 | C renders it |
| Validation scorer (fills this table from full runs) | branch: `scripts/score_validation.py`, tested on Oct 31's records | 0 | Run on Nov 2's manifest; the PM adds the README markers in round four |

**Total: about 72 hours of B work left** (14.5 for the hero lenses, 46 for the rest of the engine, 12 for validation). Since the table was written, most engine items have landed on the branch, which leaves about 20 hours. Most of that waits on others: A's probes, the paid divergence and AMD runs, and the Nov 2 full runs. That is more than the time left before noon Sunday. I'll work through it in the order you gave, hero lenses first, so whatever you cut on Sunday comes off the end of the list.

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
  - **Flags:** see "What each `engine.*` flag gates" below.
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

## What each `engine.*` flag gates (Nov 1, for the README)

D found that most `engine.*` flags switched nothing. Checked in the code on ws-b:

| Flag | `flags.json` | Production | What it gates |
| --- | --- | --- | --- |
| `engine.memory` | off | on | **Every prediction**, not only memory. The runner fills the record's predictions (memory, time, cost, risk, the grids and the model cards) only with it on (`runner/pipeline.py::with_predictions`); off, the record keeps its measurements and has no predictions. |
| `engine.steptime_cross_gpu` | on | on (from `flags.json`) | Moving to a GPU with no probe: on, step time takes the held-out cross-GPU band (about ±45%) and shows it past ±50%, labelled rough; another vendor abstains with "no AMD measurements". Off, the usual abstain. |
| `engine.rl` | off | off | The RL lens's three extra readouts: zero-spread groups, rollout time and truncation. Memory, step time and cost on the RL lens don't depend on it. |
| `engine.serving` | off | off | The serving lens's readouts: throughput, time to first token (p95), time per token and KV capacity. |
| `engine.divergence` | off | off | The `risk.divergence` readout from the classifier. It stays off: the classifier missed its target ([report](../../python/preflight/corpus/reports/README.md)), and the threshold rules (D6) give the warning either way. |

**Wired:** until now the last four were constants in `engine/api.py` (`FEATURES`), so changing them in `flags.json` did nothing. `fill_predictions(record, flags=flag)` now reads them from the caller's flag reader for that fill (`FLAG_OF` maps each to its key; a test holds `flags.json` and `FEATURES` to the same values). The runner has to pass its reader: one line in A's `pipeline.py`, requested in `from-b.md`. Until A lands it, the `FEATURES` defaults decide, and they equal `flags.json` today, so nothing shown changes.

**Removed:** three B flags that nothing read. `engine.steptime_same_gpu` (same-GPU step time is always on, behind `engine.memory`), `corpus.sweep` (an offline tool; Chiya's yes per run is its gate) and `campaign.planner` (the planner runs only in C's sample script; the campaign view is behind C's `c.campaign_calendar`).

## GRPO peak check (Nov 1, A's 50-step run on an A10)

- The engine reads the run's peak as **measured 23.50 GB** (band 21.11 to 26.15), and the device's own most-used was 23.50 GB over 52 steps: it matches. `risk.oom` reads **tight, 34%**: the run fitted with 0.21 GB to spare on the A10's 23.71 GB (99.1% used), inside the ±10% held-out band. That's the honest reading of a run this close, not a bug.
- Fixed on the way: a peak after a GPU move read "measured" (it's a prediction now), and the per-term split scaled every term to the peak, so a 494M-parameter model's weights read 9.9 GB; weights and gradients now read 1.98 GB each, as the probe measured, and activations take the rest.
- C's 23.8 GB on the L4 fixture predates the measured context; the fixture reads 23.25 GB now.

## Monday's validation runs: expected cost (for Chiya's one approval, Nov 1)

Scored with `scripts/score_validation.py` (dry run on the zoo today: `tests/fixtures/validation/manifest_zoo_dryrun_nov1.json`; 2 runs scored, 8 "not measured yet", nothing crashes, predicted under production's flags). Each script needs a Preflight run (its record) and a full run (A's `ops_app.py::ground_truth`). The full runs are on an **L4** at $0.80/h: `ground_truth_on_l4` has the GPU fixed (`ops_app.py:325`). The L4 queued 19 to 59 minutes on Nov 1, which Modal doesn't bill but which costs time; an A10 option (no queue seen) would be A's change. **Blocker for Monday:** `ground_truth` accepts only `finetune_lora_hf`, `pretrain_nanogpt` and `grpo_trl` (`recorder/ground_truth.py` `ZOO_SCRIPTS`), so the other 15 full runs can't run through it until A extends it (asked in `from-b.md`). `--budget-min` is a soft target; the hard limit is each full run's 50-minute timeout (`GROUND_TRUTH_TIMEOUT_S`), and the Preflight runs have no fixed cap.

| Run | Full-run basis | Expected (L4) | At the run's budget | Hard ceiling (50 min) |
| --- | --- | --- | --- | --- |
| finetune_lora_hf | measured: 210 + 40 steps, 9.5 min of compute on an L4 (Oct 31) | about $0.15 | 35 min, $0.47 | $0.67 |
| grpo_trl | measured on an A10: 3 + 50 steps in 25 min; slower on an L4, and it ran out of memory there twice before A's fix | about $0.45 | 46 min, $0.61 | $0.67 |
| pretrain_nanogpt | no finished full run yet (Oct 28's hit its limit) | about $0.20 | 30 min, $0.40 | $0.67 |
| gnn_pyg, retrieval_st, vision_timm, eval_sweep_reasoning, serving_vllm | no full run yet; small models | about $0.15 each | 20 min each, $1.33 | $3.33 |
| torchtitan_debug | the debug model, in its own 15-minute container (`ops_app.py::torchtitan_check`) | about $0.10 | 15 min, $0.20 | $0.20 |
| agent_eval_spark | the 500-task benchmark, priced by the engine from the 30-task sample: **Spark API, not GPU** | $7.15 (90%: $5.90 to $13.71) | a dollar cap to set: the agent recorder's `max_usd_total`; $14 covers the band | $14 if set |
| 9 unseen scripts | unknown size: room left for 9 full runs | about $0.20 each | 30 min each, $3.60 | $6.00 |
| 19 Preflight runs (the records) | measured: $0.086 GPU and $0.018 Muse Spark for the LoRA run (Oct 28) | about $0.10 each, $1.98 | no fixed cap | |

**Total:** GPU and Spark for the records about $5.50 expected, $8.59 at every run's budget, about $13.50 if every full run hit its ceiling; plus the agent benchmark's Spark API, about $7.15, held to $14 if `max_usd_total` is set to it. **About $13 expected, about $27.50 at the ceilings.** The ceilings are estimates of the worst case, not hard bounds for the Preflight runs.

## Blocked
- A wiring the engine flags into the runner (one line, `from-b.md`).
- AMD: waits on Chiya's access approval.
- "Fits" on L4 and A10 fine-tunes: 19 held-out runs can't show P(OOM) below 2.5%, so nothing reads "fits"; a choice for Chiya (`from-b.md`).

## Next (Sun Nov 1, evening)
- Spark Open labelled the whole corpus: done, $8.79 of the $10 (`corpus/DATASHEET.md`).
- The top-up: done (about $16.54 of $25); the five-GPU table is in `docs/decisions/calibration-l4-phase1.md`, the rebuilt calibration table installed on ws-b.
- LC-PFN: ported and checked on the published checkpoints (quantiles to 1e-6); `lossforecast.ensemble` combines it with the power law. Not in `predict()` yet: it needs a runner hook (A) and a check on real probe losses.
- Serving calibration when A's capped vLLM record lands (the simulation now carries capped outputs past the cap).

## Earlier (Sun Nov 1)
- Divergence: done. The final fit on 720 runs misses (AUROC 0.66, lead 138 steps); not used. Both fit reports are committed under `corpus/reports/`. No more divergence spend.
- Agent runaways: a stalled run's tasks with no stop reason are possibly stuck; at most one per worker counts (`AgentSample.concurrency`, 0.6.1), and one when the worker count isn't recorded.
- Calibrate RL on A's 50-step GRPO run first (a hero lens), then retrieval, GNN and vision records as they land; serving once A's vLLM probes land.
- Validation: the engine freezes at the `validation-freeze` tag after round four; Monday's full runs are scored with `scripts/score_validation.py`, which prints the model ids it used. The Oct 31 LoRA run is used for calibration, not held out.
- AMD: no AMD measurements (the account is on a verification hold). If it lifts by Monday morning, the 300 paired configs (`data/amd_grouped.jsonl`) run first.
- Contract 0.6.0: filled (campaign calendar fields, `Estimate.rate` on every cost estimate).

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
| Oct 31 | `b-corpus-a100-3` (GitHub run 41197041814, cap $2.50 total): the other 93 of the list's 95 distinct configs (the 100 sampled repeat 5), 14 waves, all ok; 42 rows landed on A100-PCIe-80GB and 53 on A100-SXM-80GB, and the sweep kept going across both | A100-80GB x 1–2 | 12 min wall, 1,271 container-s | $0.97 ($1.46 of the $2.50 with the earlier runs) |
| Oct 31 | `b-corpus-export-3` (GitHub run 41197942422): every kernels row, A100 included, for the one holdout score | CPU | about 2 min | about $0.01 |
| Oct 31 | `b-corpus-export-2` and `b-corpus-export-hf-2` (GitHub runs 41191537650, 41191862052): every kernels and HF row back | CPU | about 7 min wall | about $0.02 |
| Oct 31 | `b-divergence-smoke` (GitHub run 41250984181, cap $0.25): 10 divergence configs on L4, **none ran**. The first wave (1 container x 2 configs, 540 s cut-off) gave nothing back in about 13 minutes, and the workflow's 15-minute limit killed the driver (exit 137). Either no L4 was free (queueing isn't billed) or the container hung past Modal's cut-off; Modal's page for app `ap-cqcYrRtr2P2TGmqjPSveJx` says which. Fixed in `d26ef36`: the driver now cancels a wave with nothing back after its cut-off plus 300 s | L4 x 1 (if it started) | 0 to 9 (container), 15 (wall) | $0 to $0.14 by the driver's worst case (at most $0.20 if the cut-off didn't hold); Modal's bill says |
| Oct 31 | `b-divergence-smoke-2` (GitHub run 41257375094, cap $0.25 total): 1 of 10 configs ran (20.9 s), because the first smoke run's unsettled worst case ($0.14) still counted against the cap | L4 x 1 | 1 (wall), 89 container-s | $0.02 by container time ($0.16 of the $0.25 with the first run) |
| Nov 1 | `b-divergence-sweep-1` (GitHub run 41257611597, first slice of the 800, cap $1.00 total): 24 configs ok, 16 to 25 s each (median about 22 s). Waves of 2 containers x 6. The third wave ran nothing: its containers waited for an L4 past the start-up allowance and found their end time gone (fixed locally: the end time now includes the driver's 300 s wait) | L4 x 2 | 21 (wall), 2,664 container-s | $0.65 by the driver's container time, which counts queue waits ($0.81 of the $1.00 with the smoke runs); Modal's bill is likely lower |
| Nov 1 | `b-divergence-sweep-2` (GitHub run 41275399391, cap $12.00 total): 304 configs ok (329 of 800 with the first slice), 16 to 29 s each (median 16.9 s), 16 a container on 4 L4s. Each batch spent 230 to 390 s in its container, but waves took 790 to 1,480 s: the containers waited for L4s. The last wave's 4 containers never got one and the driver cancelled them after 1,140 s with nothing back | L4 x 4 | 110 (wall), 26,093 container-s by the driver (in-container work about 5,300) | $6.38 by the driver's container time, which counts the queue ($7.19 of the $12.00 so far); Modal's bill is likely nearer $1.50 |
| Nov 1 | `b-divergence-export-1` (GitHub run 41299862454): the 329 divergence rows back, for the fit | CPU | about 2 | about $0.01 |
| Nov 1 | `b-divergence-sweep-3-l4` (GitHub run 41300143834, cap $9.59 for L4): 48 configs ok (377 on L4 in all); the second wave's 2 containers waited 2,640 s for L4s and ran nothing | L4 x 2-3 | 65 (wall), 9,241 container-s by the driver | $2.26 by the driver's measure ($9.45 on L4 so far) |
| Nov 1 | `b-divergence-sweep-3-a10` (GitHub run 41300145883, cap $2.41 for A10): 343 configs ok, a fresh draw (`--seed 1`), the board on every row; no queueing | A10 x 1-4 | about 70 (wall) | $2.10 by the driver's measure. Divergence total $11.55 of the $12.00 cap; no more divergence sweeps (Chiya, Nov 1) |
| Nov 1 | `b-glimmer-startup-check` (GitHub run 41327084275, pre-approved, CPU only, 70 min limit): failed while building images (Modal can't read the Python version in vLLM's registry image); nothing downloaded or ran | CPU | about 3 (wall), image builds only | about $0.00 |
| Nov 1 | `b-glimmer-startup-check-2` (GitHub run 41327585942, pre-approved, CPU only): both of Meta's downloads into the `preflight-glimmer-weights` volume (bf16 59.6 GB in 333 s, revision cec94fcf; GGUF Q4_K_M 16.8 GB in 124 s, revision b44f1072; both Apache 2.0, not gated); vLLM 0.28.0 and llama.cpp build 12030 (a91eab67d) images built; `vllm serve --help` failed on the CPU container, so the parser check read false there (the A100 smoke checks the parsers on a GPU) | CPU x 4 | about 10 (wall) | about $0.05 |
| Nov 1 | `b-glimmer-quantize` (GitHub run 41328761634): the FP8 copy of Meta's bf16 checkpoint (revision cec94fcf), 34.4 GB, made in 224 s | CPU x 8, 96 GiB | about 5 (wall) | about $0.10 |
| Nov 1 | `b-glimmer-smoke-l4` (GitHub run 41328764340): Meta's Q4_K_M GGUF on llama.cpp b12030; 5 triages, 43.7 s each (14.4 tokens/s), all valid JSON; the L4 took 19 min to schedule (not billed) | L4 x 1 | 248 container-s | about $0.06 |
| Nov 1 | `b-glimmer-smoke-a100` (GitHub run 41330014731): the FP8 copy on vLLM 0.28.0 with the muse_glimmer parsers; cold start 286 s; 5 triages alone at 12.8 s (38 tokens/s), 4 at once in 27.7 s; all valid JSON | A100-40GB x 1 | 374 container-s | about $0.23. **Glimmer so far: about $0.44 of the $2** (startup checks, quantisation and both smokes) |
| Nov 1 | `b-glimmer-deploy` (GitHub run 41331603537, pre-approved): `modal deploy` of `preflight-glimmer`; endpoint `https://shuureiyashi99--preflight-glimmer-glimmer-serve.us-east.modal.run`, proxy auth on, min 0 and max 1 containers, scaled to zero after 120 s idle; no container started by the deploy | none | 1 | $0.00. **Glimmer total about $0.44 of the $2** |
| Nov 1 | `b-spark-open-labels` (GitHub run 41333478249, pre-approved up to $1): Muse Spark Open labelled 64 of 73 failed corpus rows (80 calls; 16 failed: 8 at the 8,192-token output cap, $0.28 of the spend, 6 timeouts at 20 s, 2 re-asks cut off); agreement with B's rule labels: out-of-memory rows 11 of 17, 65% (Wilson 41% to 83%), the real test; diverged runs 41 of 41 (a non-finite step is in the evidence); 52 of 58 in all | Baseten API | about 9 (wall) | $0.54 of the $1 (37,471 tokens in, 115,463 out) |
| Nov 1 | `b-spark-open-relabel` (GitHub run 41340028097, under $0.50): the 9 unanswered rows with a 16,384-token cap and 90 s; 8 answered, 1 ran past the cap twice. Final agreement: out-of-memory rows 15 of 21, 71% (Wilson 50% to 86%); diverged runs 44 of 44; 59 of 65 in all, 91% (81% to 96%) | Baseten API | about 5 (wall) | $0.17 (10 calls; labels $0.70 in all) |
| Nov 1 | `b-lcpfn-check` (GitHub run 41365962299, CPU): the port (`corpus/lcpfn.py`) on both published checkpoints in torch 2.8.0, loaded with `weights_only=True`: the package's quantiles reproduced to 1e-6 on both; a falling loss seen for 200 of 2,000 steps forecast at 2.15 (1.92 to 2.39) against the true 2.09 | CPU x 2 | 8 s (function), about 3 (wall) | about $0.01 |
| Nov 1 | `b-spark-all-pilot` (GitHub run 41365960062, budget $0.30): 48 rows in 7 batches, 40 answered; one batch failed twice at the 12,000-token cap | Baseten API | 2 (wall) | $0.20 |
| Nov 1 | `b-spark-all-labels` (GitHub run 41366240357, budget $9.30): all 2,339 rows in 293 batches, 1,997 answered; agreement in `corpus/DATASHEET.md` | Baseten API | 14 (wall) | $8.59 recorded. **Whole-corpus labels $8.79 of the $10** |
| Nov 1 | `b-topup-dry` (GitHub run 41370815841): the sweep's dry run, each GPU's earlier spend read from its ledger | CPU | 3 (wall) | about $0.00 |
| Nov 1 | `b-topup-seed1` (GitHub run 41371132428, caps H100 $11, L40S $6.78, A10 $5.08, L4 $4.92, T4 $1.14, each its earlier spend plus the new cap): H100 293, L40S 293, A10 280, L4 89 (its sixth wave timed out after 690 s with no L4 free, $4.66 of $4.92 counted; the wave's 720 container-s, about $0.18, came after), T4 62 | H100, L40S, A10, L4, T4, at most 4 containers each (8 GPUs at once at the peak; PM, 8:12 PM: at most 3 from now on) | 100 (wall) | $12.82 with the L4's timed-out wave |
| Nov 1 | `b-topup-seed0` (GitHub run 41376298722, the same caps): H100 199 more (492 in all), L40S 99 more (392) | H100, L40S | 25 (wall) | $3.68 |
| Nov 1 | `b-topup-export` (GitHub run 41378563052): failed, the one-list read ran past 15 minutes | CPU | 16 (wall) | about $0.02 |
| Nov 1 | `b-topup-export-2` (GitHub run 41380335963): every GPU's rows, per GPU, gzipped | CPU | 6 (wall) | about $0.02. **Top-up total about $16.54 of the $25** |
| Nov 2 | `b-grpo-entropy-check` (GitHub run 41391382671, Chiya's yes, under $1): zoo/grpo_trl through ground_truth with the TRL 0.14 entropy hook, 2 bare and 12 recorder steps; no OOM, peak 21.46 GB allocated and 22.91 GB reserved of 23.68 GB; rl.entropy 12 values, 0.33 to 0.50 nats (rl_entropy_exact) | A10 x 1 | 8 (wall), about 450 GPU-s | about $0.15 |

**Measured:** phase 0 and phase 1's first run, plus A's sitting and both ground-truth attempts, cost **at most $1.55 combined**, from Modal's monthly total ($1.91 for October, all credits). Phase 1's resume adds $1.37 by the bound.

Total so far: ledger $1.21 plus about $0.03 today; billed estimate $4.80 to $6.70 until Modal's usage page gives the real figure. **The ledger under-counts container time.** It sums each config's own seconds, but containers are also billed for start-up, the per-batch volume reload, a volume commit after every config, and the waits between waves. That overhead was small in phase 0's single container and dominant in phase 1 (about 80 container-seconds per config against about 9 counted). So the $6 cap was enforced on the wrong measure. Phase 2 is on hold until the budget counts container wall time. Measured per config on the L4: grouped median 18.8 s, p90 25.7 s (n = 10); HF median 16.9 s, p90 25.2 s (n = 50). The HF jobs were far quicker than the 120 s guess, so phase 0 finished phase 1's HF set too. The ledger counts the configs' GPU-seconds; Modal's bill also counts container start-up between waves, so check it on Modal's usage page.

## Pushes to `ws-b`

Standing rule (Chiya, Nov 1, 11:13 AM ET): B pushes to `ws-b` without asking once gitleaks is clean (`git` over the pushed range and `dir`) and one read-only review has checked the change; each push is listed here. Anything for `main`, production flags or the contract is still asked first.

| When (ET) | Commits sent (oldest first) | Merges | Review | gitleaks |
| --- | --- | --- | --- | --- |
| Sun Nov 1, about 11:30 AM | 977d5ef contract 0.6.0 fields; 7c803b8 P(OOM) scored as shown, table 67ab8edae6; 1cfa689 divergence not used; 0143b6a small fixes; 6b388f0 engine flags wired, stalled-run cap, both fit reports; ab72de6 review fixes; and this status commit | none (f42a30a went out with the earlier push) | read-only reviews of 977d5ef to 1cfa689 (earlier), of 0143b6a and 6b388f0, and of ab72de6; 0143b6a also on Chiya's yes (11:04 AM) | clean, both scans |
| Sun Nov 1, about 11:45 AM | 285afc5 review minors (stalled-run reason wording, the scorer says the lead once, the API test checks the count); and this status commit | none | read-only review of 285afc5: nothing blocking | clean, both scans |
| Sun Nov 1, about 11:50 AM | 9789338 Glimmer Modal app (startup check, quantize, smokes, endpoint); and this status commit | none | read-only review of 9789338: nothing blocks the startup check; its deploy and smoke findings are fixed in the next commit, reviewed before any GPU run | clean, both scans |
| Sun Nov 1, about 12:10 PM | e2a7457 Glimmer review fixes (proxy auth, a vLLM image Modal can run); 47e61fe GRPO peak and RL calibration on the 50-step A10 run; and this docs commit (Glimmer decision, status, requests) | none | read-only review of e2a7457 and 47e61fe: nothing blocks the CPU startup check; its CUDA 13 point is fixed and reviewed before the A100 smoke | clean, both scans |
| Sun Nov 1, about 12:20 PM | e0775be Glimmer before the GPU runs (CUDA 13 base, CUDA versions in the startup check, 18-minute A100 cap); and this status commit | none | read-only review of e0775be: nothing blocking | clean, both scans |
| Sun Nov 1, about 12:35 PM | the CI format fix for tests/test_engine_api.py (CI red on d2dc950: `ruff format --check .`), the Glimmer revisions and spend, the reply to C; this commit | none | read-only review: nothing blocking, wording fixes applied; locally ruff format and check clean, 1,339 no-torch tests pass | clean, both scans |
| Sun Nov 1, about 12:30 PM | a10722a Glimmer measured (self-hosted price rows, deploy check, smoke numbers); d0ad37f its review fixes; fa7fbb7 the Spark Open corpus-labelling job and inputs; and this status commit | none | read-only reviews of a10722a, and of d0ad37f with fa7fbb7: nothing blocking | **not clean**: gitleaks flagged a corpus config key in fa7fbb7 (a false positive), and B's check piped its output through `tail`, so the push went ahead on a warning. CI went red: secrets (that finding) and python (the price version "2026-11-01b" isn't a date). B now gates every push on gitleaks' exit code |
| Sun Nov 1, about 12:50 PM | 5ec50e7 the CI fix (price version a date, the false positive allowlisted, row ids unique); and this status commit | none | read-only review of 5ec50e7: nothing blocking; the PM signs off the new `.gitleaksignore` entry | clean, both scans, by exit code |
| Sun Nov 1, about 2:10 PM | 5a1fd2c corpus export and datasheet; 5204042 the scorer's dry-run fixes; e319c8c Monday's cost; 1067d37 Spark Open's corpus labels; 74c745d and 9b4e0ae their review fixes; and this status commit | none | read-only reviews of 5a1fd2c, 5204042 and e319c8c (re-checked after 74c745d) and of 1067d37 (re-checked after its amend): nothing blocking | clean, both scans, by exit code |
| Sun Nov 1, about 3:20 PM | b223455 GRPO on A's real A10 record (the probe's parameter count; an unrecorded GPU named); e61de35 the Spark Open relabel; 1bf079d the LC-PFN fetch app; 4caef97 and the amended review-fix commit; and this status commit (round five's main was a fast-forward, no merge commit) | none | read-only reviews of b223455 and of e61de35 with 1bf079d, each re-checked after its fixes: nothing blocking | clean, both scans, by exit code |
| Sun Nov 1, about 3:45 PM | b49f095 the final Spark Open labels; a339b89 the LC-PFN fetch's return value; and this commit (the datasheet's cost rounding, this record) | none | read-only review of b49f095 and a339b89: nothing blocking | clean, both scans, by exit code |
| Sun Nov 1, about 4:30 PM | ff4d554 LoRA memory: queries sized by the training set's longest row when every row was scanned, a note on every readout; d7d7d04 the MI300X at $1.99/h; 15453a7 the AMD command repinned; effb801 the critical review fix (partial scans keep seq_len); and this status commit | none | read-only reviews of ff4d554 (blocked on partial scans, then re-checked after effb801) and of d7d7d04 with 15453a7: nothing blocking | clean, both scans, by exit code |
| Sun Nov 1, about 5:20 PM | 196b62f combined moves (local 2-D grids over knob pairs that clear an OOM no single move clears); d623e67 the corpus's Hugging Face link and the grid shape for C; d0b1949 their review fixes (a GPU swap doesn't stop the search, .clears ids); and this status commit | none | read-only review of 196b62f and d623e67, re-checked after d0b1949: nothing blocking | clean, both scans, by exit code |
| Sun Nov 1, about 6:40 PM | 79a9f8f fits/tight/fails reasons and serving past the cap (CI green on its own push) | none | read-only review of 79a9f8f: the fails prefix and a docstring, fixed before the push | clean, both scans, by exit code |
| Sun Nov 1, about 8:30 PM | a79d3c2 whole-corpus labelling job and inputs; ae0609a the top-up plan and `--exclude-keys`; 0d57c8a its review fixes; b5e510a the labeller's try/finally; 8eab96b the LC-PFN port, its Modal check and the loss ensemble; 225d542 the run length from the probes' dataset length; 6c520ce and 8440567 its review fixes; 0e594ec the review minors; f6388c1 requests and status; and this status commit | none | read-only reviews: a79d3c2 (three blocking points fixed, then re-checked; the re-check's hang fixed in b5e510a and checked again), ae0609a (nothing blocking), 8eab96b with b5e510a (nothing blocking), 225d542 (blocking, fixed in 6c520ce, re-checked clean) | clean, both scans, by exit code |
| Sun Nov 1, about 9:00 PM | 46f5522 and acb3138 the whole-corpus labels and their datasheet fixes (pushed about 8:45 PM, CI green); c8882d9 the fits floor (option b); f42cc91 serving reads A's probe cap; eb5e8d0 requests; 4929979 their review fixes (the pool only lowers P(OOM), the calibration does the same, serving's min cap); the equal-caps fix; and this status commit | none | read-only review of c8882d9 and f42cc91 (blocking on c8882d9: P(OOM) fell as memory rose), re-checked clean after 4929979; the equal-caps point from the re-check fixed | clean, both scans, by exit code |
| Sun Nov 1, about 11:10 PM | ef5c17e per-GPU export (pushed 9:55 PM); 9b7a897 FakeTensor cross-check, 12526ec its review fixes; 8c03ee8 RL signals, 3a4088d the entropy off-by-one fix, a5e6b1d its test, b6ee8be formatting (pushed 10:55 PM, CI green); fa831b7 the rebuilt calibration table and the five-GPU table; 77992a7 requests; 2a1404a review fixes; and this status commit | none | read-only reviews: ef5c17e (clean); 9b7a897 (blocking: the check's own 638x and two definitional gaps), re-checked clean after 12526ec; 8c03ee8 (blocking off-by-one), re-checked after 3a4088d with its test; fa831b7 (clean: reproduces byte for byte, holdout untouched; doc fixes in 2a1404a) | clean, both scans, by exit code |

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
