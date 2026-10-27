# Status · B (prediction engine and corpus)

Updated Tue Oct 27, afternoon ET, on `ws-b` rebased onto A's `ws-a` (b59334b).

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
- Tests live in root `tests/`; 701 pass across the repo (A's included), ruff clean with the repo config.

## Blocked
- Modal access, for the NVIDIA corpus slice.

## Next (Wed Oct 28)
- `engine/api.py`: `predict`, `build_grids`, `sensitivity` against the v0.1 schema, then the `make_golden.py` call for A.
- Conformal layer (CQR, log-ratio scores) replacing the simulated band constants.
- `engine/api.py` wires `agent_totals` into `predict` for agent and API lenses, with D's API price table as `price`.

## Contract requests
In `docs/requests.md`. Open from B to A: one golden-record GPU name, FLOP counts for in-place matmuls, and the reasoning-token convention; four flags to Chiya, target ids and cut-offs to C (now with `tokens.total` and `risk.runaway`), the price table to D. Every A -> B request is closed except the golden predictions, which wait for `engine/api.py`. A reply to Chiya on the agent sample size: 30 and 50 tasks both clear 88% (S6).
