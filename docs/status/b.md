# Status · B (prediction engine and corpus)

Updated Tue Oct 27, early morning ET, on `ws-b` rebased onto A's `ws-a`.

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
- Tests live in root `tests/`; 513 pass across the repo (A's included), ruff clean with the repo config.

## Blocked
- Modal access, for the NVIDIA corpus slice.
- Cross-GPU K1 needs a per-kernel table on probes; until A's recorder writes one, cross-GPU uses the phase-level fallback with a wider band.

## Next (Wed Oct 28)
- `engine/api.py`: `predict`, `build_grids`, `sensitivity` against the v0.1 schema, then the `make_golden.py` call for A.
- Conformal layer (CQR, log-ratio scores) replacing the simulated band constants.
- Agent cost model (bootstrap with the rule-of-three tail from A's D3).

## Contract requests
In `docs/requests.md`, B -> A: kernel table on probes, parameter counts in `Probe.extras`, a `gpu_type` knob, canonical `env.gpu.name`, optional `ModelInfo` shape fields. (`accelerate` in the CI torch job is done.) B -> Chiya: four flags. B -> C: target ids and the fits/tight/fails cut-offs. B -> D: the price table. Plus replies to A's five requests.
