# Requests from B (prediction engine and corpus)

Only B edits this file. Each line is something this agent needs from another (or a reply to one). The receiving agent answers in its own file with a REPLY line naming the request; the asker moves the line to Done.

`- TODO(contract) [B -> X] what is needed, and why. (date)`

## Open

- TODO(contract) [B -> Chiya] Add flags `engine.memory`, `engine.steptime_same_gpu`, `engine.steptime_cross_gpu`, `corpus.sweep`, all off until their validation rows pass. (Oct 27)
- TODO(contract) [B -> D] GPU prices live in `engine/gpus.py` (`price()`, `prices()`, `on_demand_reference()`), each with provider, source and as-of date; D3 reads and extends that table rather than keeping a second one. (Oct 27)
- REPLY [B -> A] Grid.scale: confirmed; B writes grids with `encoding.choose_scale`. (Oct 27)
- REPLY [B -> A] p50_s and three memory probes: adopted. The same-GPU model converts `by_phase` fractions to seconds of `p50_s`; calibration fits the three probes by least squares (S1, 93.6% within 10% near the cap on your plan). (Oct 27)
- REPLY [B -> A] `recorder/gpus.py`: `engine.gpus.peak_flops(gpu_name, precision, memory_bytes=None)` is a drop-in with the same arguments and None semantics; 47 GPUs (40 NVIDIA, 7 AMD), plus ROCm and MIG names. (Oct 27)
- REPLY [B -> A] Golden fixtures: B sends the `make_golden.py` call once `engine/api.py` lands (Wed Oct 28). (Oct 27)
- TODO(contract) [B -> A] The golden pretrain record's `env.gpu.name` is torch's raw "A100-SXM4-40GB", not "A100-SXM-40GB". B resolves both, so it only matters for display and for grouping records by GPU; fix it whenever the goldens are next regenerated. (Oct 27)
- REPLY [B -> Chiya] Agent sample size: with the runaway bound in B's code, 30 tasks covers 90.3% (interval 1.16x the total) and 50 tasks 89.5% (0.96x). Both clear 88%, so the choice is interval width against sampling cost (S6 in docs/decisions/engine-settings.md). (Oct 27)
- TODO(contract) [B -> A] `recorder/kernels.py` `_MATMUL` leaves out in-place and other matmul ops (`aten::addmm_`, `aten::baddbmm_`, `aten::addbmm`, `aten::addmv`, `aten::mv`, `aten::dot`, `aten::_int_mm`), so those rows carry no FLOPs and B has to scale them by class. B classes them all as gemm now; counting their FLOPs lets the roofline do the rest. (Oct 27)
- TODO(contract) [B -> A, schema] Pin whether `AgentTask.output_tokens` includes `reasoning_tokens` (OpenAI and Anthropic count reasoning inside output). B assumes it does and adds input plus output for `tokens.total`; a line in the field description would settle it. (Oct 27)

## Done

- DONE [B -> C] (C, Oct 27: a tokens dial replaces speed on agent runs, `risk.runaway` labelled, and `cost.total`'s reason shows on the cost dial's hover) More targets, for agent and API runs: `tokens.total` (unit `tokens`) and `risk.runaway` (unit `fraction`: the share of the run's tasks that run away, over 8x the median). `cost.total` there carries the runaway margin in its upper end, and `reason` says so in a sentence C can show. (Oct 27)
- DONE [B -> C] (C, Oct 27: `web/src/workspace/record/terms.ts`) Targets B emits: `memory.peak` and `memory.<term>` (weights, master_weights, gradients, optimizer, frozen_models, comm_buffers, activations, logits, kv_cache, generation_engine, context, eval_spike), `time.step`, `time.total`, `cost.total`, `risk.oom` (point = P(OOM), unit `fraction`). Show fits below 0.01, fails at 0.80 or more, tight between (docs/decisions/engine-settings.md S2b); `engine.memory.FITS_BELOW` and `FAILS_AT` hold the cut-offs. (Oct 27)
- DONE [B -> A, CI] (A added it in 67641f7) The torch job installs `transformers` without `accelerate`; current transformers' Trainer requires `accelerate>=1.1`, so `test_hf_trainer_probe_measures_padding` and `test_trainer_logs_become_rl_signals_and_logging_is_forced_every_step` fail outside your env. Add `accelerate` to that `uv pip install` line. (Oct 27)
- DONE [B -> Chiya] (Chiya approved Oct 27; B adds the keys in ws-b) `engine.memory`, `engine.steptime_same_gpu`, `engine.steptime_cross_gpu`, `corpus.sweep`, all off until their validation rows pass. (Oct 27)
- DONE [B -> A] `Probe.extras` carries `params_total`, `params_trainable`, `tokens_per_step`, `samples_per_step`, `micro_steps_per_step` (and `host_rss_peak_bytes`) under exactly those keys (A, ws-a: `probes/local.py` EXTRAS_KEYS). (Oct 27)
- DONE [B -> A] `gpu_type` knob: categorical, values = `Constraints.gpu_types` or `engine.gpus.MODAL_GPU`, current = the requested or chosen probe GPU; `skeleton_record(..., gpu=, constraints=)` (A, ws-a: `harness/skeleton.py`). Not on agent jobs. (Oct 27)
- DONE [B -> A] `env.gpu.name` via `engine.gpus.resolve(name, total_memory, multi_processor_count)`, then the ROCm `gcnArchName`; an unknown device keeps its raw name and sets `env["gpu_unresolved"]` (A, ws-a: `recorder/gpuinfo.py`). MFU uses `engine.gpus.peak_flops` the same way. (Oct 27)
- DONE [B -> A, schema] `Probe.kernels: KernelTable` (rows of `KernelStat`: name, `kernel_class`, calls, device_s, flops, bytes, dtype, per step; then steps, backend, allow_tf32, classified_by), filled from the profiled current-settings probe with `with_flops=True, record_shapes=True` and your `classify_kernel`. The class field is `kernel_class`, not `class`: records are dumped without aliases, so a `class` alias would not survive. Chiya approved A adding it before the freeze; golden finetune and pretrain p1 carry one. (Oct 27)
- DONE [B -> A, schema] `ModelInfo` gains `gated_mlp`, `tie_embeddings`, `norm` ("rmsnorm" | "layernorm"), `attn_impl`, `head_dim`, plus `n_experts`, `experts_per_token`, `max_positions` and `shape_source` ("code", "hf_config", "modules"). Read from the running model, HF config first (`recorder/arch.py`); `probes.untrusted.with_shape()` folds it into the script map's model. The golden fine-tune carries them. (Oct 27)
- DONE [B -> A] Knobs `lora_alpha`, `lora_dropout`, `attn` (attn_implementation), `quantization` (your `QUANT_BYTES` keys) and `context_limit` are declared from the code (A, ws-a: `harness/static.py`). (Oct 27)
