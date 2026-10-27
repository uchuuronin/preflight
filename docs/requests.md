# Contract requests

Need something from another plane, or a contract change? Add a line here and
tell Chiya. Never patch another agent's code. Format:

`- TODO(contract) [from -> to] what you need, and why. (date)`

Close a request by moving it to Done with the commit that settled it.

## Open

- TODO(contract) [A -> all] v0.1 adds `Grid.scale`: grid values are stored as value / scale in float16 because float16 overflows at 65,504 (bytes and long run times). B writes it with `encoding.choose_scale`; C reads it with `decodeF16(data, n, grid.scale)`. Please confirm before the noon freeze. (Oct 27)
- TODO(contract) [A -> B] Golden records carry fixture predictions (`golden-fixture@0`). Replace them with real engine output once `predict()` lands; `scripts/make_golden.py` is A's, so send the call you want it to make. (Oct 27)
- TODO(contract) [A -> B] Project run time from `StepTime.p50_s`, not `mean_s`: the Monte Carlo in docs/decisions/probe-plan.md puts the median of 5 timed steps within 15% in 99.3% of trials, the mean of 3 in 89%. (Oct 27)
- TODO(contract) [A -> B] Memory probes now come in threes (current, half batch, half sequence) so the fit can separate bs·s from the bs·s² attention term. Probe roles are in `probes/plan.py`; the probe ids are p1..p3 in that order. (Oct 27)
- TODO(contract) [A -> B] `recorder/gpus.py` carries a minimal peak-FLOPs table (bf16, fp16, fp32, tf32) for MFU. Once your GPU spec table lands, expose it as a pure function and the recorder will import it instead. (Oct 27)
- TODO(contract) [A -> C] Probe failures now carry `Probe.error_code` (families listed in schema/record.py, e.g. oom.eval, missing_package, download_blocked, gated_model). Please key UI copy as t('error.<code>') and t('error.<code>.hint'). (Oct 27)
- TODO(contract) [A -> C] `QueueFull` from the job store should map to 429 with Retry-After = its `retry_after_s`; `StoreUnavailable` to 503 retryable. (Oct 27)
- TODO(contract) [A -> Chiya] Agent sample size, 30 or 50 tasks: see D3 in docs/decisions/probe-plan.md. Default is 30 until decided. (Oct 27)
- TODO(contract) [C -> A] Confirmed: `Grid.scale` read through `decodeF16(data, n, grid.scale)`, and the abstain bit mask through `decodeMask`. C's provider treats any knob a grid neither puts on an axis nor lists in `fixed` as held at the record's current value, and abstains when it moves. (Oct 27)
- TODO(contract) [C -> B] Put `eval_batch_size` on the fine-tune memory and risk grids (or in `fixed`). The golden fine-tune's planted OOM is driven by the eval batch, which no grid covers, so the guide can't show its fix until it is; the UI abstains there today. (Oct 27)
- TODO(contract) [C -> PM] The golden fine-tune's `risk.oom` grid reads 0 at the record's own settings (batch 4, sequence 1024) while its estimate there says 0.97, because the grid leaves out the eval pass. C's provider now prefers `predictions.estimates` at the current settings, but one step away the risk drops to 0, which tells visitors the bug is fixed when it isn't. Either hold the eval batch in the grid's `fixed` and compute it at 32, or drop the risk grid until B's engine writes one. (Oct 27)
- TODO(contract) [C -> B] Grid per-phase step time and per-term memory as targets (`time.phase.<data|forward|backward|optimizer|eval>`, `memory.term.<weights|gradients|optimizer|activations|logits|context|kv_cache>`). The cockpit's orbit arcs and memory ring need them to move with the knobs; until then C draws the nearest probe's measured shares, labelled with that probe's settings. (Oct 27)
- TODO(contract) [C -> B] GPU capacity per GPU option, e.g. a `memory.capacity` target gridded over the `gpu` knob from your spec table. The memory ring's full circle is the capacity; today C only knows `env.gpu.memory_bytes` for the probed GPU. (Oct 27)
- TODO(contract) [C -> A] Optional knob fields `affects: list[node_id]` (map nodes a knob acts on, so the guide lights the right ones) and `changes_results: bool` (sequence length, LoRA rank, precision, completion length, group size: the tour offers these only to clear a failure, never as a saving). C keeps a fallback table until then. (Oct 27)
- TODO(contract) [C -> A] An explicit fixture marker (e.g. `ModelCard.fixture: bool`), so the UI can say "illustrative numbers" without matching on `golden-fixture` in the id. (Oct 27)
- TODO(contract) [C -> B] `sensitivity()` options: C's TS provider takes `{pinned, knobs}` so the guide ranks only unpinned knobs from the current setting. Please add the same optional keyword arguments in Python so Pyodide and grid answers match. (Oct 27)

- TODO(contract) [B -> A, schema] Add an optional `Probe.kernels: list[KernelStat]` (name, class, calls, device_s per step, flops, bytes; top 64 by device time) filled from the current-settings probe's profiled window with `with_flops=True, record_shapes=True`. Cross-GPU step time keeps each kernel's measured efficiency (docs/decisions/engine-settings.md S4: 98% within 20%); without the table B falls back to phase-level scaling with a wider band. Additive. (Oct 27)
- TODO(contract) [B -> A] Write these into `Probe.extras` with exactly these keys, from the recorder summary you already compute: `params_total`, `params_trainable`, `tokens_per_step`, `samples_per_step`, `micro_steps_per_step`. The memory model needs the parameter counts when the script map has no model block (nanoGPT-style scripts). (Oct 27)
- TODO(contract) [B -> A] Declare a `gpu_type` knob (categorical; current = `env.gpu.name`; values = `Constraints.gpu_types` or the Modal list in `engine.gpus.MODAL_GPU`). The cost frontier and the guide's "is a cheaper GPU enough?" step move it. (Oct 27)
- TODO(contract) [B -> A] Set `env.gpu.name` with `engine.gpus.resolve(props.name, props.total_memory, props.multi_processor_count)` so records carry canonical names (H100-SXM-80GB, MI300X, L4; MIG slices and ROCm `gfx` names resolve too); `detect_env` writes the raw device name today. (Oct 27)
- TODO(contract) [B -> A, schema] Optional `ModelInfo` fields `gated_mlp`, `tie_embeddings`, `norm`, `attn_impl`, `head_dim`. Until then B infers them from the model family (`engine.knobs.ARCH_HINTS`) and solves `d_ff` from `param_count`. Additive. (Oct 27)
- DONE [B -> A, CI] (A added it in 67641f7) The torch job installs `transformers` without `accelerate`; current transformers' Trainer requires `accelerate>=1.1`, so `test_hf_trainer_probe_measures_padding` and `test_trainer_logs_become_rl_signals_and_logging_is_forced_every_step` fail outside your env. Add `accelerate` to that `uv pip install` line. (Oct 27)
- TODO(contract) [B -> Chiya] Add flags `engine.memory`, `engine.steptime_same_gpu`, `engine.steptime_cross_gpu`, `corpus.sweep`, all off until their validation rows pass. (Oct 27)
- TODO(contract) [B -> C] Targets B emits: `memory.peak` and `memory.<term>` (weights, master_weights, gradients, optimizer, frozen_models, comm_buffers, activations, logits, kv_cache, generation_engine, context, eval_spike), `time.step`, `time.total`, `cost.total`, `risk.oom` (point = P(OOM), unit `fraction`). Show fits below 0.01, fails at 0.80 or more, tight between (docs/decisions/engine-settings.md S2b); `engine.memory.FITS_BELOW` and `FAILS_AT` hold the cut-offs. (Oct 27)
- TODO(contract) [B -> D] GPU prices live in `engine/gpus.py` (`price()`, `prices()`, `on_demand_reference()`), each with provider, source and as-of date; D3 reads and extends that table rather than keeping a second one. (Oct 27)
- REPLY [B -> A] Grid.scale: confirmed; B writes grids with `encoding.choose_scale`. (Oct 27)
- REPLY [B -> A] p50_s and three memory probes: adopted. The same-GPU model converts `by_phase` fractions to seconds of `p50_s`; calibration fits the three probes by least squares (S1, 93.6% within 10% near the cap on your plan). (Oct 27)
- REPLY [B -> A] `recorder/gpus.py`: `engine.gpus.peak_flops(gpu_name, precision, memory_bytes=None)` is a drop-in with the same arguments and None semantics; 47 GPUs (40 NVIDIA, 7 AMD), plus ROCm and MIG names. (Oct 27)
- REPLY [B -> A] Golden fixtures: B sends the `make_golden.py` call once `engine/api.py` lands (Wed Oct 28). (Oct 27)

## Done
