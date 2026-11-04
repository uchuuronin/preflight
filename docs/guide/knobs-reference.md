# Knobs and inputs

Every setting a visitor can change in Preflight, what each one moves, and what the engine
doesn't model yet. Written from the code as of Nov 3 (ws-b bc5f9483); every number carries a
source tag: measured, derived, cited, simulated or guess.

## How to read this page

A **knob** is one setting of your job that Preflight lets you move in the cockpit: batch size,
sequence length, precision, the GPU, and so on. Preflight reads the knobs from your code when you
check in, runs a few short probes on a real GPU at one or more settings, and fits a model of how
the job behaves. When you move a knob, the cockpit asks that model what changes. Some knobs carry
a **Code** badge: changing them edits your script, Muse Code writes the change, and a short verify
run checks it.

The cockpit shows four **readouts**. **Memory** is the peak GPU memory one card needs (the map's
core, "Peak memory"). **Time** is the time to finish the whole run ("Time to finish"), built from
the time per step times the number of steps. **Cost** is that time priced at the GPU's hourly rate,
times the GPUs it holds ("Run cost"); for agent and API runs it is tokens priced at the API's
rates. **Risk** is the chance the run breaks ("Chance it breaks"): for GPU jobs, the chance it runs
out of memory; for agent runs, the chance of a runaway task. Each readout is a point with an
**interval**: a range the true value should fall inside 9 times in 10 (a 90% interval
[cited: python/preflight/schema/prediction.py, `level` 0.9]). Each also carries a status word
(Measured, Predicted, Extrapolated, Not measured here) that says how far it sits from what the
probes actually ran. Read the status before the number. Every technical term on this page has a one-line
explanation under [Words used here](#words-used-here) at the end.

## What's on in production

Some engine models sit behind flags. Production uses `flags.json` overridden by
`flags.production.json`. As of Nov 3:

| Engine flag | In production | What it adds |
|---|---|---|
| `engine.memory` | on | The runner fills predictions at all (`runner/pipeline.py`) |
| `engine.steptime_cross_gpu` | on | Time on a GPU the probes didn't run on, with the held-out cross-GPU band |
| `engine.rl` | on | RL readouts: groups with no reward spread, rollout time, truncation |
| `engine.carry_free_knobs` | on | Learning-rate-like knobs keep a Measured or Predicted status when moved |
| `engine.serving` | off | Serving throughput, time to first token, and how many sequences fit in memory (the serving simulator) |
| `engine.divergence` | off | The divergence (training blows up) risk readout |
| `engine.context_same_stack` | off | Keeping the probes' measured CUDA context on a GPU move within one vendor |
| `engine.memory_reserved` | off | A reserved-pool margin on memory |
| `engine.eval_passes` | off | Time for an eval run by command (lm_eval), from its scoring passes |
| `probes.lr_range` (not an engine flag) | off | The learning-rate range test |

## Rules that apply to every knob

These hold across the page, so the entries below don't repeat them.

- **Out-of-memory risk follows memory.** Any knob that moves memory moves the chance of running
  out of memory too. When an entry says risk is "not modelled", it means training-stability risk
  (loss blowing up), whose readout is off in production (`engine.divergence`).
- **Off the probed values.** If you set a knob to a value no probe ran, memory and time turn from
  Predicted to Extrapolated, with the reason "is off the probed values; the model carries it
  there". This doesn't apply to batch size, sequence length and the other size knobs, run
  length, the GPU and the GPU count, the save and eval intervals, gradient accumulation (for
  time), the fan-outs, or the free knobs below [derived: engine/api.py `_off_probe`].
- **A knob the engine doesn't model for a readout.** Moved off the probed values, it makes
  memory abstain ("Not measured here", "changes memory in a way the engine doesn't model yet")
  and holds time at the probe's value with a band 1.5 times wider, labelled Extrapolated, with
  the reason "{knob} isn't modelled for time yet; held at its probed effect" [guess: engine/api.py
  `UNMODELLED_WIDEN` 1.5]. If that wider band would pass plus or minus 50%, time abstains
  instead ("Not measured here"). Below, "time holds" means this. The step-time model reads only
  the work per step (batch, sequence length, image size, fan-outs, accumulation, and for RL the
  prompt and completion lengths), precision (and `tf32` inside a precision change),
  checkpointing, the GPU, the number of GPUs, nodes and the bandwidth between them, and the
  script's own distributed strategy; every other knob's time effect holds.
- **Free knobs carry.** With `engine.carry_free_knobs` on (production), moving `lr`,
  `lr_schedule`, `warmup`, `warmup_steps`, `weight_decay`, `grad_clip`, `lora_alpha`, `clip_low`,
  `clip_high` or `seed` leaves memory and time, and their status, as they were [derived: engine/api.py
  `FREE_KNOBS`]. `kl_beta`, `lora_dropout`, `temperature` and `loss_type` are kept out on
  purpose, so moving them marks memory and time Extrapolated even though the numbers don't change.
- **Work units.** Memory and step time are fitted against one size axis per job type: batch x
  sequence length for language models, batch x image size squared for vision, batch x the
  product of the neighbour fan-outs for GNNs, times gradient accumulation for step time
  [derived: engine/knobs.py `work_units`]. Knobs on that axis move the numbers directly. Past
  twice the largest probe on that axis, memory and time abstain [cited: engine/memory.py
  `EXTRAPOLATE_FACTOR` 2.0, A's probe-plan D2].

## Knobs

Each entry: **Label** (what the site shows) · `id`. Then what it is, when to change it, what moves,
and what isn't modelled. "No label" means `web/src/i18n/en.json` has no `knob.<id>` key, so the
site shows the id.

### Shape of the work

**Batch size** · `batch_size`
- What: how many examples go through the model in one step.
- When: raise it to use the GPU better; lower it if memory is tight.
- Moves: memory up (activations and logits grow with it). Time per step up. Steps down
  (steps = examples x epochs / (batch x accumulation x GPUs)). Total time and cost go whichever
  way the product goes; the model computes it, it doesn't assume it falls.
- Not modelled: training-stability risk.

**Eval batch** · `eval_batch_size`
- What: how many examples go through at once during evaluation.
- When: lower it when the peak comes during eval (the map shows an "Eval spike").
- Moves: memory, when the eval pass sets the peak: up with the eval batch.
- Not modelled: time and cost. Each eval pass is counted at the probes' eval batch, so moving it
  changes neither the number nor its band.

**Encode batch** · `encode_batch_size`
- What: how many passages a retrieval job encodes at once.
- When: same as batch size, for the corpus-encoding phase.
- Moves: memory and time per step up with it (it is the batch on the work axis when the script has no batch-size knob); cost follows time.
- Not modelled: nothing beyond the general rules.

**Sequence length** · `seq_len`
- What: the longest an example may be, in tokens; longer rows are cut there.
- When: lower it to save memory; raise it if your data is being cut.
- Moves: memory up while your rows are longer than the limit. When the recorder scanned every
  training row, memory is sized at the shorter of the limit and your longest row, so raising it
  past your longest row changes nothing. Step time follows the tokens a step really holds
  (padded to the longest row in each batch), so it too only rises while rows reach the limit.
- Not modelled: above the length the dataset was scanned at, rows past the scan aren't
  measured; time is held at the cut and labelled Extrapolated.

**Image size** · `image_size`
- What: the side length images are resized to.
- When: lower it to save memory and time; raise it for accuracy.
- Moves: memory and time per step up, with batch x size squared on the work axis.
- Not modelled: nothing beyond the general rules.

**Gradient accumulation** · `grad_accum`
- What: adding up gradients over several small batches before one update, so it acts like a
  bigger batch (effective batch = batch x accumulation x GPUs).
- When: to keep the effective batch while lowering batch size for memory.
- Moves: on its own, memory up, by as much as one copy of the trainable gradients (above 1, the summed gradients stay in memory through
  later forward passes). Time per update up; updates per epoch down by the same factor. Cost
  follows time. The memory saving comes from lowering batch size alongside it.
- Not modelled: training-stability risk.

**Training steps** · `max_steps`
- What: how many updates the run makes; when set, it wins over epochs.
- When: to plan a shorter or longer run.
- Moves: time and cost, in proportion.
- Not modelled: nothing; it doesn't change one step's memory or time.

**Epochs** · `epochs`
- What: how many passes over the training data.
- When: to plan run length when the script counts in epochs.
- Moves: time and cost, in proportion (through the step count).
- Not modelled: quality.

**Save every** · `checkpoint_every`
- What: how many steps between checkpoint saves.
- When: save less often to save time; more often to lose less work if the job is interrupted.
- Moves: time down as you raise it (each save adds the probes' measured save time). If no probe
  measured a save, the reason says "checkpoint time not measured, so not included". Risk: with a
  preemption rule set (see Other inputs), raising it raises the work lost per interruption.
- Not modelled: nothing beyond that.

**Eval interval** · `eval_every`
- What: how many steps between eval passes.
- When: evaluate less often to finish sooner.
- Moves: time down as you raise it (each pass adds the probes' measured eval-pass time).
- Not modelled: memory: the eval peak comes from the eval batch, not from how often it runs.

**Log interval** · `log_every`
- What: how many steps between log lines.
- When: rarely; it matters only if logging is slow.
- Moves: time, in principle.
- Not modelled: time holds. Logging cost is folded into the measured step time.

**Warm-up steps** · `warmup_steps`
- What: steps at the start where the learning rate ramps up from low.
- When: to calm an unstable start.
- Moves: risk and quality only.
- Not modelled: both. It feeds the divergence classifier, which is off in production. A free knob.

### Data loading

**Data workers** · `num_workers`
- What: background processes that load and prepare batches.
- When: raise it if data loading shows up as a big share of each step.
- Moves: time, in principle.
- Not modelled: time holds (see the rules above).

No label · `prefetch_factor`
- What: how many batches each worker prepares ahead.
- When: with `num_workers`, when loading stalls steps.
- Moves: time, in principle.
- Not modelled: time holds, as for data workers.

**Packing** · `packing`
- What: joining short examples into full-length rows so no tokens are padding.
- When: your examples are much shorter than the sequence length.
- Moves: time and cost, in principle down (more real tokens per step).
- Not modelled: both. Time holds at the probe's value with a wider band. If packing differs from
  what the probes ran and the dataset size isn't known, time abstains ("packing changes the row
  count"). The probes' measured padding fraction bounds the possible gain.

No label · `group_by_length`
- What: batching examples of similar length together, so less padding.
- When: example lengths vary a lot.
- Moves: time and cost, in principle down. The memory peak doesn't change: one batch of the
  longest examples still happens.
- Not modelled: time and cost; held with a wider band.

No label · `pin_memory`
- What: page-locked host memory for faster copies to the GPU.
- When: rarely; usually already on.
- Moves: time, in principle.
- Not modelled: time holds; copy speed is folded into the measured data phase.

### Model shape

**Layers** · `n_layer`
- What: how many transformer blocks the model has.
- When: designing a model from scratch (pretraining).
- Moves: memory up (weights, gradients, optimizer state and activations per layer).
- Not modelled: time and cost: the step-time model works on the work axis, not the layer count,
  so time holds. Quality.

**Model width** · `d_model`
- What: the size of the model's hidden vectors.
- When: same as layers.
- Moves: memory up.
- Not modelled: time and cost (as for layers), quality.

**Attention heads** · `n_head`
- What: how many attention heads each layer splits into.
- When: same as layers.
- Moves: memory, a little: only the attention scores and their saved statistics depend on it.
- Not modelled: time (as for layers), quality.

**Hidden width** · `hidden_channels`
- What: the width of each GNN layer.
- When: sizing a graph neural network.
- Moves: memory, time and cost, in principle.
- Not modelled: memory, time and cost: Preflight doesn't read this knob yet; moved off the
  probes' value, the numbers hold there, labelled Extrapolated. Quality.

**GNN layers** · `gnn_layers`
- What: how many message-passing layers the GNN has.
- When: sizing a graph neural network.
- Moves: memory, time and cost, in principle.
- Not modelled: same as hidden width.

**Neighbours at hop 1 to 4** · `fanout.hop1` ... `fanout.hop4`
- What: how many neighbours a GNN samples per node at each hop.
- When: to trade accuracy for memory and speed on large graphs.
- Moves: memory, time and cost up: the product of the fan-outs is on the work axis.
- Not modelled: quality. A fifth hop has no label.

**LoRA rank** · `lora_r`
- What: the size of the small trainable matrices LoRA adds (LoRA: training a few small added
  matrices instead of the whole model).
- When: raise it for more capacity, lower it for less memory.
- Moves: memory up, a little (adapter weights, their optimizer state and one small saved
  activation per adapter).
- Not modelled: time holds: the step-time model doesn't read the rank. On a record with only a parameter count (no layer shape), memory abstains
  too. Quality.

**LoRA alpha** · `lora_alpha`
- What: how strongly the LoRA update is scaled.
- When: tuning quality.
- Moves: quality only.
- Not modelled: quality. No memory or time effect. A free knob.

**LoRA dropout** · `lora_dropout`
- What: dropout on the LoRA path.
- When: tuning quality.
- Moves: quality only.
- Not modelled: quality. Not a free knob (at 0 PEFT drops the dropout and its saved mask), so
  moving it marks memory and time Extrapolated.

### Precision and memory savers

**Precision** · `precision` (aliases with no label: `dtype`, `torch_dtype`, `ptdtype`)
- What: the number format training computes in: fp32, bf16, fp16, fp8, or mixed (bf16 kernels
  over fp32 weights).
- When: bf16 on a GPU that supports it is the usual memory and speed win.
- Moves: memory: activations and logits shrink with a 2-byte format. With an fp32 model and bf16
  set, weights stay fp32 (mixed precision); a model loaded in bf16 trains pure bf16 with bf16
  optimizer state, which saves the most. fp8 keeps bf16 weights and an fp32 master copy, so it
  doesn't save weight memory here. Time: scaled by the GPU's peak for the two formats (70% of the
  step) and by bytes (the rest) [guess: engine/api.py `_dtype_ratio` `compute_share`, untagged in
  code], Extrapolated whenever it differs from what the probes ran. A format with no
  fast path on the GPU (bf16 on a T4) abstains.
- Not modelled: training-stability risk (fp16 blow-ups).

**Weight quantisation** · `quantization`
- What: storing frozen weights in fewer bits (4-bit NF4, 8-bit, GPTQ, AWQ).
- When: to fit a big model for QLoRA, eval or serving.
- Moves: memory down: NF4 is about 0.56 bytes a weight against 2 for bf16 [derived:
  engine/memory.py `QUANT_BYTES`, 4 bits plus one fp32 scale per 64 weights].
- Not modelled: time (it holds), quality.

**Optimiser** · `optimizer`
- What: the update rule (AdamW, 8-bit AdamW, Lion, SGD, Adafactor, ...).
- When: an 8-bit or lighter optimizer is the cheapest memory win for full fine-tuning.
- Moves: memory: AdamW keeps 2 states a trainable weight, Lion 1, SGD none, 8-bit variants 1 byte
  a state, Adafactor a factored state [derived: engine/memory.py `OPTIMIZERS`]. An optimizer the
  table doesn't know is counted as AdamW, the conservative choice.
- Not modelled: time (it holds), risk, quality.

**Checkpointing** · `gradient_checkpointing`
- What: activation checkpointing: dropping saved activations in the forward pass and recomputing
  them in the backward pass.
- When: memory is tight and you can spend some time.
- Moves: memory down (full: one input per layer is kept, plus one layer's activations; selective
  drops the attention-score terms). Time up: one extra forward pass inside backward, added only
  when the probes ran with it off.
- Not modelled: turning it off when the probes ran with it on doesn't take the recompute out
  (the number keeps the recompute, labelled Extrapolated, "is off the probed values"; the band
  isn't widened). On a record with only a parameter count, memory abstains.

**Eval without gradients** · `eval_no_grad`
- What: running evaluation under no_grad, so no activations are kept for a backward pass.
- When: it's off and the eval pass sets the peak.
- Moves: memory: on, the eval pass holds one layer's activations; off, all of them.
- Not modelled: time (it holds). On a record with only a parameter count, memory abstains.

**Attention kernel** · `attn`
- What: how attention is computed (eager, SDPA, FlashAttention, ...).
- When: eager attention at long sequences is a common hidden memory cost.
- Moves: memory: eager keeps the full attention-score matrix (grows with the square of sequence
  length); the fused kernels don't. Unset counts as a fused kernel.
- Not modelled: time (it holds). On a record with only
  a parameter count, memory abstains.

No label · `compile`
- What: torch.compile, which fuses operations for speed.
- When: a long run where a few minutes of compiling pays back.
- Moves: time and cost, in principle down.
- Not modelled: both. Speed-up measured by a verify run, not predicted.

**Channels-last layout** · `channels_last`
- What: a memory layout for image tensors that convolution kernels run faster on.
- When: vision models on tensor-core GPUs.
- Moves: time and cost, in principle down.
- Not modelled: both. Speed-up measured by a verify run, not predicted.

### Optimisation hyperparameters

All four are free knobs except where noted: moving them leaves memory and time as they were.

**Learning rate** · `lr`
- What: how big each update step is.
- When: tuning, or when the loss blows up.
- Moves: risk and quality only.
- Not modelled: both. It would feed the LR range test (`probes.lr_range`, off) and the divergence
  classifier (off).

**Learning rate schedule** · `lr_schedule`
- What: how the learning rate changes over the run (constant, cosine, linear decay).
- When: tuning.
- Moves: risk and quality only.
- Not modelled: both.

**Weight decay** · `weight_decay`
- What: a pull of the weights toward zero each update.
- When: tuning against overfitting.
- Moves: quality only.
- Not modelled: quality.

**Gradient clip** · `grad_clip`
- What: a cap on the size of each update's gradient.
- When: to stop occasional loss spikes.
- Moves: risk and quality only.
- Not modelled: both.

### Hardware

**GPU** · `gpu_type`
- What: the GPU model the run uses. The dial lists the 11 GPUs Modal offers (not the
  MI300X), or only your allowed GPUs if you set any.
- When: to compare cost and time across cards, or to fit a bigger job.
- Moves: memory: the usable memory per card changes, so the out-of-memory chance changes; the peak
  itself barely does, except that a move replaces the probes' measured CUDA context with a 0.75
  GiB guess [guess: engine/memory.py `DEFAULT_CONTEXT_BYTES`, "typically 0.5-1 GB"], because
  `engine.context_same_stack` is off. Time: carried to the new card by the cross-GPU roofline
  model, Extrapolated, with a held-out band wider than plus or minus 50% on every card in today's table and the reason
  "rough, no probe on this GPU; run a probe here". Across vendors (NVIDIA to AMD) time abstains.
  Cost: time x the card's hourly rate; a card with no price abstains.
- Not modelled: across vendors, time and cost. See The GPU options for the numbers.

**GPUs** · `gpus`
- What: how many GPUs the run uses in data parallel.
- When: to finish sooner, or to shard a model that doesn't fit one card.
- Moves: memory per card: down under FSDP or ZeRO sharding; up under plain DDP, by one copy of the
  trainable gradients (its bucket copy); unchanged when the script sets no distributed strategy. Time per step up by the estimated gradient all-reduce; steps down (the
  effective batch grows). Cost: time x GPUs x rate. Always Extrapolated: "the gradient
  all-reduce is estimated, not measured".
- Not modelled: measured multi-GPU communication. Probes run on one GPU; if a record's probes ran
  on more, another count abstains. The all-reduce band is a guess [guess: engine/steptime.py
  `SIGMA_COMM` 0.35, `EXPOSED_COMM_SHARE` 0.3].

No label · `nodes`
- What: how many machines the GPUs are split across.
- When: planning a multi-node run on a cluster.
- Moves: time and cost: a node-to-node all-reduce term is added over the network. GPUs must split
  evenly across nodes, or time abstains.
- Not modelled: analytical only, never measured: "no multi-node run was measured"; its band is a
  guess [guess: engine/steptime.py `SIGMA_MULTINODE` 0.5].

No label · `inter_node_gbps`
- What: the network bandwidth between nodes, in Gbit/s.
- When: your cluster's network is known.
- Moves: time and cost down as it rises (the network term shrinks). Default 200 Gbit/s
  [guess: engine/steptime.py `INTER_NODE_GBPS`].
- Not modelled: as for nodes.

**Tensor parallel** · `tensor_parallel`
- What: splitting each layer's matrices across GPUs.
- When: a model too big for one card even with sharding.
- Moves: memory per card down: weights and state divide by the split, and the parts of the
  activations Megatron splits.
- Not modelled: time and cost (time holds).

No label · `pipeline_parallel`
- What: splitting the model's layers into stages on different GPUs.
- When: very deep models across several cards.
- Moves: memory per card: weights and state divide by the stages; activations don't shrink on the
  first stage (it keeps that many micro-batches in flight).
- Not modelled: time and cost (time holds).

### RL

The RL readouts (groups with no reward spread, rollout time, truncation) need `engine.rl`, which
is on in production.

**Group size** · `group_size`
- What: how many responses GRPO samples per prompt.
- When: bigger groups give more prompts a learning signal; smaller ones are cheaper.
- Moves: memory up (more sequences in the KV cache: the stored attention keys and values for each
  sequence being generated). Rollout time up (a batch waits for its longest response, and more
  responses make a longer longest one). Risk: the share of groups that teach nothing, p^G +
  (1 - p)^G for a prompt with pass rate p, falls as the group grows.
- Not modelled: the zero-spread readout abstains below 8 prompts [guess: engine/rl.py
  `MIN_PROMPTS`, checked against one pair of real runs, Nov 1].

**Max completion length** · `max_completion_len`
- What: the cap on each generated response, in tokens.
- When: raise it if many rollouts are cut; lower it to save time.
- Moves: memory up (KV cache). Rollout time up (the cap bounds the longest response). Risk: the
  share of rollouts cut at the cap falls as it rises.
- Not modelled: nothing beyond the general rules.

**Max prompt length** · `max_prompt_len`
- What: the cap on each prompt, in tokens.
- When: your prompts are long or being cut.
- Moves: memory and time per step up (prompt plus completion is the sequence on the work axis).
- Not modelled: nothing beyond the general rules.

No label · `rl_iterations`
- What: how many RL iterations the run makes.
- When: planning run length.
- Moves: time and cost, in principle.
- Not modelled: time and cost: Preflight doesn't read it yet (time holds); run length comes from
  `max_steps` or epochs. Quality.

**vLLM rollouts** · `use_vllm`
- What: generating rollouts with vLLM instead of Hugging Face generate.
- When: rollouts dominate the step.
- Moves: memory: with a GPU memory share set, vLLM's reserve replaces the KV cache in the peak.
- Not modelled: time (it holds).

**Loss type** · `loss_type`
- What: the GRPO loss variant (DAPO, Dr. GRPO, ...).
- When: tuning length bias.
- Moves: risk and quality: these change length bias, not cost.
- Not modelled: both. Not a free knob (it changes how long generations run).

**KL coefficient** · `kl_beta`
- What: how strongly training is pulled back toward the reference model.
- When: tuning stability.
- Moves: risk and quality.
- Not modelled: both. At 0, TRL loads no reference model, which would save memory; the engine
  doesn't model that, so moving it marks memory and time Extrapolated.

**Clip range (low)** · `clip_low`, **Clip range (high)** · `clip_high`
- What: how far one update may move each token's probability.
- When: tuning stability.
- Moves: risk and quality.
- Not modelled: both. Free knobs.

**Temperature** · `temperature`
- What: how random sampling is.
- When: tuning exploration.
- Moves: risk and quality: it changes pass rates.
- Not modelled: both; re-probe to measure. Not a free knob.

### Generation and serving

The serving simulator (`engine.serving`) is **off in production**. Without it, serving knobs that
only the simulator reads don't move any number; they only change the status.

**Max new tokens** · `max_new_tokens`
- What: the cap on tokens generated per request.
- When: outputs are cut, or generation takes too long.
- Moves: memory up (KV cache): with no sequence length set, the memory model sizes each sequence
  at prompt plus new tokens. It isn't on the work axis, so time holds.
- Not modelled: the risk and time effects through serving need `engine.serving` (off).

**Max model length** · `max_model_len`
- What: the longest prompt plus output a server accepts.
- When: to fit more sequences in memory, or to accept longer prompts.
- Moves: memory and time per step, on the work axis, when no sequence length is set.
- Not modelled: its serving effects (refused prompts, KV capacity) need `engine.serving` (off).

**Sequences in flight** · `max_num_seqs`
- What: how many requests a server runs at once.
- When: tuning serving throughput.
- Moves: memory and time, through the serving simulator only.
- Not modelled: in production (simulator off), no number moves; moved off the probed value,
  memory is labelled Extrapolated and time holds.

No label · `request_rate`
- What: requests per second arriving at an online server (Poisson arrivals).
- When: planning online serving.
- Moves: time and cost, through the serving simulator only.
- Not modelled: in production, time holds, as for sequences in flight.

No label · `max_batched_tokens`
- What: the tokens a server processes in one scheduler step.
- When: tuning serving latency against throughput.
- Moves: memory and time, through the serving simulator only.
- Not modelled: in production, as for sequences in flight.

**GPU memory share** · `gpu_memory_utilization`
- What: the share of the card vLLM reserves up front.
- When: vLLM shares the GPU with training (RL), or for serving capacity.
- Moves: memory up as it rises, when vLLM generates on the training GPU (its reserve is part of the
  peak).
- Not modelled: time (it holds). Serving capacity needs
  `engine.serving` (off).

**KV-cache dtype** · `kv_cache_dtype`
- What: the number format of the stored attention keys and values.
- When: fp8 to fit more or longer sequences.
- Moves: memory: fp8 halves the KV cache against bf16.
- Not modelled: time (it holds), quality.

**Prefix caching** · `prefix_caching`
- What: reusing the stored keys and values of a shared prompt start.
- When: many requests share a long system prompt.
- Moves: time and cost, in principle down.
- Not modelled: both; the serving model would use the measured hit rate, and it is off.

### Retrieval

**HNSW links (M)** · `index_m`
- What: how many links each vector keeps in an HNSW index.
- When: trading recall for index memory and speed.
- Moves: memory and time, in principle.
- Not modelled: memory and time: Preflight doesn't read this knob yet (time holds). Quality.

**IVF lists** · `index_nlist`
- What: how many clusters an IVF index splits vectors into.
- When: trading recall for search speed.
- Moves: time, in principle.
- Not modelled: time holds: Preflight doesn't read this knob yet. Quality.

### Agents

Agent and API runs are measured by sampling 20 to 50 real tasks [cited: engine/agent_cost.py
docstring], not by GPU probes. Cost and tokens are projected from the sample with a bootstrap
interval, widened for runaway tasks the sample may have missed.

**Model** · `model`
- What: which API model the agent calls.
- When: comparing a cheaper or stronger model.
- Moves: cost, time and runaway risk, from the tasks sampled on that model.
- Not modelled: a model with no sampled task abstains: "no sampled task ran on {model}; sample it
  first". Quality.

**Prompt caching** · `prompt_caching`
- What: the API's discount for reused prompt prefixes.
- When: long, repeated system prompts.
- Moves: cost, in principle down.
- Not modelled: moved off what was sampled, cost abstains ("prompt-cache pricing needs the API
  price table; sample with caching on first").

**Concurrency** · `concurrency`
- What: how many tasks run at once.
- When: to finish a large batch of tasks sooner.
- Moves: time down: task-seconds divided by concurrency, until the API's tokens-per-minute or
  requests-per-minute limit binds; then the limit sets the time.
- Not modelled: risk.

**Max turns** · `max_turns`
- What: the most model calls one task may make.
- When: to cap runaway tasks.
- Moves: cost, tokens and time down for tasks it cuts (they stop at the cap, unfinished, and the
  reason counts them); runaway risk changes with the cut tasks. Raised above a cap the sample hit,
  what those tasks spend past it is unknown, and the reason says so.
- Not modelled: quality (cut tasks don't finish).

**Tasks** · `n_tasks`
- What: how many tasks the full run has.
- When: planning the full benchmark.
- Moves: cost and time, scaled from the sample.
- Not modelled: nothing beyond that.

**Tool retry cap** · `tool_retry_cap`
- What: how many times a failing tool call is retried.
- When: to stop retry loops.
- Moves: cost, time and risk, in principle.
- Not modelled: retry loops are counted, not yet simulated.

**Reasoning effort** · `reasoning_effort`
- What: how much the model thinks before answering.
- When: trading quality for cost.
- Moves: cost and time, in principle.
- Not modelled: cost and time. Preflight doesn't read it yet: a move changes no number, and the
  site doesn't say so yet. Sample the tasks again at the effort you want.

**Context limit** · `context_limit`
- What: the model's context window, in tokens.
- When: checking whether long tasks overflow.
- Moves: the "Context window filled" readout (the share the fullest 5% of tasks reach) falls as
  the limit rises. If any sampled task would overflow, it abstains and says what share would.
- Not modelled: cost and risk: the engine computes only the fill share. Quality.

### Distributed and kernels

**FSDP mode** · `fsdp_mode`
- What: how FSDP shards the model: full_shard, shard_grad_op, hybrid_shard, no_shard.
- When: with more than one GPU, to fit a bigger model.
- Moves: memory per card down with more sharding (full_shard splits weights, gradients and
  optimizer state; shard_grad_op splits gradients and state; no_shard is DDP). Only applies with
  GPUs above 1.
- Not modelled: time holds. The all-reduce term uses the script's own distributed setting (FSDP
  or ZeRO-3 traffic counts 1.5 times DDP's [derived: engine/steptime.py]; no setting counts as
  DDP), not this knob. Communication is analytical, as for GPUs.

No label · `deepspeed_config`
- What: a DeepSpeed config; the ZeRO stage (0 to 3) is read from it.
- When: with more than one GPU.
- Moves: memory per card down with higher stages, as for FSDP. A config with no stage counts as
  ZeRO-2.
- Not modelled: offload (optimizer or parameter offload in the config doesn't lower memory
  here); time holds, as for FSDP mode.

No label · `tf32`
- What: letting fp32 matrix multiplies run on TF32 tensor cores.
- When: training in fp32 on Ampere or newer.
- Moves: time, when precision also moves off the probed value: it picks the fp32 matmul peak the
  precision scaling uses.
- Not modelled: on its own, it doesn't change the number; moved off the probed value it marks time
  Extrapolated.

### Storage

**Dump hidden states** · `dump_hidden_states`
- What: saving every layer's hidden states to disk during an eval sweep.
- When: interpretability or probing work.
- Moves: storage and time, in principle up.
- Not modelled: both; the storage model lands with campaign mode.

No label · `dump_attentions`
- What: saving attention maps (batch x heads x sequence squared, per layer).
- When: attention analysis.
- Moves: storage, time and memory, in principle up.
- Not modelled: all three. Memory abstains when it moves off the probed value.

Any knob id not on this list moves no number. Moved off the probed value, it marks memory and
time Extrapolated ("is off the probed values; the model carries it there").

## Other inputs you set

These aren't knobs, but they change what Preflight runs or shows.

**On New flight** (`web/src/app/screens/NewFlight.tsx`):

| Label | What it does |
|---|---|
| Where your job lives: Repo link, Upload, Run a command (and a Notebook tab) | The source. Repo: public GitHub only, pinned to its commit. Upload: .py or .ipynb plus small text files, 5 MB in all [derived: web/src/app/upload.ts], at most 2,000 files [derived: same]. Command: runs as written, no shell; `lm_eval` is the one command run by name [derived: web/src/app/command.ts], at most 64 words [derived: same] |
| Extra packages (optional) | PyPI names your code imports, up to 10 [derived: schema `JobRequest.extra_packages`]. Only names on Preflight's list, wheels only; a name off the list stops the check-in before any GPU time |
| What matters most: Lowest cost, Finish soonest, Most headroom | The objective the guide aims for |
| Your GPU: L4 or A10 (default), L40S, A100-SXM-80GB, H100-SXM-80GB, MI300X, Let Preflight pick | The GPU you ask for. See The GPU options for which ones probes actually run on |
| Private triage, by Muse Glimmer on Preflight's own GPU | Private mode: failed probes' logs are triaged on Preflight's own GPU and never sent to Baseten. Muse Spark still reads your code to draw the map |
| What are you trying to do? The guide reads this. | Free text for the guide |

**After a stop** (`web/src/app/CheckinHelp.tsx`): "Your script's path in the repo" sends the
check-in again reading only that file and the files it imports (`source.scope = entrypoint`);
"Extra packages" sends it again with packages added.

**In the cockpit, Your limits** (the guide panel): Allowed GPUs, Budget, Must finish within,
Quota (GPU-hours) and Don't change (a pin). The guide never suggests anything outside them;
moving a knob yourself is always up to you.

**In the job request but not on New flight** (`python/preflight/schema/record.py`
`Constraints`, sent through the API): `wall_time_cap_s` (a scheduler's per-job limit, like Slurm's)
adds the readout "chance the run outlasts the job limit"; `preemption` (none, a Poisson rule with a
mean time between interruptions, or a policy in words) adds "Work lost to a preemption", which
abstains unless a mean time is given; `gpu_types` limits which GPUs may be used; `max_gpus`,
`budget_usd`, `deadline` and `quota` are carried for the guide and the dials (a budget or wall-time
cap gives the dial a linear scale to that limit).

## The GPU options

Rows come from `python/preflight/engine/gpus.py` (78 GPUs, NVIDIA and AMD [derived: count of
`_SPECS`]), priced from `python/preflight/engine/prices` (table version 2026-11-01). The cockpit's
GPU dial lists the GPUs Modal offers; the rest of the table is there to recognize the GPU in a
record (your own card, through `preflight watch` or a local run) and to scale time to it.

### Where probes run

The public site probes on GPUs no bigger and no dearer than an L40S [derived: runner/modal_app.py
`GPU_CEILING`, probes/gpu_choice.py `under_ceiling`]. That leaves four:

| GPU | Card memory | Usable to PyTorch | Price per hour | Spec row |
|---|---|---|---|---|
| T4 | 16 GB | 14.74 GiB [cited: torch total_memory as published, gpus.py `usable_gib`] | $0.59 [cited: Modal list rate, July 2026] | datasheet |
| L4 | 24 GB | 22.06 GiB [measured: torch total_memory on Modal, Oct 27 smoke run, gpus.py] | $0.80 [cited: Modal list rate, July 2026] | derived |
| A10 | 24 GB | 22.08 GiB [derived: 92% of card memory, the ratio cited from the T4 and L4 rows, gpus.py] | $1.10 [cited: Modal list rate, July 2026] | datasheet |
| L40S | 48 GB | 44.16 GiB [derived: 92% of card memory, the ratio cited from the T4 and L4 rows, gpus.py] | $1.95 [cited: Modal list rate, July 2026] | datasheet |

- **"L4 or A10"** (the default) starts on an L4. If none is free within 2 minutes it tries an A10,
  then an L40S [derived: runner/sandbox_runner.py `SCHEDULE_WAIT_S` 120, runner/pipeline.py
  `GPU_FALLBACK`]. A script that runs out of memory may be measured again on the next bigger card.
- **"Let Preflight pick"** takes the cheapest card with usable memory at least 1.3 times a rough
  estimate of the need [simulated: probes/gpu_choice.py `MARGIN`], from the ladder L4, L40S
  (A100-80GB and H200 sit above the cap).
- **Nov 5 to 8 (judging)**, a job that would start on an L4 starts on an A10 when your allowed GPUs
  include it [derived: runner/judging.py].
- A T4 can be probed when you or the guide ask for it, but the default never picks it: it has no
  bf16 tensor path, so a bf16 probe there would time an emulated path.

### Predicted from the cross-GPU model

Pick one of these and the probes run on a card under the cap; the boarding log says "probes here
run on GPUs up to L40S; {GPU} numbers come from the cross-GPU model".

| GPU | Card memory | Usable to PyTorch | Price per hour | Spec row |
|---|---|---|---|---|
| A100-SXM-40GB | 40 GB | 39.39 GiB [cited: torch total_memory as published, gpus.py] | $2.10 [cited: modal.com/pricing, Nov 1] | datasheet |
| A100-SXM-80GB | 80 GB | 79.15 GiB [cited: torch total_memory as published, gpus.py] | $2.50 [cited: Modal list rate, July 2026] | datasheet |
| RTX-PRO-6000-96GB | 96 GB | 88.32 GiB [derived: 92% of card memory, the ratio cited from the T4 and L4 rows, gpus.py] | $3.03 [cited: Modal list rate, July 2026] | secondary |
| H100-SXM-80GB | 80 GB | 79.18 GiB [cited: torch total_memory as published, gpus.py] | $3.95 [cited: Modal list rate, July 2026] | derived |
| H200-SXM-141GB | 141 GB | 139.59 GiB [cited: 99% of HBM, gpus.py] | $4.54 [cited: Modal list rate, July 2026] | derived |
| B200-SXM-180GB | 180 GB | 178.2 GiB [cited: 99% of HBM, gpus.py] | $6.25 [cited: Modal list rate, July 2026] | datasheet |
| B300-SXM-288GB | 288 GB | 285.12 GiB [cited: 99% of HBM, gpus.py] | $7.10 [cited: Modal list rate, July 2026] | derived |
| MI300X (AMD) | 192 GB | 190.08 GiB [cited: 99% of HBM, gpus.py] | $1.99 [cited: the team's AMD droplet, Nov 1] | datasheet |

How far to trust a move to one of these:

- **Time on another NVIDIA card** is Extrapolated, with a band from held-out pairs of corpus runs.
  Its 90% half-width is 0.44 to 0.64 in log terms depending on the card, a factor of 1.55 to 1.90
  either way [derived: engine/_calibration_table.py `time.step.cross` cells, exp of the
  half-width], so it is usually shown as rough. On the corpus, the roofline model put 3,826 of
  6,694 NVIDIA pairs within 20% (57%) [simulated: docs/validation/table-final-nov3.md], and full
  runs on another GPU 1 of 3 [measured: same table].
- **MI300X**: cross-vendor step time is within 20% on only 22% of pairs (254 of 1,147, Wilson
  95% 20% to 25%) [simulated: MI300X corpus slice, 295 configs, Nov 2; docs/validation/table-final-nov3.md].
  NVIDIA torch 2.8.0 to MI300X torch 2.5.1+rocm6.2: card and software both differ. So a move
  across vendors abstains: time and cost show "Not measured here". Memory and the
  out-of-memory chance are still computed against the MI300X's memory.

### What the table's marks mean

- **Spec row** (`confidence` in gpus.py): **datasheet** is the vendor's figure; **derived** is
  worked out from a vendor figure (half a with-sparsity peak, or cores times the per-core rate);
  **secondary** comes from a spec database. The simulated cross-GPU band is 1.5 times wider for a
  secondary card [guess: engine/steptime.py `SIGMA_SECONDARY_SPECS`]; in production the held-out
  cross-GPU band, never narrower, sets the width.
- **Usable memory** is what PyTorch reports, which is less than the box says. Rows with a
  `torch.cuda` figure carry it (the L4's measured on Modal, the others as published). The others are estimated: 99% of the
  memory for HBM cards [cited: published torch total_memory for the A100 and H100 rows], 92% for
  data-centre GDDR cards with ECC on [cited: published figures for the T4 and L4 rows], 98.5% for
  other GDDR cards [guess].
- **Price** is the on-demand list rate, billed per second on Modal. Your own cluster's rate can
  replace it in the cockpit; cost estimates carry the GPU-hours and the rate separately for that.

## Status words

Every number carries one of four statuses (`EstimateStatus` in
`python/preflight/schema/prediction.py`). The site shows them as:

| Status | Site shows | What it means | How much to trust it |
|---|---|---|---|
| `measured` | **Measured** | A probe ran at exactly these settings on this GPU. Run length, the learning rate and the free knobs don't count as differences. | The most. The memory band is held to the calibrated width [simulated: engine/memory.py `HALF_WIDTH_CALIBRATED` 0.107 in log, about plus or minus 11%]. Time to finish and cost are never Measured: the step is, the run length is multiplied out, so they read Predicted at best. The out-of-memory chance is Measured only when a probe ran out of memory there. |
| `predicted` | **Predicted** | Worked out from nearby measured settings, inside what was measured (from the smallest probe up to twice the largest, on the work axis), with every other knob at a probed value. | "Should hold." Memory bands are about plus or minus 11% with two or more probes and 17% with one [simulated: engine/memory.py 0.107 and 0.160 in log]; same-GPU step time about plus or minus 14% [derived: 1.645 x `SIGMA_SAME_GPU` 0.08, simulated]. Held-out bands replace these where the calibration table has a cell. |
| `extrapolated` | **Extrapolated** | Worked out past what was measured: below the smallest probe, a knob off the probed values, a precision change, another GPU, more than one GPU, a knob the model doesn't capture, a peak at a moment no probe ran (eval), a sequence length past the dataset scan. | "A rough guide." The band is wider, and the reason says why. A GPU move is labelled rough when its band passes plus or minus 50%. |
| `abstain` | **Not measured here** | Nothing measured is close enough to say. Past twice the largest probe; a band wider than plus or minus 50% [cited: Hackathon Project Goals, engine/memory.py `ABSTAIN_HALF_WIDTH`]; a memory knob the engine doesn't model; across vendors; a GPU or price not in the table; a precision with no fast path; an unknown run length; an agent model that wasn't sampled. | No number is shown, on purpose. Move back toward a measured setting, or run a probe there. |

Related words on the site:

- **Ran out of memory here**: a probe ran out of memory at these settings on this GPU. Memory
  shows no number; the out-of-memory chance is 1 and Measured.
- **Between measured points**: the cockpit's precomputed grid has no value at this exact setting.
- **Example numbers**: a sample flight whose numbers show how the cockpit works. None were
  measured. Samples marked Measured come from a real run.

How much the intervals have held so far: 90% intervals covered 10 of 18 checks against real full
runs, against a target of at least 88% [measured: docs/validation/table-final-nov3.md, "Interval
coverage"]. Treat Predicted bands as narrower than they should be until that row passes.

### Out-of-memory words

The chance of running out of memory is shown with one of three words, and every reason starts
with the word:

| Word | Site shows | Chance of running out of memory |
|---|---|---|
| `fits` | **Fits**; "Fits on {gpu}." | below 1% [simulated: engine/memory.py `FITS_BELOW`] |
| `tight` | **Tight**; "Might run out of memory on {gpu} ({risk} chance)." | 1% up to 80% |
| `fails` | **Fails**; "Likely runs out of memory on {gpu} ({risk} chance)." | 80% or more [simulated: engine/memory.py `FAILS_AT`] |

A chance between 5% and 95% isn't verified on real outcomes yet, so the site shows that range
instead of a single number [derived: engine/oom.py `UNCERTAIN`]. Where held-out runs exist for the
GPU and job type, the chance comes from their errors ("from N held-out runs"); otherwise from the
model's own band ("simulated band"). On the fit map: dot fits, triangle tight, cross fails,
hatched not measured here. The words are simulated cut-offs, to be replaced by reliability-checked
bins. On real scripts so far, 10 of 11 out-of-memory outcomes were called right, with one miss
and no false alarm [measured: docs/validation/table-final-nov3.md, "OOM prediction"].

## What a check-in costs and how long it takes

**You pay nothing.** Preflight pays for the GPU minutes, with no account.

**What it costs Preflight.** Two production check-ins of the LoRA sample cost $0.21 in 10 min 21 s
and $0.11 in 4 min 59 s: 1 of 2 under the $1 and 5-minute target [measured:
docs/validation/table-final-nov3.md, "Cost of one Preflight run"; README.md]. The first run (Oct
28) took 6 min 45 s from upload to record, with 386 s of probes on one L4 [measured: README.md, job
`a-smoke-l4-main-4`]; its GPU cost $0.0858 [derived: measured probe time at the L4 list rate] and
Muse Spark's map $0.0178 [derived: reported tokens at list prices]. Those totals leave out the
sandbox's CPU and memory.

**Per-job caps.** A job may spend up to $0.60 of GPU [guess: runner/pipeline.py `JOB_BUDGET_USD`]
and 15 minutes of probing [guess: runner/pipeline.py `JOB_PROBE_WALL_S`]; one probe point is
stopped at 6 minutes [guess: runner/pipeline.py `PROBE_TIMEOUT_S`]. Past a cap, probing stops and
what was measured stays. Muse Spark Open's check of model-written findings stops at $0.10 a job and
falls back to code only [guess: harness/claim_check.py `JOB_CAP_USD`].

**Per visitor, per day** (days run midnight to midnight, US Eastern):

| Limit | Value | Source |
|---|---|---|
| Probe jobs (a check-in, or a probe or check from the cockpit) | 3 a day | [derived: runner/queue.py `VISITOR_DAILY_JOBS`] |
| Jobs queued or running at once | 3 | [simulated: runner/queue.py `MAX_QUEUED_PER_VISITOR`, a load test on a simulated clock] |
| Free resend after a check-in stopped at an import (nothing measured) | 1 a day; doesn't count against the 3 jobs, but the site's GPU cap still holds | [derived: runner/queue.py `FREE_RESENDS_PER_DAY`; `runner.free_resend` on in production] |
| Guide rounds | 64 a day per visitor, 256 per network address | [derived: api/preflight_api/ledger.py, from tours measured Oct 27 / guess for the address limit] |
| Guide spend | $0.40 a tour | [derived: api/preflight_api/ledger.py] |

**For the whole site, per day:** a GPU dollar cap of $15 (the worker image's
`PREFLIGHT_DAILY_GPU_CAP_USD`, Chiya, Nov 3; live from the runner deployed at 10 PM ET), and $10 of guide spend [cited:
api/preflight_api/ledger.py, pending confirmation]. Past your 3 jobs or the site's GPU cap, the
site opens samples and shared records only, until midnight ET ("You've run today's 3 probe jobs" or
"Preflight has used today's GPU budget").

**Waiting for a GPU.** The boarding screen shows "Waiting for a GPU" and, after a minute, how long
so far. The default request moves from an L4 to an A10 after 2 minutes, and to an L40S after 2 more; the L40S is waited for up to the cap
[derived: runner/sandbox_runner.py `SCHEDULE_WAIT_S`]; the whole wait is capped at 20 minutes,
after which the probe fails as "no GPU could be scheduled" [derived: runner/sandbox_runner.py
`WAIT_CAP_S`]. L4 waits of 14 to 26 minutes were seen Oct 31 to Nov 1, before the fallback
[measured: observed waits, runner/sandbox_runner.py comment]. A busy queue refuses a check-in with "Try again in
{s} seconds".

**Private triage.** Muse Glimmer naps when nobody is using it. Waking it took 5 to 8 minutes on
Nov 1; after that a triage takes seconds [measured: en.json `newflight.private.wait`; README.md].

## Words used here

- **Abstain**: show no number because nothing measured is close enough ("Not measured here").
- **Activations**: the intermediate values a forward pass keeps so the backward pass can compute gradients.
- **All-reduce**: the step where GPUs in data parallel add up their gradients over the link
  between them.
- **Band, interval**: the range a readout should fall inside 9 times in 10.
- **Baseten**: the hosting service that runs Muse Spark Open for Preflight's failure triage.
- **Bootstrap interval**: a range found by resampling the sampled tasks many times and seeing how the total varies.
- **Corpus**: Preflight's collection of measured runs (records of real jobs) that the models are fitted and checked on.
- **Cross-GPU model**: the roofline model that carries a step measured on one GPU to another, by
  each kernel's compute and memory-bandwidth limits.
- **CUDA context**: memory the GPU driver and libraries take before your model loads.
- **DDP, data parallel**: every GPU holds a full copy of the model and works on its own slice of each batch.
- **Effective batch**: batch size x gradient accumulation x GPUs; what one update sees.
- **Free knob**: a knob that changes what the run learns but not one step's memory or time.
- **GRPO**: an RL method that samples a group of answers per prompt and learns from how they score against each other.
- **HBM, GDDR, ECC**: kinds of GPU memory (HBM on data-centre cards, GDDR on others); ECC is error correction, which reserves a little of it.
- **Held-out**: checked on runs the model wasn't fitted on, so the check is fair.
- **KV cache**: the stored attention keys and values for each sequence being generated.
- **lm_eval**: EleutherAI's lm-evaluation-harness, a common tool for scoring language models on benchmarks.
- **Log half-width**: a band's half-width on a log scale; 0.44 means a factor of e^0.44, about 1.55, either way.
- **Logits**: the model's raw score for every word in its vocabulary at each position; large for big vocabularies.
- **Megatron**: NVIDIA's library for splitting a model's layers across GPUs (tensor parallel).
- **Micro-batch, 1F1B**: in pipeline parallel, a batch is cut into micro-batches that flow through the stages; 1F1B is the usual schedule (one forward, one backward in turn).
- **Optimizer state**: what the optimizer keeps per weight between updates (AdamW keeps two running averages).
- **PEFT, QLoRA**: Hugging Face's library for LoRA-style training; QLoRA is LoRA on a model whose frozen weights are stored in 4 bits.
- **Poisson arrivals**: requests arriving at random, independent moments at a set average rate.
- **Probe**: a short, real run of your script on a rented GPU, a few profiled steps at one setting.
- **PyPI, wheel**: Python's public package index; a wheel is a ready-built package, installed without compiling.
- **Readout**: one of memory, time, cost and risk.
- **Reserved pool**: memory PyTorch's allocator holds on to after use, so the GPU shows more in use than the model needs.
- **TRL**: Hugging Face's library for RL and preference training of language models.
- **vLLM**: a fast inference server for language models; also used to generate RL rollouts.
- **Wilson interval**: a 95% range for a success rate, valid for small counts.
- **Work axis (work units)**: the one size measure memory and step time are fitted against,
  such as batch x sequence length.
- **ZeRO, FSDP**: ways to split weights, gradients and optimizer state across GPUs so each card
  holds a share.
