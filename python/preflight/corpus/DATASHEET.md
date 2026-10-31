# Preflight corpus: datasheet

Following *Datasheets for Datasets* (Gebru et al.). Version: corpus@0.3.0, exported Oct 31, 2026 (`remote-runs` folders `b-corpus-export-3` and `b-corpus-export-hf-2`). Owner: workstream B.

## Motivation

**Why it exists.** Preflight predicts a training job's memory, step time and cost from a few minutes of probes, and every prediction carries an interval. Those intervals have to be checked on runs the models never saw. The corpus is that check. It was generated on purpose: each config is a probe-like job measured on real GPUs, so calibration data looks like what users send.

**Who paid.** Modal credits on the team's account, about $10 in total for the NVIDIA slices. Per run, see `docs/status/b.md`.

## Composition

**What a row is.** One training config, measured once on one GPU. Rows are JSON, one gzipped JSON-lines file per GPU and kind (`kernels/<GPU>.jsonl.gz`, `hf/<GPU>.jsonl.gz`). Each row holds:
- `config`: the family, all its parameters, and the code and library versions that are part of its key;
- `key`: content-addressed from the config, so a config is written once and re-runs resume;
- `gpu`: the board the row ran on, as PyTorch reported it;
- `status`: `ok` or `oom`; failures carry A's stable `error_code`;
- `timing`: five unprofiled steps after two warm-up optimizer steps, with the median step and the forward, backward and optimizer phases;
- `memory`: peak allocated and reserved bytes;
- `ops`: a per-op table from a three-step profiler window, with self device time, FLOPs and input shapes;
- `graph`: the FX module graph with shapes, for grouped configs;
- `n_params`, `n_trainable`, `work_units` and `gpu_seconds` (billed).

**How many** (measured):

| GPU | Grouped configs | OOM | HF jobs | OOM |
| --- | --- | --- | --- | --- |
| L4 | 577 | 11 | 50 | 1 |
| A10 | 577 | 11 | 50 | 1 |
| L40S | 195 | 1 | 0 | 0 |
| T4 | 75 | 4 | 0 | 0 |
| A100-PCIe-80GB | 42 | 0 | 0 | 0 |
| A100-SXM-80GB | 53 | 0 | 0 | 0 |

**Families.** The grouped configs come in groups of five points: three probe-like points and two queries per architecture. 111 distinct groups ran in all. The architectures are small models built in the corpus code:
- `gpt`: decoder-only transformers;
- `encoder`: BERT-style;
- `lora`: synthetic LoRA transformers;
- `cnn`: ResNet-style;
- `mlp`.

The HF jobs are 10 groups of five on real Hugging Face checkpoints, LoRA in bf16 and QLoRA in nf4. The checkpoints are Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, HuggingFaceTB/SmolLM2-360M, HuggingFaceTB/SmolLM2-1.7B and TinyLlama/TinyLlama-1.1B-Chat-v1.0.

**Splits.**
- **Calibration:** every non-A100 row, used leave-one-architecture-out for coverage.
- **Cross-GPU holdout:** every A100 row, never calibrated on and scored once (`docs/decisions/calibration-l4-phase1.md`).
- **Not in calibration:** the synthetic `lora` family (lens `finetune_synthetic`). Its scores are reported, but users send real models.

**Data inside the jobs.** None. Every job trains on random token ids or random images, so the corpus holds measurements of compute, not of any text or image dataset. The `divergence` kind, not yet run, would train on Tiny Shakespeare (public domain).

**Personal or sensitive data.** None.

## Collection

The corpus was collected between Oct 28 and 31, 2026, on Modal, in a pinned image (torch 2.8.0, plus pinned transformers, peft and bitsandbytes for the HF jobs). Each config ran in its own process with a deadline, so a hang became a `timeout` row and a kernel fault couldn't poison the next config. Sweeps were capped by a worst-case container-time bound and approved phase by phase. Every run's GPU, minutes and dollars are in `docs/status/b.md`.

## Known biases and limits

- **Small models.** Everything is sized for a single 24 GB card. Grouped models range from 0.9M to 667M parameters, with a median of 24M. The HF checkpoints range from 360M to 1.7B. Multi-GPU, FSDP, and models that need 80 GB aren't covered.
- **GPU mix.** L4 and A10 dominate, L40S has a third of the configs and T4 a tenth. T4 has fp32 and fp16 groups only, because it has no bf16 path. AMD is planned (MI300X, the same 300 groups).
- **Modal picks the A100 board.** A request for an A100-80GB landed on PCIe or SXM. Rows are filed under the board they ran on.
- **Short windows.** Five timed steps after two warm-up steps. Long-run effects (thermal throttling, allocator fragmentation, data-loader stalls) aren't in it.
- **Op tables count device time three times.** The worker kept profiler ranges, aten ops and the CUDA kernels under them, each with its own self device time, and together they sum to about 2.6× the step. Use the `aten::` rows alone, which cover about 0.9 of the step (`scripts/fit_steptime_gbdt.py`). The next corpus version keeps one view.
- **Repeated configs.** The sampler for "100 grouped configs" drew 95 distinct ones on the A100. Counts above are distinct rows.
- **Synthetic inputs.** Random tokens give every row the same length, so padding and length variation aren't measured. Preflight's own probes measure them on the user's data.

## Uses

- **Intended:** calibrating and checking Preflight's intervals (memory, step time, P(OOM)), and fitting and scoring learned cost models with held-out architectures and GPUs.
- **Not intended:** benchmarking GPUs against each other for purchasing, or estimating how fast any specific production model trains. The models are small and the windows short.

## Distribution and licence

- **Measurements:** published with the Preflight repository under its licence (see `LICENSE`).
- **HF checkpoints:** the corpus contains no weights from them, only timings and memory of training them. Qwen2.5-0.5B, Qwen2.5-1.5B, the SmolLM2 models and TinyLlama are released under Apache 2.0 by their authors.

## Maintenance

Workstream B keeps it. New rows come only through `python -m preflight.corpus.sweep` or `modal_app.py`, which write each config once under its key. A new code or library version changes the key, so old and new rows never mix silently.
