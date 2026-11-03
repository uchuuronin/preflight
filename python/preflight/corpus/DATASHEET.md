# Preflight corpus: datasheet

Following *Datasheets for Datasets* (Gebru et al.). Version: **corpus 0.4.0, final for publication (Nov 2, 2026)**. Owner: workstream B. Version 0.3.0 (Oct 31, 2,339 rows) was the first published file; 0.4.0 keeps all its rows and adds the Nov 1 NVIDIA top-up (`b-topup-seed1`, `b-topup-seed0`), the Nov 2 realistic divergence sweep (`b-div-real-1`, `b-div-real-2`) and the Nov 2 MI300X slice (`b-amd-grouped`).

**Published file** (gap 16): `preflight-corpus-0.4.0.jsonl.gz`, **4,255 rows** (3,129 timing configs, 100 HF jobs, 1,026 divergence runs), 31,723,814 bytes, sha256 `8719593587b6b6741afa6c2f12f83883065119e0634c6b6ac91c55ceaa3101f4`, built by `python scripts/export_corpus.py --store <store> --divergence <Nov 1 rows> --divergence-real <Nov 2 rows> --out <dir>` (the same rows give the same bytes). It ships with `SHA256SUMS` and a dataset card (`README.md`, written by the same script). **On Hugging Face:** [huggingface.co/datasets/chaoschia/preflight-corpus](https://huggingface.co/datasets/chaoschia/preflight-corpus) holds 0.3.0 (commit `0aac218`), private until Chiya makes it public with the repository; 0.4.0 replaces it there before then.

## Motivation

**Why it exists.** Preflight predicts a training job's memory, step time and cost from a few minutes of probes, and every prediction carries an interval. Those intervals have to be checked on runs the models never saw. The corpus is that check. It was generated on purpose: each config is a probe-like job measured on real GPUs, so calibration data looks like what users send.

**Who paid.** Modal credits on the team's account: about $9.60 for the Oct 28 to 31 timing slices (the ledger counts each config's GPU-seconds; Modal's bill, which also counts container start-up, is higher: `b-corpus-l4` alone was billed an estimated $4.50 to $6.40 against its $0.88 in the ledger), about $11.55 for the Nov 1 divergence sweep with its smoke runs, about $16.54 for the Nov 1 top-up and about $3.78 for the Nov 2 realistic divergence sweep [ledger, `docs/status/b.md`]; the MI300X slice ran on a rented AMD Developer Cloud MI300X, $8.86 by the sweep's own uptime count [ledger]. The Muse Spark Open labels below were Baseten credit: $0.70 and $8.79. Per run, see `docs/status/b.md`.

## Composition

**What a row is.** One training config, measured once on one GPU. Rows are JSON, one gzipped JSON-lines file per GPU and kind in the store (`kernels/<GPU>.jsonl.gz`, `hf/<GPU>.jsonl.gz`), and one file in the published export. Each row holds:
- `config`: the family, all its parameters, and the code and library versions that are part of its key (so the same config has a different key on the NVIDIA rows, torch 2.8.0, and the MI300X rows, torch 2.5.1+rocm6.2: pair them on `family`, `kind` and `params`);
- `key`: content-addressed from the config, so a config is written once and re-runs resume;
- `gpu`: the board the row ran on, as PyTorch reported it; `env` holds the device name, its memory and SM count, and the torch, CUDA or HIP and Python versions;
- `status`: `ok` or `oom`; failures carry A's stable `error_code`;
- `timing`: five unprofiled steps after two warm-up optimizer steps, with the median step and the forward, backward and optimizer phases;
- `memory`: peak allocated and reserved bytes;
- `ops`: a per-op table from a three-step profiler window, with self device time, FLOPs and input shapes;
- `graph`: the FX module graph with shapes, for grouped configs;
- `n_params`, `n_trainable`, `work_units` and `gpu_seconds` (billed).

A divergence row has `config`, `key`, `gpu` and `status` too, and instead of timings: `series` (per step: loss, grad norm, the largest attention logit, the output logit norm, the update-to-weight ratio per layer, non-finite gradients, the loss scale), `label` (the D9 hindsight label: diverged, the onset step, the reason), `steps_run`, `stopped_early` and `data`.

The published file adds three fields to every row: `kind` (`kernels`, `hf`, `divergence` or `divergence_real`), `split` (below) and `source_file`.

**How many** (measured, in the 0.4.0 file):

| GPU | Timing configs | of which OOM | HF jobs | of which OOM | Divergence runs (Nov 1 + Nov 2) |
| --- | --- | --- | --- | --- | --- |
| A10 | 857 | 17 | 50 | 1 | 343 + 300 |
| L4 | 666 | 11 | 50 | 1 | 377 + 6 |
| L40S | 587 | 2 | 0 | 0 | 0 |
| H100-SXM-80GB | 492 | 0 | 0 | 0 | 0 |
| T4 | 137 | 8 | 0 | 0 | 0 |
| A100-SXM-80GB | 53 | 0 | 0 | 0 | 0 |
| A100-PCIe-80GB | 42 | 0 | 0 | 0 | 0 |
| **MI300X (AMD)** | **295** | 0 | 0 | 0 | 0 |
| All | 3,129 | 38 | 100 | 2 | 1,026 |

**Families.** The grouped configs come in groups of five points: three probe-like points and two queries per architecture. 162 distinct groups ran in all; the MI300X ran 59 of them (295 configs, the same configs as the NVIDIA rows of those groups). Models range from 0.9M to 670M parameters (median 24M). The architectures are small models built in the corpus code:
- `gpt`: decoder-only transformers;
- `encoder`: BERT-style;
- `lora`: synthetic LoRA transformers;
- `cnn`: ResNet-style;
- `mlp`.

The HF jobs are 10 groups of five on real Hugging Face checkpoints, LoRA in bf16 and QLoRA in nf4. The checkpoints are Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, HuggingFaceTB/SmolLM2-360M, HuggingFaceTB/SmolLM2-1.7B and TinyLlama/TinyLlama-1.1B-Chat-v1.0.

**Splits** (the `split` field).
- **Calibration:** every NVIDIA timing row off the A100 boards, used leave-one-architecture-out for coverage.
- **Cross-GPU holdout:** every A100 row and every MI300X row, never calibrated on and scored once (`docs/decisions/calibration-l4-phase1.md`; the MI300X: `corpus/reports/amd_cross_nov2.json`, NVIDIA to MI300X step time within 20% on 254 of 1,147 pairs, 22%).
- **Not in calibration:** the synthetic `lora` family (lens `finetune_synthetic`). Its scores are reported, but users send real models.
- **Divergence:** the 1,026 divergence runs of both kinds, which score the divergence classifier only.

**Data inside the jobs.** The `kernels` and `hf` kinds train on random token ids or random images, so they hold measurements of compute, not of any text or image dataset. Both divergence kinds train a small GPT (4 layers, d256, 600 steps) on Tiny Shakespeare (public domain) and keep each run's loss, grad-norm, logit and update series with its D9 label.

**Divergence runs** (measured, Nov 1): 720 runs, 377 on L4 and 343 on A10 (two draws of the same sampler, the board on every row), 44 diverged: 37 of them in fp16, 15 in the first 30 steps. They score the divergence classifier, which is not used: held-out AUROC 0.66 (95% 0.56 to 0.76) against a 0.85 target (`corpus/reports/divergence_nov1.json`).

**Realistic divergence runs** (`divergence_real`, measured, Nov 2): 306 runs (300 on A10; 6 on L4, which also ran on the A10) at usual settings: learning rate 1e-4 to 3e-2, no warm-up, no gradient clipping, and fp16 without a loss scaler for half the runs. 12 diverged, all fp16 without a scaler, at learning rates 0.0045 to 0.029. The classifier fitted on Nov 1's runs scores AUROC 0.90 (0.83 to 0.96) on them but with a lead of 0 steps, and the learning rate alone scores 0.95, so it stays off (`corpus/reports/divergence_real_nov2.json`).

**Failure labels by Muse Spark Open** (Nov 1; contract item, the second logged Spark Open job). Every row that ran out of memory (29) and every diverged run (44) was sent, with its own numbers, to Muse Spark Open on Baseten (`meta-models/muse-spark-open-1026`, effort low, structured output; `scripts/spark_label_corpus.py`) to name its failure mode, and the answers were compared with B's rule labels, which were never in the prompt: the engine's largest memory term at the row's settings for an out-of-memory row, the D9 reason for a diverged run.

| Task | Compared | Agree | Rate (Wilson 95%) | Where they differ |
| --- | --- | --- | --- | --- |
| Out-of-memory rows | 21 | 15 | 71% (50% to 86%) | the rule says logits for 9 rows, Spark Open activations for 3 of them; the rule says model state for 2, Spark Open activations; the rule says activations for 10, Spark Open "other" for 1. This is the real test |
| Diverged runs | 44 | 44 | 100% (92% to 100%) | nowhere; 37 non-finite and 7 blow-ups, which the numbers show plainly (a non-finite step is in the evidence) |
| All | 65 | 59 | 91% (81% to 96%) | the diverged runs carry most of this |

Two passes (Chiya, Nov 1). The first answered 64 of 73 rows: of its 16 failed calls, 8 ran into the 8,192-token output cap (an answer that never closed its JSON), 6 timed out at 20 s, and 2 re-asks were cut off when the row's 30 s ran out. The second asked only those 9 rows again with a 16,384-token cap and 90 s, and answered 8; one row ran past the cap on both tries and stays unanswered. 7 out-of-memory rows have no rule label (5 CNN rows have no analytical model, and 2 HF rows didn't record their model shape when they failed), so 65 of the 73 rows are compared. 90 calls in all, 42,454 input and 153,953 output tokens, $0.70 ($0.538 for the first pass and $0.167 for the second, under its $0.50). Every call, failed or not, is in `corpus/provenance/spark_open_labels_nov1.jsonl` with its tokens and cost; each row's label and reason are in `corpus/reports/spark_labels_nov1_rows.jsonl` and the agreement in `corpus/reports/spark_labels_nov1.json`. A timed-out call is logged at $0, though Baseten may have billed it.

**Whole-corpus labels by Muse Spark Open** (Nov 1, evening; Chiya: Baseten credit, up to $10). **These labels are written by a model, not measured, and nothing trains on them**: they sit beside the rows as a second opinion, and the agreement below is how far to trust them. Every one of the 2,339 rows of the 0.3.0 file was sent in batches of 8 rows of one kind (`scripts/spark_label_all.py`; same model and effort, rows as `r1` to `r8`, never the corpus key) with one question per kind: for a memory row, fitted or out of memory, which memory use is the largest part of its peak; for a divergence run, whether it stayed stable and, if not, how it failed. Neither the rule label nor the hindsight label was ever in the prompt. 1,997 rows were answered; 342 were not: 328 in 41 batches whose answer ran to the 12,000-token output cap twice, padding its JSON with whitespace, and 14 left out of 2 batches that otherwise answered. The 111 failed calls cost $5.86 of the $8.59. **Most of that spend was avoidable** (Chiya's review, Nov 2): the pilot just before this run (`b-spark-all-pilot`, B's status file) had a batch fail twice at the same 12,000-token cap, and the first pass had shown answers running to the cap without closing their JSON (8 of its 16 failed calls), yet the run went out with nothing against it. All 111 failed calls were cap hits. A stop after a batch's first cap hit would have saved $2.47 (70 retries, 41 of which failed again, and the 29 batches answered on retry would have gone unanswered); an answer schema that leaves no room for padding would have kept most of the $5.86.

| Task | Compared | Agree | Rate (Wilson 95%) | Where they differ |
| --- | --- | --- | --- | --- |
| Memory rows, all | 881 | 545 | 62% (59% to 65%) | the rule (the engine's largest analytical term, not a measurement) says logits for 440 rows; Spark Open says activations for 136 and model state for 103 of them |
| Out-of-memory rows | 21 | 16 | 76% (55% to 89%) | the same 21 as the first pass, which had 15 |
| Fitted rows | 860 | 529 | 62% (58% to 65%) | as for all memory rows |
| Divergence runs, the label | 641 | 572 | 89% (87% to 91%) | Spark Open calls 66 of 600 stable runs a blow-up. **Always answering "stable" scores 600 of the same 641, 94% (91% to 95%)**, so on stable runs Spark Open does worse than that baseline |
| Diverged or not | 640 | 574 | 90% (87% to 92%) | the 66 false blow-ups |
| The runs whose loss stayed finite | 606 | 539 | 89% (86% to 91%) | the harder test: a non-finite loss stops a run early, so those runs read off the evidence |
| How it failed, where both say it diverged | 41 | 39 | 95% (84% to 99%) | 2 non-finite runs called collapse |
| All compared | 1,522 | 1,117 | 73% (71% to 76%) | |

Batching didn't change Spark Open's answer much: on the 67 rows both passes answered, the batched label matches the single-row one for 63, 94% (86% to 98%). 571 rows have no rule label (569 CNN and MLP rows have no analytical model; 2 HF rows didn't record their shape), so they are labelled and not compared. 363 calls (111 failed), 741,716 input and 1,845,364 output tokens, $8.59 recorded, plus the 48-row pilot's $0.20: **$8.79 of the $10**. Every call is in `corpus/provenance/spark_open_labels_all_nov1.jsonl`; each row's label and reason in `corpus/reports/spark_labels_all_nov1_rows.jsonl.gz` and the agreement in `corpus/reports/spark_labels_all_nov1.json`. Read the memory agreement as Spark Open against the engine's analytical split, not against a measured one; the corpus measures only the peak. In short: memory agrees 62% of the time; divergence 89%, which is below the 94% that always answering "stable" would score; and where both say a run diverged, the kind of failure agrees on 39 of 41.

**Personal or sensitive data.** None.

## Collection

The corpus was collected between Oct 28 and Nov 2, 2026: the NVIDIA rows on Modal, in a pinned image (torch 2.8.0 with CUDA 12.8, plus pinned transformers, peft and bitsandbytes for the HF jobs; Nov 1's divergence runs and the top-up on Nov 1, the realistic divergence runs on Nov 2), and the MI300X rows on Nov 2 on one rented AMD Developer Cloud MI300X through `.github/workflows/amd-run.yml` (`python -m preflight.corpus.sweep run --kind grouped --gpu MI300X`, torch 2.5.1+rocm6.2). Each config ran in its own process with a deadline, so a hang became a `timeout` row and a kernel fault couldn't poison the next config. Sweeps were capped by a worst-case container-time bound and approved phase by phase. Every run's GPU, minutes and dollars are in `docs/status/b.md`.

## Known biases and limits

- **Small models.** Everything is sized for a single 24 GB card. Grouped models range from 0.9M to 670M parameters, with a median of 24M. The HF checkpoints range from 360M to 1.7B. Multi-GPU, FSDP, and models that need 80 GB aren't covered.
- **GPU mix.** A10 and L4 dominate, then L40S and H100, with T4 small and the A100 boards as a holdout. T4 has fp32 and fp16 groups only, because it has no bf16 path. **One AMD GPU, the MI300X** (Nov 2, a rented AMD Developer Cloud machine, ROCm 6.4.4 host): 295 configs of 59 groups, the same configs as the NVIDIA GPUs, on torch 2.5.1+rocm6.2 where the NVIDIA rows are torch 2.8.0, so a cross-vendor difference mixes the card with the software stack. They are a holdout: scored once for NVIDIA to MI300X step time and never calibrated on. No other AMD card, and no Intel or Apple GPU.
- **Modal picks the board.** A request for an A100-80GB landed on PCIe or SXM, and rows are filed under the board they ran on. A request for an A10 sometimes lands on an A10G (about 11% slower on one LoRA script); those rows are filed as `A10` (70 of them: 16 timing rows, 12 HF jobs and 42 realistic divergence runs), and `env.device_name` says "NVIDIA A10G".
- **Short windows.** Five timed steps after two warm-up steps. Long-run effects (thermal throttling, allocator fragmentation, data-loader stalls) aren't in it; the zoo's full runs show two of them matter (a loader that stalls only after its prefetch drains, an allocator pool that keeps growing for epochs).
- **Op tables count device time three times.** The worker kept profiler ranges, aten ops and the CUDA kernels under them, each with its own self device time, and together they sum to about 2.6× the step. Use the `aten::` rows alone, which cover about 0.9 of the step (`scripts/fit_steptime_gbdt.py`). The next corpus version keeps one view.
- **Version strings.** Every row's `config.versions.corpus` says `corpus@0.3.0`: that is the worker code's version, part of the key, and it didn't change; 0.4.0 is the published file's version.
- **Repeated configs.** The sampler for "100 grouped configs" drew 95 distinct ones on the A100. Counts above are distinct rows.
- **Synthetic inputs.** Random tokens give every row the same length, so padding and length variation aren't measured. Preflight's own probes measure them on the user's data.

## Uses

- **Intended:** calibrating and checking Preflight's intervals (memory, step time, P(OOM)), fitting and scoring learned cost models with held-out architectures and GPUs, and studying how far a roofline carries step time across GPUs and vendors.
- **What it can't be used for:**
  - ranking GPUs or vendors for purchase: the models are small, the windows are five steps, and the MI300X ran another torch version;
  - estimating how fast a particular production model trains, or anything multi-GPU (FSDP, tensor parallel), long-context or over 80 GB: none of it is in the corpus;
  - throughput or quality of real datasets: the timing kinds use random inputs, so padding, sequence-length spread and data loading aren't measured;
  - judging the divergence classifier on settings it wasn't drawn from: the Nov 1 runs over-represent learning rates far above usual, and the Nov 2 runs diverge only in fp16 without a scaler;
  - training any model on the Muse Spark Open labels as ground truth: they are a model's second opinion, measured against B's rules, not measured failure modes.

## Distribution and licence

- **Measurements:** Apache 2.0, the Preflight repository's licence (`LICENSE`); the dataset card says so (`license: apache-2.0`). Chiya uploads the file: a Hugging Face dataset or a GitHub release asset.
- **Sources:**
  - the configs: generated by `python/preflight/corpus` (sweep plans and the sampler's seeds are in the rows);
  - the HF jobs: Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, HuggingFaceTB/SmolLM2-360M, HuggingFaceTB/SmolLM2-1.7B and TinyLlama/TinyLlama-1.1B-Chat-v1.0, all Apache 2.0 from their authors. The corpus holds no weights from them, only the timings and memory of training them;
  - the divergence runs: trained on Tiny Shakespeare (`karpathy/char-rnn`, `data/tinyshakespeare/input.txt`; Shakespeare's text is in the public domain). The rows hold training curves, not text.
- **Checked before publishing:** no hostnames, IP addresses, container or account ids, user names or emails in any row (0.3.0 scanned Nov 1; 0.4.0 scanned Nov 2, the MI300X rows included: `env` holds only the device name, memory, SM count and library versions), and gitleaks finds nothing in the file.

## Maintenance

Workstream B keeps it. New rows come only through `python -m preflight.corpus.sweep` or `modal_app.py`, which write each config once under its key. A new code or library version changes the key, so old and new rows never mix silently.
