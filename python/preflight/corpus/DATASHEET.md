# Preflight corpus: datasheet

Following *Datasheets for Datasets* (Gebru et al.). Version: corpus@0.3.0, exported Oct 31, 2026 (`remote-runs` folders `b-corpus-export-3` and `b-corpus-export-hf-2`), with the divergence runs of Nov 1 (`b-divergence-export-2`). Owner: workstream B.

**Published file** (gap 16): `preflight-corpus-0.3.0.jsonl.gz`, 2,339 rows (1,519 kernels, 100 HF jobs, 720 divergence runs), 18,462,637 bytes, sha256 `b7cf58d13c291759b8ef23004d87ae0f6b6e42290a7a7e71c998f805a04feb20`, built by `python scripts/export_corpus.py` (the same rows give the same bytes). It ships with `SHA256SUMS` and a dataset card (`README.md`). **Published on Hugging Face:** [huggingface.co/datasets/chaoschia/preflight-corpus](https://huggingface.co/datasets/chaoschia/preflight-corpus) (all three files, commit `0aac218`). It is private until the repository goes public on Nov 3 to 4.

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

A divergence row has `config`, `key`, `gpu` and `status` too, and instead of timings: `series` (per step: loss, grad norm, the largest attention logit, the output logit norm, the update-to-weight ratio per layer, non-finite gradients, the loss scale), `label` (the D9 hindsight label: diverged, the onset step, the reason), `steps_run`, `stopped_early` and `data`.

The published file adds three fields to every row: `kind` (`kernels`, `hf` or `divergence`), `split` (below) and `source_file`.

**How many** (measured):

| GPU | Grouped configs | OOM | HF jobs | OOM |
| --- | --- | --- | --- | --- |
| L4 | 577 | 11 | 50 | 1 |
| A10 | 577 | 11 | 50 | 1 |
| L40S | 195 | 1 | 0 | 0 |
| T4 | 75 | 4 | 0 | 0 |
| A100-PCIe-80GB | 42 | 0 | 0 | 0 |
| A100-SXM-80GB | 53 | 0 | 0 | 0 |

Plus 720 divergence runs: 377 on L4 and 343 on A10 (below).

**Families.** The grouped configs come in groups of five points: three probe-like points and two queries per architecture. 111 distinct groups ran in all. The architectures are small models built in the corpus code:
- `gpt`: decoder-only transformers;
- `encoder`: BERT-style;
- `lora`: synthetic LoRA transformers;
- `cnn`: ResNet-style;
- `mlp`.

The HF jobs are 10 groups of five on real Hugging Face checkpoints, LoRA in bf16 and QLoRA in nf4. The checkpoints are Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, HuggingFaceTB/SmolLM2-360M, HuggingFaceTB/SmolLM2-1.7B and TinyLlama/TinyLlama-1.1B-Chat-v1.0.

**Splits** (the `split` field).
- **Calibration:** every non-A100 row, used leave-one-architecture-out for coverage.
- **Cross-GPU holdout:** every A100 row, never calibrated on and scored once (`docs/decisions/calibration-l4-phase1.md`).
- **Not in calibration:** the synthetic `lora` family (lens `finetune_synthetic`). Its scores are reported, but users send real models.
- **Divergence:** the 720 divergence runs, which score the divergence classifier only.

**Data inside the jobs.** The `kernels` and `hf` kinds train on random token ids or random images, so they hold measurements of compute, not of any text or image dataset. The `divergence` kind trains a small GPT (4 layers, d256, 600 steps) on Tiny Shakespeare (public domain) and keeps each run's loss, grad-norm, logit and update series with its D9 label.

**Divergence runs** (measured, Nov 1): 720 runs, 377 on L4 and 343 on A10 (two draws of the same sampler, the board on every row), 44 diverged: 37 of them in fp16, 15 in the first 30 steps. They score the divergence classifier, which is not used: held-out AUROC 0.66 (95% 0.56 to 0.76) against a 0.85 target (`corpus/reports/divergence_nov1.json`).

**Failure labels by Muse Spark Open** (Nov 1; contract item, the second logged Spark Open job). Every row that ran out of memory (29) and every diverged run (44) was sent, with its own numbers, to Muse Spark Open on Baseten (`meta-models/muse-spark-open-1026`, effort low, structured output; `scripts/spark_label_corpus.py`) to name its failure mode, and the answers were compared with B's rule labels, which were never in the prompt: the engine's largest memory term at the row's settings for an out-of-memory row, the D9 reason for a diverged run.

| Task | Compared | Agree | Rate (Wilson 95%) | Where they differ |
| --- | --- | --- | --- | --- |
| Out-of-memory rows | 21 | 15 | 71% (50% to 86%) | the rule says logits for 9 rows, Spark Open activations for 3 of them; the rule says model state for 2, Spark Open activations; the rule says activations for 10, Spark Open "other" for 1. This is the real test |
| Diverged runs | 44 | 44 | 100% (92% to 100%) | nowhere; 37 non-finite and 7 blow-ups, which the numbers show plainly (a non-finite step is in the evidence) |
| All | 65 | 59 | 91% (81% to 96%) | the diverged runs carry most of this |

Two passes (Chiya, Nov 1). The first answered 64 of 73 rows: of its 16 failed calls, 8 ran into the 8,192-token output cap (an answer that never closed its JSON), 6 timed out at 20 s, and 2 re-asks were cut off when the row's 30 s ran out. The second asked only those 9 rows again with a 16,384-token cap and 90 s, and answered 8; one row ran past the cap on both tries and stays unanswered. 7 out-of-memory rows have no rule label (5 CNN rows have no analytical model, and 2 HF rows didn't record their model shape when they failed), so 65 of the 73 rows are compared. 90 calls in all, 42,454 input and 153,953 output tokens, $0.70 ($0.538 for the first pass and $0.167 for the second, under its $0.50). Every call, failed or not, is in `corpus/provenance/spark_open_labels_nov1.jsonl` with its tokens and cost; each row's label and reason are in `corpus/reports/spark_labels_nov1_rows.jsonl` and the agreement in `corpus/reports/spark_labels_nov1.json`. A timed-out call is logged at $0, though Baseten may have billed it.

**Whole-corpus labels by Muse Spark Open** (Nov 1, evening; Chiya: Baseten credit, up to $10). **These labels are written by a model, not measured, and nothing trains on them**: they sit beside the rows as a second opinion, and the agreement below is how far to trust them. Every one of the 2,339 published rows was sent in batches of 8 rows of one kind (`scripts/spark_label_all.py`; same model and effort, rows as `r1` to `r8`, never the corpus key) with one question per kind: for a memory row, fitted or out of memory, which memory use is the largest part of its peak; for a divergence run, whether it stayed stable and, if not, how it failed. Neither the rule label nor the hindsight label was ever in the prompt. 1,997 rows were answered; 342 were not: 328 in 41 batches whose answer ran to the 12,000-token output cap twice, padding its JSON with whitespace, and 14 left out of 2 batches that otherwise answered. The 111 failed calls cost $5.86 of the $8.59. **Most of that spend was avoidable** (Chiya's review, Nov 2): the pilot just before this run (`b-spark-all-pilot`, B's status file) had a batch fail twice at the same 12,000-token cap, and the first pass had shown answers running to the cap without closing their JSON (8 of its 16 failed calls), yet the run went out with nothing against it. All 111 failed calls were cap hits. A stop after a batch's first cap hit would have saved $2.47 (70 retries, 41 of which failed again, and the 29 batches answered on retry would have gone unanswered); an answer schema that leaves no room for padding would have kept most of the $5.86.

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

The corpus was collected between Oct 28 and Nov 1, 2026 (the divergence runs on Nov 1), on Modal, in a pinned image (torch 2.8.0, plus pinned transformers, peft and bitsandbytes for the HF jobs). Each config ran in its own process with a deadline, so a hang became a `timeout` row and a kernel fault couldn't poison the next config. Sweeps were capped by a worst-case container-time bound and approved phase by phase. Every run's GPU, minutes and dollars are in `docs/status/b.md`.

## Known biases and limits

- **Small models.** Everything is sized for a single 24 GB card. Grouped models range from 0.9M to 667M parameters, with a median of 24M. The HF checkpoints range from 360M to 1.7B. Multi-GPU, FSDP, and models that need 80 GB aren't covered.
- **GPU mix.** L4 and A10 dominate, L40S has a third of the configs and T4 a tenth. T4 has fp32 and fp16 groups only, because it has no bf16 path. **One AMD GPU, the MI300X** (Nov 2): 295 grouped configs, the same configs as the NVIDIA GPUs, on torch 2.5.1+rocm6.2 (the NVIDIA rows are torch 2.8.0, so their keys differ; pair them on the config). They are a holdout: scored once for NVIDIA to MI300X step time (K1 within 20% on 22% of 1,147 pairs) and never calibrated on.
- **Modal picks the A100 board.** A request for an A100-80GB landed on PCIe or SXM. Rows are filed under the board they ran on.
- **Short windows.** Five timed steps after two warm-up steps. Long-run effects (thermal throttling, allocator fragmentation, data-loader stalls) aren't in it.
- **Op tables count device time three times.** The worker kept profiler ranges, aten ops and the CUDA kernels under them, each with its own self device time, and together they sum to about 2.6× the step. Use the `aten::` rows alone, which cover about 0.9 of the step (`scripts/fit_steptime_gbdt.py`). The next corpus version keeps one view.
- **Repeated configs.** The sampler for "100 grouped configs" drew 95 distinct ones on the A100. Counts above are distinct rows.
- **Synthetic inputs.** Random tokens give every row the same length, so padding and length variation aren't measured. Preflight's own probes measure them on the user's data.

## Uses

- **Intended:** calibrating and checking Preflight's intervals (memory, step time, P(OOM)), and fitting and scoring learned cost models with held-out architectures and GPUs.
- **Not intended:** benchmarking GPUs against each other for purchasing, or estimating how fast any specific production model trains. The models are small and the windows short.

## Distribution and licence

- **Measurements:** Apache 2.0, the Preflight repository's licence (`LICENSE`); the dataset card says so (`license: apache-2.0`). Chiya uploads the file: a Hugging Face dataset or a GitHub release asset.
- **Sources:**
  - the configs: generated by `python/preflight/corpus` (sweep plans and the sampler's seeds are in the rows);
  - the HF jobs: Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, HuggingFaceTB/SmolLM2-360M, HuggingFaceTB/SmolLM2-1.7B and TinyLlama/TinyLlama-1.1B-Chat-v1.0, all Apache 2.0 from their authors. The corpus holds no weights from them, only the timings and memory of training them;
  - the divergence runs: trained on Tiny Shakespeare (`karpathy/char-rnn`, `data/tinyshakespeare/input.txt`; Shakespeare's text is in the public domain). The rows hold training curves, not text.
- **Checked before publishing:** no hostnames, container or account ids, user names or emails in any row (scanned Nov 1), and gitleaks finds nothing in the file.

## Maintenance

Workstream B keeps it. New rows come only through `python -m preflight.corpus.sweep` or `modal_app.py`, which write each config once under its key. A new code or library version changes the key, so old and new rows never mix silently.
