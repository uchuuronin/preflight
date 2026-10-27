## Corpus sweep

The corpus trains Preflight's learned models: the LightGBM step-time residual, the GNN cost model, memory calibration checks and the divergence classifier. It is generated on purpose during the hackathon and published with a datasheet on Nov 3 (gap 16).

| Slice | Configs | Where | Budget |
|---|---|---|---|
| Kernels and graphs, NVIDIA | 1,500 × L4, A100-SXM-80GB, H100-SXM-80GB | Modal | $140 |
| Kernels and graphs, AMD | 500 (`data/amd_kernels_slice.jsonl`) | MI300X droplet, run by A | $16 |
| Divergence short runs | 800 small-GPT runs | Modal L4 | $35 |

Every config has a content-addressed key and is written once. A restarted sweep skips what the store already holds and counts what it already spent against its budget, so it never runs past it (gap 13).

**Budget rule.** This is S5's B5 in `docs/decisions/engine-settings.md`, which never overspent in 3,000 simulated sweeps:
- Once three configs have finished on a GPU, each next config is priced at the running p90 of real configs, times a billing overhead of 1.25.
- One wave is held in reserve.

### What one record holds

- **kernels:** parameter counts, phase times (forward, backward, optimizer) from 5 unprofiled steps after 2 warm-up optimizer steps, peak allocated and reserved memory, mean power draw, a per-op table from a 3-step profiler window (time, FLOPs, input shapes), the FX module graph with shapes, the environment, and billed GPU-seconds. OOM is a label, not a failure.
- **divergence:** per-step loss, grad-norm, per-layer update-to-weight ratio, max attention logit, output-logit norm, loss scale and non-finite gradient counts. Each run is labelled diverged or not by rule D9, with the step it happened. Spikes that recover don't count; collapses to the trivial baseline do.
- **failures:** status and A's stable `error_code`, so corpus rows and probes share one vocabulary.

### Run the NVIDIA slices on Modal

From the repo root, once Modal access lands:

```
uv run --with modal --with torch modal token new
```

```
PYTHONPATH=python modal run python/preflight/corpus/modal_app.py --kind kernels --dry-run
```

```
PYTHONPATH=python modal run --detach python/preflight/corpus/modal_app.py --kind kernels --budgets "L4=25,A100-SXM-80GB=45,H100-SXM-80GB=70"
```

```
PYTHONPATH=python modal run --detach python/preflight/corpus/modal_app.py --kind divergence --n 800 --budgets "L4=35"
```

```
modal volume get preflight-corpus /kernels ./corpus/kernels
```

### Calibration data: grouped configs and real HF jobs (Review, Oct 27)

The engine's conformal intervals need data shaped like a user's job, so two more kinds run on an L4:
- `grouped`: one architecture at five batch × sequence points. Three are probes laid out like A's probe plan; two are held-out queries, one inside the queried range and one a halving below the smallest probe.
- `hf`: about ten real Hugging Face fine-tunes, five ungated models each as bf16 LoRA and 4-bit QLoRA, at the same five points.

Every key carries the corpus code version and the library versions it ran with, so a torch or transformers upgrade never mixes with older rows.

```
PYTHONPATH=python modal run --detach python/preflight/corpus/modal_app.py --kind grouped --n 250 --budgets "L4=10"
```

```
PYTHONPATH=python modal run --detach python/preflight/corpus/modal_app.py --kind hf --budgets "L4=5"
```

```
modal volume get preflight-corpus / ./corpus
```

```
PYTHONPATH=python python -m preflight.corpus.calibrate --store ./corpus --out python/preflight/engine/_calibration_table.py
```

The last step prints leave-one-architecture-out coverage with Wilson intervals and writes the table the engine reads. Model cards switch from `calibration: simulated` to `held_out` for the targets the table covers.

### Run the AMD slice on the MI300X (for A)

On the droplet, with the repo cloned and ROCm PyTorch installed:

```
cd preflight
```

```
PYTHONPATH=python python -m preflight.corpus.sweep run --kind kernels --configs python/preflight/corpus/data/amd_kernels_slice.jsonl --gpu MI300X --out ~/corpus --budget-usd 16
```

Stop and start it freely; it resumes. It refuses to start if `--gpu` doesn't match the GPU it finds, so rows can't be mislabelled. Then copy `~/corpus/kernels/MI300X` back for B.

The droplet bills for uptime, not per config, so add `--billing uptime --droplet-started <when it was created, ISO 8601 with a time zone>` to the command above. The budget then counts the droplet's whole bill since it started, idle time and restarts included, and every start, config and stop goes into `~/corpus/ledger-MI300X.jsonl` with the spend at that moment. Destroy the droplet when the sweep stops: the bill keeps running until you do.
