## Corpus sweep

The corpus trains Preflight's learned models: the LightGBM step-time residual, the GNN cost model, memory calibration checks and the divergence classifier. It is generated on purpose during the hackathon and published with a datasheet on Nov 3 (gap 16).

The NVIDIA slice is re-planned to $45 of Modal (PM, Oct 27): grouped probe-like configs and real Hugging Face jobs first, on T4, L4, A10 and L40S, with a few A100 rows only for the cross-GPU holdout and no H100 rows. Nothing launches without Chiya's yes (ask before spending, Oct 27).

| Phase | Kind | GPU | Configs | Estimate | Cap |
|---|---|---|---|---|---|
| 0. Smoke | grouped (2 groups) and hf | L4 | about 20 | 25 GPU-min, $0.35 | $0.70 |
| 1. Calibration core | grouped (60 groups) | L4 | 300 | 281 GPU-min, $3.75 | $5.50 |
| 1. Calibration core | hf (10 jobs × 5 points) | L4 | 50 | 125 GPU-min, $1.67 | $2.50 |
| 2. Other GPUs | grouped, fp32 and fp16 only | T4 | 75 | 70 GPU-min, $0.69 | $1.00 |
| 2. Other GPUs | grouped (60 groups) | A10 | 300 | 281 GPU-min, $5.16 | $7.50 |
| 2. Other GPUs | hf | A10 | 50 | 125 GPU-min, $2.29 | $3.50 |
| 2. Other GPUs | grouped (40 groups) | L40S | 200 | 188 GPU-min, $6.09 | $9.00 |
| 2. Cross-GPU holdout | grouped (20 groups) | A100-SXM-80GB | 100 | 94 GPU-min, $3.91 | $5.50 |
| | | | **1,095** | **$23.90** | **$35.20**, $9.80 reserve |

- **Estimates:** 45 s per grouped config and 120 s per HF job (guesses until the smoke run measures them), times the 1.25 billing overhead.
- **Caps:** each is the `--budgets` value for that GPU, and it is hard. Until three configs have finished, each config is priced at its deadline (600 s grouped, 900 s HF), so even a wave that hangs to its deadlines stays inside the cap. After that, the sweep prices the rest at the running p90 with one wave in reserve (S5).
- **Phase 0** measures real per-config seconds. Phases 1 and 2 are re-priced from those numbers before they are put to Chiya.
- **T4** has no bf16 path, so it runs only the fp32 and fp16 groups and no HF jobs (`modal_app.runs_on`).
- **Shared groups:** every GPU runs the same groups (the first n / 5 of seed 0). The MI300X runs the same 300, so every row has pairs across GPUs and vendors.

| Slice | Configs | Where | Budget |
|---|---|---|---|
| Grouped, AMD | 300 (`data/amd_grouped.jsonl`), the NVIDIA groups | MI300X droplet, Thursday | about 5 droplet-hours, $13 |
| Kernels and graphs, AMD (earlier plan) | 500 (`data/amd_kernels_slice.jsonl`) | MI300X droplet | $16, superseded by the grouped slice |
| Divergence short runs | 800 small-GPT runs | Modal L4 | not in the $45; only if Chiya adds money after G1 |

Every config has a content-addressed key and is written once. A restarted sweep skips what the store already holds and counts what it already spent against its budget, so it never runs past it (gap 13).

**Budget rule.** This is S5's B5 in `docs/decisions/engine-settings.md`, which never overspent in 3,000 simulated sweeps:
- Once three configs have finished on a GPU, each next config is priced at the running p90 of real configs, times a billing overhead of 1.25.
- One wave is held in reserve.

### What one record holds

- **kernels:** parameter counts, phase times (forward, backward, optimizer) from 5 unprofiled steps after 2 warm-up optimizer steps, peak allocated and reserved memory, mean power draw, a per-op table from a 3-step profiler window (time, FLOPs, input shapes), the FX module graph with shapes, the environment, and billed GPU-seconds. OOM is a label, not a failure.
- **divergence:** per-step loss, grad-norm, per-layer update-to-weight ratio, max attention logit, output-logit norm, loss scale and non-finite gradient counts. Each run is labelled diverged or not by rule D9, with the step it happened. Spikes that recover don't count; collapses to the trivial baseline do.
- **failures:** status and A's stable `error_code`, so corpus rows and probes share one vocabulary.

### Run the NVIDIA slices on Modal (only after Chiya's yes)

Cloud runs go through `remote-run.yml`; B holds no keys (docs/setup/cloud.md). Each phase is one workflow run of at most 120 minutes, and a sweep cut short resumes from the volume on its next run. Phase 0:

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-smoke-grouped -f ref=ws-b -f timeout=60 -f cmd="modal run python/preflight/corpus/modal_app.py::main --kind grouped --n 10 --budgets L4=0.3 --batch-size 2 --max-containers 1"
```

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-smoke-hf -f ref=ws-b -f timeout=90 -f cmd="modal run python/preflight/corpus/modal_app.py::main --kind hf --budgets L4=0.4 --batch-size 2 --max-containers 1"
```

Phases 1 and 2, with the caps from the table (each GPU runs in parallel inside one run):

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-l4 -f ref=ws-b -f timeout=120 -f cmd="modal run python/preflight/corpus/modal_app.py::main --kind grouped --n 300 --budgets L4=5.5"
```

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-hf -f ref=ws-b -f timeout=90 -f cmd="modal run python/preflight/corpus/modal_app.py::main --kind hf --budgets L4=2.5,A10=3.5"
```

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-gpus -f ref=ws-b -f timeout=120 -f cmd="modal run python/preflight/corpus/modal_app.py::main --kind grouped --n 300 --budgets T4=1,A10=7.5"
```

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-l40s -f ref=ws-b -f timeout=120 -f cmd="modal run python/preflight/corpus/modal_app.py::main --kind grouped --n 200 --budgets L40S=9"
```

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-a100 -f ref=ws-b -f timeout=60 -f cmd="modal run python/preflight/corpus/modal_app.py::main --kind grouped --n 100 --budgets A100-SXM-80GB=5.5"
```

Bring the rows back (a CPU container, about a cent), then read them from the `remote-runs` branch:

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-export -f ref=ws-b -f timeout=20 -f cmd="modal run python/preflight/corpus/modal_app.py::export --kind kernels"
```

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=b-corpus-export-hf -f ref=ws-b -f timeout=20 -f cmd="modal run python/preflight/corpus/modal_app.py::export --kind hf"
```

Each run's GPU, minutes and dollars go in `docs/status/b.md`.

### Calibration data: grouped configs and real HF jobs (Review, Oct 27)

The engine's conformal intervals need data shaped like a user's job, so two more kinds run on an L4:
- `grouped`: one architecture at five batch × sequence points. Three are probes laid out like A's probe plan; two are held-out queries, one inside the queried range and one a halving below the smallest probe.
- `hf`: about ten real Hugging Face fine-tunes, five ungated models each as bf16 LoRA and 4-bit QLoRA, at the same five points.

Every key carries the corpus code version and the library versions it ran with, so a torch or transformers upgrade never mixes with older rows.

They run as phases 0 to 2 above. Once the exports are on the `remote-runs` branch, copy their `out/corpus` folders into one local folder (for example `./corpus`) and build the table:

```
PYTHONPATH=python python -m preflight.corpus.calibrate --store ./corpus --out python/preflight/engine/_calibration_table.py
```

The last step prints leave-one-architecture-out coverage with Wilson intervals and writes the table the engine reads. Model cards switch from `calibration: simulated` to `held_out` for the targets the table covers.

### Run the AMD slice on the MI300X (Thursday's droplet session)

Through `amd-run.yml`, once Chiya has created the droplet and said yes. Fill in the droplet's IP and its creation time (ISO 8601 with a time zone):

```
gh workflow run amd-run.yml -R uchuuronin/preflight -f name=b-amd-grouped -f host=<ip> -f ref=ws-b -f timeout=230 -f cmd="python -m preflight.corpus.sweep run --kind grouped --configs python/preflight/corpus/data/amd_grouped.jsonl --gpu MI300X --out /root/corpus --copy-to out/corpus --budget-usd 13 --billing uptime --droplet-started <created at>" -f destroy_droplet_id=
```

- **Billing:** the droplet bills for uptime, so the budget counts the droplet's whole bill since it was created, A's runs on it included. Give the sweep what is left of the session's money, not a fixed $13.
- **Interruptions:** the store lives in `/root/corpus`, which `amd-run` doesn't wipe, so a second run resumes. `--copy-to out/corpus` brings the rows and `ledger-MI300X.jsonl` back even when the run times out.
- **Mislabelling guard:** the sweep refuses to start if `--gpu` doesn't match the GPU it finds.
- **Teardown:** leave `destroy_droplet_id` empty while more runs follow, and destroy the droplet when the session ends.
