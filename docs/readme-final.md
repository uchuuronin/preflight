<!--
Final README by D, Sun Nov 1, for the PM to swap in for README.md. It combines C's README on
`main` (the Validation intro, Develop, "Verify a fix in your own CI", Layout) with D's draft.
The status columns are as of `main` `a52753f` (Nov 1): update them at the swap from the status
files if anything has turned on since. Keep both `validation` markers: B's scorer rewrites what
is between them. Placeholders to fill before Nov 4 are marked TODO(PM). Every number carries
its source.
-->

# Preflight

<!-- TODO(PM): once the site is live and workers/ is in the deploy, uncomment the badge and
replace TODO-PM-HOST with the live host. It reads A's Oct 28 record (a-smoke-l4-main-4) and shows
its first headline number with its status word; until then it would be a broken image.
[![Preflight flight record: the LoRA fine-tune, measured on an L4](https://TODO-PM-HOST/badge/ICDYrfHQ2nBNbv3mlRHHMg)](https://TODO-PM-HOST/s/ICDYrfHQ2nBNbv3mlRHHMg)
-->

**A flight simulator for ML jobs.** Preflight measures a few minutes of your real training,
evaluation, serving or agent job, builds a model of how it behaves, and lets you fly the whole
run, or a whole campaign of runs, before you spend real GPU hours or API dollars.

Everything in this repository was built during the Meta Global AI Developer Hackathon
(Oct 26 to Nov 4, 2026). The first commit is `11a0ce9`, Oct 26, 22:25 ET; nothing was carried
over from earlier work.

## Status

As of `main` `a52753f` (Sun Nov 1). Every feature ships behind a flag that is off until its check
passes (`flags.json`); `flags.production.json` turns on what the public site runs. TODO(PM): the
public deploy isn't up yet, so "on in production" means "on once deployed"; update this line
with the live link.

**On in production**

| What | Where it shows | Source |
| --- | --- | --- |
| Muse Spark reads each script (the script map) | Boarding's "Reading your script", and the cockpit's job map | `harness.spark_script_map`, on in `flags.production.json` (Chiya, Nov 1) |
| Predictions with intervals and abstention: memory, step time, cost, risk | The cockpit's four readouts, the knobs and the flight plan | `engine.memory`, on in production (Chiya, Oct 30); it gates the whole prediction step (`runner/pipeline.py`) |
| Campaign calendar | The campaign sample (illustrative numbers, B's planner on a 12-run grid) | `c.campaign_calendar`, on in production (Chiya, Nov 1) |
| GRPO lens on a measured run | The GRPO sample: reward spread, completion lengths, the tour | A's L4 run `a-gt-grpo-l4-3`; no flag |
| Cost readout at your own rate | Every dollar readout: quantity x rate = total, with the price table's source and date | No flag |
| Verify a fix in your own CI | A GitHub Action (below) | No flag; usable from other repositories once this one is public |
| The `preflight` CLI (`map`, `run`) | Your own machine | On `main`; the `pip install git+...` line works once the repository is public |

**Behind a flag, off for now**

| What | Flag | Where it stands |
| --- | --- | --- |
| Checking in your own repo, script or notebook | `runner.queue` | Built on `main`; until it is on, the site opens samples and shared records only |
| The guide (Muse Spark moves the knobs with tool calls) | `c.guide.live` | Smoke runs pass: GitHub run 41273350602, 5 rounds, all called tools [measured], US$0.019 [derived: reported tokens at list prices] |
| The fix loop (Muse Code races candidate fixes) | `c.fix_loop` | Route `/v1/patches` on `main`; in a clean smoke run (41250977670) Muse Code produced the correct patch in 19 s [measured], US$0.09 [derived: reported tokens at list price]. Gating a fix on Preflight's re-probe isn't wired end to end yet (it needs A's runner to apply the patch) |
| Log triage by Muse Spark Open on Baseten | `harness.spark_open_triage` | Wired into the pipeline on `main`; not yet run for real |
| HOPs view, fit map, exports, the CLI re-check item | `c.hops`, `c.fit_map`, `c.exports`, `d.exports_cli` | Built on `main` |
| The add-ons (detail level, currency, free GPUs, notebook) | `d.*` (below) | Built on `main`; checked in Chromium (`docs/audit/scripts/addons.mjs`) |

**Not used, or not measured**

| What | Why | Source |
| --- | --- | --- |
| Divergence classifier | Misses on held-out runs: AUROC 0.58 [cited: Chiya, Nov 1] against a 0.85 target, so `risk.divergence` stays off | B's held-out run, as given by Chiya on Nov 1; TODO(PM): link B's report, which isn't in the repository yet |
| AMD | No AMD measurements: the AMD Developer Cloud account is on a verification hold, so every AMD figure says "not measured" | `docs/status/b.md`, `docs/status/a.md` |
| GBDT correction for an unmeasured GPU | Fails held out (51% within 20%, against 60% for the analytical model [measured, corpus]), so it isn't used | `docs/status/b.md` |
| Step time on an unmeasured GPU | Used, with its range, but 60% of held-out queries land within 20% [measured, corpus], under the 90% target | `docs/status/b.md` (leave-one-GPU-out on the corpus) |
| RL rollout and serving models | Built, switched off in the engine until their validation rows pass | `python/preflight/engine/api.py` |
| Claim check by Muse Spark Open | Module written, not wired in | `python/preflight/harness/claim_check.py` |
| Muse Glimmer | Not hosted: stays deferred until its download terms and licence are confirmed in writing | `docs/decisions/glimmer-host.md` |
| Full-run validation | Scored on Mon Nov 2 (the table below) | B's scorer |

## Try it

**In the browser, no install:** open TODO(PM): public link. Pick a sample to open it straight in
the cockpit (samples use illustrative numbers unless they say measured, and cost nothing), or open
a shared record's link. Checking in your own repo or script opens with `runner.queue`.

**Share a flight** (once `workers/` is in the deploy; TODO(PM)): post
`https://TODO-PM-HOST/s/<share token>`. Link previews show the record's headline numbers, each with
its status word (Measured, Predicted, Extrapolated, Not trusted or Illustrative), as text, with a
card image for the records whose card is pre-rendered, and the link opens the flight in the
cockpit.
For a README badge:
`[![Preflight](https://TODO-PM-HOST/badge/<share token>)](https://TODO-PM-HOST/s/<share token>)`
(`workers/README.md`).

**On your own machine, one command** (Python 3.11 or 3.12, in the environment that has your
torch):

```bash
pip install git+https://github.com/uchuuronin/preflight
```

**Quickstart:**

```bash
# 1. Read the script without running it (seconds): lens, framework, knobs and where the code sets them.
preflight map train.py

# 2. Measure a few steps on this machine (its GPU if it has one) and write a flight record.
#    On a real script this takes a few minutes: probes run real training steps.
preflight run . --entrypoint train.py --out .preflight/records

# Arguments for your script go after --, e.g.
preflight run . --entrypoint train.py -- --batch_size 16
```

The record lands in `.preflight/records/<share_token>.json`, a JSON flight record. Opening a local
record in the web app isn't built yet: the app opens records by their share link.
(Checked Oct 31 by D: the package installed from this checkout in a fresh Python 3.11
environment with torch 2.5.1 ran three probes of a small torch loop on the CPU and wrote a record;
the GitHub install line itself can be tried once the repository is public.)

## How it works

GPU time goes to a few minutes of probes, a background corpus sweep and verify runs; everything a
visitor explores afterwards is computed in the browser from the flight record.

```mermaid
flowchart TB
	IN["Your job: a sample, a GitHub repo link,<br/>an uploaded script or a notebook"]
	CLI["preflight CLI on your own GPU<br/>(NVIDIA, or AMD ROCm)<br/>probes run on your machine"]
	CK["Check-in desk<br/>(browser, Cloudflare Pages)"]
	API["API and job store<br/>(Modal)"]
	MAP["Script map<br/>(Muse Spark)"]
	PRB["Probes in GPU sandboxes (Modal, NVIDIA)<br/>and an agent proxy with OTel GenAI spans"]
	CORPUS["Corpus sweep<br/>(Modal)"]
	ENG["Prediction engine: analytical and learned models,<br/>conformal intervals, abstention"]
	REC[("Flight record")]
	COCK["Cockpit in the browser: job map, four readouts with intervals,<br/>knobs, flight plan; prediction grids interpolated locally"]
	EXP["Exports: plan, edits,<br/>Slurm file, command, link"]
	GUIDE["Guide: tool calls on the cockpit<br/>(Muse Spark, through the API)"]
	FIX["Fix loop: candidate fixes raced in parallel<br/>(Muse Code, in a Modal sandbox)"]
	TRI["Log triage<br/>(Muse Spark Open on Baseten)"]

	IN --> CK
	CK -.->|"check-in: behind a flag"| API
	API --> MAP --> PRB --> ENG --> REC --> COCK
	COCK -.->|"behind a flag"| EXP
	CLI -->|"writes"| REC
	CORPUS --> ENG
	COCK <-.->|"behind a flag"| GUIDE
	COCK -.->|"behind a flag"| FIX -.->|"verify by re-probe, not wired yet"| PRB
	PRB -.->|"behind a flag"| TRI -.-> REC
```

The shared interfaces (flight record, prediction API, job events, UI extension points) and who
owns which path are in [CONTRACT.md](CONTRACT.md).

## Meta models and partner tools

| Tool | Where it is used | Status on `main` `a52753f` |
| --- | --- | --- |
| **Muse Spark** (`muse-spark-1.3`, Meta Model API) | Maps each script with structured output (`python/preflight/harness/`); runs the guide through the Responses API with tool calls on the live cockpit (`api/preflight_api/guide/`); the model behind the agent bug-zoo script (`zoo/agent_eval_spark/`) | Script map on in production (Oct 28 runs, before the flag: US$0.0178 and US$0.0200 a job [derived: reported tokens at list prices]); guide behind `c.guide.live` |
| **Muse Code** (headless) | Fix loop: candidate fixes raced in parallel worktrees inside a sandbox, edits only; the API key reaches Muse Code on stdin only, never in its environment or files; each fix is gated on Preflight's own re-probe and verify, not on Muse Code's exit code (`api/preflight_api/fix/`) | Behind `c.fix_loop`; in clean smoke run 41250977670 Muse Code produced the correct patch in 19 s [measured], US$0.09 [derived: reported tokens at list price]; gating on the re-probe isn't wired end to end yet |
| **Muse Spark Open** (Baseten) | Triage: reads each probe's log tail and names the cause, quoting the line it read; code checks the quote is really in the log (`python/preflight/harness/triage.py`) | Behind `harness.spark_open_triage`, off; not yet run for real. The claim check (`claim_check.py`) is written but not wired in |
| **Muse Glimmer** | One of three models in the agent model sweep's price table (`python/preflight/engine/prices/`) | Not hosted (`docs/decisions/glimmer-host.md`) |
| **Modal** | Probes in GPU sandboxes, the job store and API, the background corpus sweep, the fix loop's sandboxes | Every measured run so far (Oct 28 to Nov 1, `docs/status/a.md`); the runner is deployed for each run and stopped after; public deploy TODO(PM) |
| **AMD Developer Cloud** | ROCm probes through the CLI on an MI300X, and the AMD half of the corpus (`python/preflight/corpus/`, `.github/workflows/amd-run.yml`) | No AMD measurements: the account is on a verification hold |
| **Cloudflare Pages** | The web app, the `/v1` edge function that forwards to the API with a shared secret and the visitor's address for quotas (`web/functions/`), and share cards (`workers/`) | Built; TODO(PM): live link |

## What a run costs to measure

One run so far (A, Oct 28, job `a-smoke-l4-main-4`, the LoRA fine-tune zoo script): upload to
flight record in 6 min 45 s [measured], with 386 s of probe time on one L4 [measured]; that is
over the plan's 5-minute target. GPU $0.0858 [derived: measured probe time at the L4 list rate]
and Muse Spark $0.0178 [derived: reported tokens at list prices]. The sandbox's CPU and memory
are not in that total yet; Modal's bill replaces the estimate.
TODO(PM): replace with the validation table's "Cost of one Preflight run" row once scored.

## Validation

Predictions are checked against real full runs, hit or miss, starting with the runs scored on
Mon Nov 2. The table between the markers below scores each validation target and says so when a
row has nothing to score yet. It is written by
`scripts/score_validation.py <manifest.json> --readme README.md` from one manifest of full runs;
edit the manifest, not the table. The knob response row's timings
come from a stranger test run at 4x CPU slowdown (`KNOB_MS_OUT=out/knob_ms.json node
web/e2e/stranger.mjs`), copied into the manifest's `knob_ms` list.

<!-- validation:start -->
| Metric | Target | Measured on | Result | 95% interval | Hit | Notes |
|---|---|---|---|---|---|---|
| Peak memory error | Within 10% | Bug zoo and unseen scripts, full runs | not measured |  |  | no full run scored yet |
| OOM prediction | No missed OOMs; at most 1 false alarm per 10 scripts | Bug zoo and unseen scripts | not measured |  |  |  |
| Step time, same GPU | Within 15% | Full runs | not measured |  |  | no full run scored yet |
| Step time, unmeasured GPU | Within 20% | Train on 3 GPU types, test on the 4th | not measured |  |  | no corpus store given |
| GNN + GBDT ensemble vs GBDT alone | Lower error on held-out architectures | Corpus holdout |  |  | miss | paired gain +4.8% (95% -1.4% to +11.6%); the interval includes zero, so the GBDT is used |
| Divergence warning | AUROC 0.85+, 100+ steps early | Held-out short runs | not measured |  |  | the classifier's held-out report isn't in yet |
| Rollout time (RL) | Within 20% | An unseen TRL script (the GRPO zoo script calibrated it, Nov 1) | not measured |  |  | no full run scored yet |
| Serving tokens/sec | Within 15% | vLLM full runs | not measured |  |  | no full run scored yet |
| Agent run total cost | Inside the 90% interval | Full benchmark run vs the 30-task sample | not measured |  |  | no full run scored yet |
| Interval coverage | 90% intervals cover at least 88% | All of the above | not measured |  |  | no full run scored yet |
| Abstention | Every query outside the probed range is flagged | Automated test | not measured |  |  | tests/test_engine_api.py, run with the suite |
| Planted problems caught | All of them | Bug zoo | not measured |  |  |  |
| Unseen scripts harnessed | 8 of 9 without hand edits | One per job type | not measured |  |  |  |
| Knob response | Under 400 ms at p95 | Browser timing | not measured |  |  | no browser timings given |
| Cost of one Preflight run | Under $1 and under 5 minutes | Modal billing and timer | not measured |  |  | no full run scored yet |
A "within" target is a hit when at least 90% of runs are within it; intervals are Wilson 95% on the proportion, and a hit is "confident" only when its lower end is at least 90% ("hit, few runs" otherwise). Unseen scripts are never tuned on. Not scored yet: no full runs (seeded from the scorer with an empty manifest, Nov 1).
<!-- validation:end -->

## Add-ons

Smaller features, each behind its own flag in `flags.json`, all off on `main` `a52753f`.
`docs/audit/scripts/addons.mjs` checks that each still renders where it belongs, with no errors,
no accessibility violations inside it and nothing under 24 px, and that it is gone with its flag
off: 20 of 20 states pass on `main` `c4b95d4` [measured, Chromium, Nov 1]. TODO(PM): mark the rows
whose flags are on in production at the swap.

| Add-on | What a visitor gets | Flag |
| --- | --- | --- |
| Detail level (D1) | "New to this" shows the few knobs that matter most for the goal, in plain words, with a glossary card in each knob's control; knobs that a warning names stay in view | `d.expertise` |
| Show costs in (D3) | Dollar figures in one of 30 currencies at the European Central Bank's dated reference rates, picked from the browser's region or set in the Logbook; regions without a rate stay in US dollars | `d.currency` |
| Free GPUs (D4) | Whether the plan finishes inside one free session on Google Colab, Kaggle, Modal's starter credit or Lightning AI, and how often to save a checkpoint if not | `d.freetier` |
| Notebook (D5) | A check-in tab for a Jupyter notebook: it previews the script and requirements that would run; checking it in opens with live runs | `d.notebook` |
| Take it with you | The plan, an edit script for the changed settings, a Slurm batch file and the run command, built from the settings on screen, and the record's link once it has one | `c.exports`; the "Check it again on your machine" item also needs `d.exports_cli` |

## Develop

```bash
uv sync                                   # Python 3.11 env with dev tools
npm ci                                    # schema codegen (json-schema-to-typescript)
uv run pytest -q                          # tests
uv run ruff check . && uv run ruff format --check .
uv run python -m preflight.schema.generate   # after any schema change
```

Probes need PyTorch: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
for CPU, or the CUDA or ROCm wheel on a GPU box.

## Verify a fix in your own CI

The verify GitHub Action fails a job unless a fix improved what it was for and left the
training run's meaning alone: the same effective batch, tokens seen and learning-rate
schedule, throughput no worse than 90% of before, and the same fixed-seed loss within a small tolerance. It runs
Preflight's own verify code (`api/preflight_api/verify.py`) on two flight records, with
Python's standard library only and no keys.

```yaml
# .github/workflows/preflight-verify.yml in your repository
on: pull_request
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: uchuuronin/preflight/.github/actions/verify@main  # pin a commit hash for production
        with:
          before: preflight/before.record.json  # the run before the fix
          after: preflight/after.record.json    # the verify run after it
          metric: memory.peak                   # what the fix was for
          # may-change: throughput_floor        # checks this fix is allowed to move
          # keep-eval: "true"                   # the fix touches evaluation; it must still run
          # no-grad: "true"                     # the fix runs evaluation without gradients
```

The `status` output is `passed`, `failed` or `error` (a check couldn't be measured, which never
counts as a pass), and the job summary lists every check with its before and after values. The
records are flight record JSON files: the API returns any record by its share link at
`GET /v1/records/{share_token}`. Another repository can use the action only while this one is public, or, while
it is private, from repositories of the same owner with Actions access to it allowed.

## Layout

See [CONTRACT.md](CONTRACT.md) for the shared interfaces and who owns which path.

## Licences

- **Preflight:** Apache License 2.0 ([LICENSE](LICENSE)).
- **Dependencies:** every bundled package and its licence is in
  [docs/licences.md](docs/licences.md); CI fails on GPL, AGPL and SSPL.
- **torchtitan test files** (`zoo/torchtitan_debug/`): torchtitan's own debug config, test
  tokenizer and `c4_test` slice, from its v0.1.0 tag, under BSD-3-Clause (`LICENSE.txt` beside
  them). The `c4_test` slice is text from C4 by AllenAI, released under ODC-BY 1.0; Common
  Crawl's terms of use apply to its content.
- **Bug-zoo scripts:** written for Preflight. They download public models and data at run time
  and bundle none of them:

| Script | Model (licence) | Data (licence) |
| --- | --- | --- |
| `finetune_lora_hf` | Qwen/Qwen2.5-1.5B (Apache 2.0) | yahma/alpaca-cleaned (CC BY 4.0) |
| `pretrain_nanogpt` | GPT-2 tokenizer (MIT) | roneneldan/TinyStories (CDLA-Sharing 1.0) or random tokens |
| `grpo_trl` | Qwen/Qwen2.5-0.5B-Instruct (Apache 2.0) | openai/gsm8k (MIT) |
| `agent_eval_spark` | muse-spark-1.3, hosted (Meta's terms) | google-research-datasets/mbpp (CC BY 4.0) |
| `eval_sweep_reasoning` | DeepSeek-R1-Distill-Qwen 1.5B and 7B (MIT; distilled from Qwen2.5, Apache 2.0) | openai/gsm8k (MIT) |
| `serving_vllm` | Qwen/Qwen2.5-7B-Instruct (Apache 2.0) | databricks/databricks-dolly-15k (CC BY-SA 3.0) |
| `retrieval_st` | sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0) | sentence-transformers/natural-questions (CC BY-SA 3.0, from Google's Natural Questions) |
| `gnn_pyg` | none (trained from scratch) | synthetic graph made in the script |
| `vision_timm` | timm/resnet50.a1_in1k (Apache 2.0) | AI-Lab-Makerere/beans (MIT) |
