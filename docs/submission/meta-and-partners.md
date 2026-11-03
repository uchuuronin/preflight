# Meta models and partner tools, and where each is used

Draft by D, Sun Nov 1, updated Nov 1, 21:30 ET (production flags as of the 8 PM change). The same table is in `docs/readme-final.md`; keep
the two the same. "Off" means built on `main` behind a flag that is off in production.

| Tool | What it does in Preflight | Where in the code | Today |
| --- | --- | --- | --- |
| **Muse Spark** (`muse-spark-1.3`, Meta Model API) | Reads each script with structured output (the script map) | `python/preflight/harness/mapper.py` | On in production |
| | The guide: tool calls on the live cockpit through the Responses API | `api/preflight_api/guide/` | On in production (`c.guide.live`) |
| | The model inside the agent evaluation Preflight measured (130 tasks) | `zoo/agent_eval_spark/` | Measured runs, Oct 31 and Nov 1 |
| **Muse Code** (headless) | The fix loop: candidate fixes raced in parallel sandboxed worktrees, edits only, each meant to be gated on Preflight's re-probe | `api/preflight_api/fix/`, `api/fix_app.py` | Off (`c.fix_loop`); in a clean smoke run Muse Code produced the correct patch in 19 s; gating on the re-probe isn't wired end to end yet |
| **Muse Spark Open** (`meta-models/muse-spark-open-1026`, Baseten) | Triage of each probe's log: a category, one sentence, and a quoted log line that code checks | `python/preflight/harness/triage.py` | On in production (`harness.spark_open_triage`); first real run `a-triage-lora-prod`, 4 calls, US$0.0184; quoted a line on all 11 failed zoo probes (`a-triage-agreement`) |
| | Claim check: which record field a sentence's number means, checked by code | `python/preflight/harness/claim_check.py` | Written, not wired in |
| **Muse Glimmer** (open weights, Apache 2.0) | Private mode: triage of a private job's failed probes on Preflight's own GPU; also a row in the agent model sweep | `python/preflight/glimmer/`, `python/preflight/runner/pipeline.py`, `docs/decisions/glimmer-host.md` | Off (`harness.glimmer_private`); hosted on an A100-40GB on Modal that Preflight deploys, scaled to zero; US$0.24 for one call through a 411 s cold start, US$0.33 for 11 calls in one run (likely one cold start among them); same category as Spark Open on 9 of 11 failed zoo probes |
| **Modal** | GPU sandboxes for probes, the job store and API, the corpus sweep, the fix loop's sandboxes | `python/preflight/runner/`, `api/`, `python/preflight/corpus/` | Every measured run so far; public deploy TODO(PM) |
| **Baseten** | Hosts Muse Spark Open | `python/preflight/harness/spark.py` | Used by triage, on in production |
| **AMD Developer Cloud** | ROCm probes on an MI300X through the CLI, and the AMD half of the corpus | `.github/workflows/amd-run.yml`, `python/preflight/corpus/` | Measured Nov 2 on a rented MI300X: probes and full runs of two zoo scripts (same-GPU step time 2 of 2 within 15%, the LoRA eval OOM caught) and 295 corpus configs (NVIDIA to MI300X step time within 20% on 22% of 1,147 pairs, so a move across vendors abstains) |
| **Cloudflare Pages** | The web app, the `/v1` edge proxy, and share cards | `web/`, `web/functions/`, `workers/` | Built; live link TODO(PM) |
