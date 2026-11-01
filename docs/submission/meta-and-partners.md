# Meta models and partner tools, and where each is used

Draft by D, Sun Nov 1, as of `main` `a52753f`. The same table is in `docs/readme-final.md`; keep
the two the same. "Off" means built on `main` behind a flag that is off in production.

| Tool | What it does in Preflight | Where in the code | Today |
| --- | --- | --- | --- |
| **Muse Spark** (`muse-spark-1.3`, Meta Model API) | Reads each script with structured output (the script map) | `python/preflight/harness/mapper.py` | On in production |
| | The guide: tool calls on the live cockpit through the Responses API | `api/preflight_api/guide/` | Off (`c.guide.live`); smoke runs pass |
| | The model inside the agent evaluation Preflight measured (130 tasks) | `zoo/agent_eval_spark/` | Measured runs, Oct 31 and Nov 1 |
| **Muse Code** (headless) | The fix loop: candidate fixes raced in parallel sandboxed worktrees, edits only, each meant to be gated on Preflight's re-probe | `api/preflight_api/fix/`, `api/fix_app.py` | Off (`c.fix_loop`); in a clean smoke run Muse Code produced the correct patch in 19 s; gating on the re-probe isn't wired end to end yet |
| **Muse Spark Open** (`meta-models/muse-spark-open-1026`, Baseten) | Triage of each probe's log: a category, one sentence, and a quoted log line that code checks | `python/preflight/harness/triage.py` | Off (`harness.spark_open_triage`); not yet run on a real job |
| | Claim check: which record field a sentence's number means, checked by code | `python/preflight/harness/claim_check.py` | Written, not wired in |
| **Muse Glimmer** | A row in the agent model sweep's price table; a private mode is proposed | `python/preflight/engine/prices/`, `docs/decisions/glimmer-host.md` | Not hosted: deferred until its terms are confirmed |
| **Modal** | GPU sandboxes for probes, the job store and API, the corpus sweep, the fix loop's sandboxes | `python/preflight/runner/`, `api/`, `python/preflight/corpus/` | Every measured run so far; public deploy TODO(PM) |
| **Baseten** | Hosts Muse Spark Open | `python/preflight/harness/spark.py` | Used by triage once its flag is on |
| **AMD Developer Cloud** | ROCm probes on an MI300X through the CLI, and the AMD half of the corpus | `.github/workflows/amd-run.yml`, `python/preflight/corpus/` | No AMD measurements: the account is on a verification hold |
| **Cloudflare Pages** | The web app, the `/v1` edge proxy, and share cards | `web/`, `web/functions/`, `workers/` | Built; live link TODO(PM) |
