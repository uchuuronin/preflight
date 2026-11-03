# Submission drafts

By D, Sun Nov 1 (updated Tue Nov 3), for the PM to finish and paste on Wed Nov 4 (internal deadline 6 PM ET). Every
number in these drafts is in the table below, with its source; nothing else may be added
without one. Dollar figures are measured time or tokens priced at list rates: tagged derived, as
the README tags them, and said "at list price" in the copy (Chiya, Nov 1). The fine-tune, agent
and GRPO examples are our own test scripts with problems planted on purpose, and the copy says so.
The links only Chiya can add (the video, the X post) say so where they go; the checklist is [`docs/submission.md`](../submission.md).

| File | What it is |
| --- | --- |
| `portal.md` | Answers for the Cerebral Valley portal, Indie Developer track. The field names follow the usual form; check them against the live form. |
| `gallery.md` | The gallery blurb, in two lengths. |
| `meta-and-partners.md` | Each Meta model and partner tool, where it is used, and what it does today. |
| `x-post.md` | The X post for the 45-second clip (`docs/demo/video-plan.md`, "The 45-second X clip", Nov 3's "cheapest place to crash" concept): the first post, two replies, the tags to check, alt text. |

## Numbers these drafts may use

| Number | What it is | Tag | Source |
| --- | --- | --- | --- |
| step 200 | Where the LoRA fine-tune zoo script's real run ran out of memory: its first evaluation | measured | A's full run `a-gt-lora-l4-5`, Oct 31 (`docs/status/a.md`), and the script (`EVAL_EVERY = 200`) |
| 189 s | Probe time until Preflight caught that out-of-memory (probe `p1-eval`) | measured | A's status, record `ICDYrfHQ2nBNbv3mlRHHMg` (run `a-smoke-l4-main-4`, Oct 28) |
| $0.042 | GPU cost of that probe, said "at list price" | derived (measured time x $0.80/h L4 list) | A's status |
| 20.4 of 22.06 GiB | Memory in use at the probe's peak, on an L4 | measured | A's status (21.9 GB) and the record's GPU (23,689,363,456 bytes) |
| 6 min 45 s | Upload to flight record for that job, over the plan's 5-minute target | measured | A's status, `a-smoke-l4-main-4` |
| 130 tasks | MBPP tasks the agent ran through Preflight's proxy, in two runs | measured | `a-agent-sample-1` (30, Oct 31) and `a-agent-builds-1` (100, Nov 1), A's status |
| $1.67 | Muse Spark calls for those 130 tasks, said "at list price" | derived ($0.4289 + $1.2371, the proxy's token counts at the price table) | A's status |
| 0 of 905,739 | Input tokens served from cache over those 130 tasks | measured | 0 of 246,463: the Oct 31 record (`zoo/agent_eval_spark/record_oct31.json`, its caching finding); 0 of 659,276: A's status (`a-agent-builds-1`) |
| 15 of 20 | GRPO prompts with zero reward spread on an L4 smoke run (75%, Wilson 95% 53 to 89%) | measured | `a-gt-grpo-l4-3`, Nov 1, A's status |
| 95.3% | Share of a GRPO step spent generating rollouts, same run | measured | `a-gt-grpo-l4-3`, A's status |
| 5 rounds | The guide calling tools in every round of its smoke run | measured | GitHub run 41273350602, `docs/status/c.md` |
| 461 of 575 | Held-out step-time queries within 15% on the same GPU (corpus) | measured | `docs/status/b.md` |
| 3 a day | Probe jobs a visitor may check in on the live site, under a site-wide daily GPU cap | cited (code) | `python/preflight/runner/queue.py`, `VISITOR_DAILY_JOBS` |
| 57% of 6,694 | Held-out cross-GPU step-time cases within 20% over five GPU types (A10, H100, L4, L40S, T4), Wilson 95% 55% to 59% treating cases as independent (they share architectures, so the honest interval is wider), under the 90% target (it was 60% on four GPUs); includes the Nov 1 top-up rows, which the final corpus 0.4.0 holds | measured | `docs/decisions/calibration-l4-phase1.md`, "The Nov 1 top-up", leave-one-GPU-out, K1 |
| 8 of 16 | Generated answers in one eval-sweep probe that ran to the 2,048-token cap without ending (95% interval 28% to 72%) | measured | D's `d-eval-sweep-a10`, GitHub run 41391396558, `zoo/eval_sweep_reasoning/record_nov2_a10.json` |
| 1,117 of 1,522 | Corpus rows where Muse Spark Open's failure label matched the rule label (73%, Wilson 95% 71% to 76%) | measured | `python/preflight/corpus/DATASHEET.md`, whole-corpus labels |
| 572 vs 600 of 641 | Divergence runs labelled right by Spark Open, against always answering "stable" (89% against 94%) | measured | the same table |
| $8.79 | Spark Open's calls labelling the whole corpus, the pilot included (all 2,339 rows sent, 1,997 answered) | measured (recorded) | the same section; `docs/status/b.md`, `b-spark-all-labels` |
| 23, 15 and 14 of 26 | Clean MBPP tasks that ended on a tool call (an upper bound on passes) with Muse Spark, Muse Glimmer and Muse Spark Open, the same rows | measured by the proxy | A's `a-pareto-muse-spark` (41325108987), `a-pareto-glimmer` (41363963718), `a-pareto-spark-open` (41325436143), `docs/status/a.md` |
| $0.39, about $0.22, $0.33 | Those three arms' model cost: Muse Spark's and Spark Open's calls by the proxy; Glimmer's A100-40GB for its 376 s run (the proxy's own $0.5383 sums each call's time and overcounts) | measured, derived (cold start aside), measured | the same rows |
| 4,255 rows | The published corpus, final file 0.4.0: 3,129 timing configs, 100 Hugging Face jobs, 1,026 divergence runs, an MI300X slice included (0.3.0 had 2,339 rows; Spark Open labelled those) | cited (the datasheet) | `python/preflight/corpus/DATASHEET.md`; https://huggingface.co/datasets/chaoschia/preflight-corpus (0.4.0 uploaded Nov 3; public with the repository) |
| $0.0184 | Spark Open's 4 triage calls on the first real production run | measured (recorded cost) | A's `a-triage-lora-prod`, GitHub run 41336726935, `docs/status/a.md` |
| 11 of 11 | Failed probes in the test scripts' records where Spark Open quoted a log line | measured | A's `a-triage-agreement`, GitHub run 41372018640 |
| 9 of 11 | The same failed probes where Muse Glimmer named the same category as Spark Open | measured | the same run |
| $0.33 | Muse Glimmer's 11 triage calls in that run, likely including one cold start | measured (recorded cost $0.3316) | the same run |
| A100-40GB | The GPU Muse Glimmer is hosted on, on Modal | cited | A's Glimmer runs, `docs/status/a.md`; B's app, `python/preflight/glimmer/` |
| $0.24, 411 s | One Muse Glimmer triage call, and the cold start it waited through | measured (recorded cost $0.2398) | A's `a-triage-glimmer-5`, GitHub run 41364641243 |
| 10.4 GiB, 0.11%, $0.39 | The fix loop on the LoRA sample's eval out-of-memory (on an A10): verify rejected Muse Code's single fix and passed the stacked fix at eval batch 8 (eval peak 10.4 GiB, loss within 0.11%); the race's cost at list price | measured; the dollars derived | the README's fix-loop row, C's race (`c.fix_loop`, on in production Nov 2) |
| 101 of 107, 93 of 102 | The number check (claim check) on planted claims: true numbers passed, altered ones caught | measured | B's `b-claim-check-eval`, the README's Muse Spark Open row |
| 5 to 8 minutes | Glimmer's first triage after a quiet spell, while it starts | measured | the README's Glimmer row; `docs/decisions/glimmer-host.md` (286 s) and `docs/status/a.md` (330 to 459 s) |
| 0.63x | The roofline's median NVIDIA-to-MI300X step prediction against the measured step | measured | `docs/validation/after-freeze.md`, the MI300X corpus slice |
| AUROC 0.66 (0.56 to 0.76) | The divergence classifier after a refit on 720 runs, held out by settings family, still under its 0.85 target, so it is off and not used (lead 138 steps, 3 to 269) | measured (B's fit report) | `python/preflight/corpus/reports/divergence_nov1.json` (`1cfa689`) |
| 3,000 characters | How much of each probe's log Spark Open triage reads | cited (code) | `python/preflight/harness/triage.py`, `TAIL_CHARS` |
| 8 characters | The shortest quoted log line code accepts from triage | cited (code) | `python/preflight/harness/triage.py` |
| 30 seconds | The longest each triage call may take | cited (A's status) | `docs/status/a.md`, known limits |

Not to be used: the derived "about $6.4" for a 500-task agent run, the derived "6 minutes and
$0.09" the fine-tune crash would have cost, anything from Mon Nov 2's scoring until B's scorer has
written it into the README, and the retrieval, GNN and vision probe runs until A logs them in
`docs/status/a.md`.
