# Submission drafts

By D, Sun Nov 1, for the PM to finish and paste on Wed Nov 4 (internal deadline 6 PM ET). Every
number in these drafts is in the table below, with its source; nothing else may be added
without one. Dollar figures are measured time or tokens priced at list rates: tagged derived, as
the README tags them, and said "at list price" in the copy (Chiya, Nov 1). The fine-tune, agent
and GRPO examples are our own test scripts with problems planted on purpose, and the copy says so.
Placeholders are marked TODO(PM).

| File | What it is |
| --- | --- |
| `portal.md` | Answers for the Cerebral Valley portal, Indie Developer track. The field names follow the usual form; check them against the live form. |
| `gallery.md` | The gallery blurb, in two lengths. |
| `meta-and-partners.md` | Each Meta model and partner tool, where it is used, and what it does today. |
| `x-post.md` | The X post for the 45-second cut (`docs/demo/storyboard.md`), with alt text. |

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
| 19 s | Muse Code producing the correct patch in a clean smoke run of the fix loop (gating on the re-probe isn't wired end to end yet) | measured | GitHub run 41250977670, `docs/status/c.md` |
| 5 rounds | The guide calling tools in every round of its smoke run | measured | GitHub run 41273350602, `docs/status/c.md` |
| 461 of 575 | Held-out step-time queries within 15% on the same GPU (corpus) | measured | `docs/status/b.md` |
| 60% | Held-out step-time queries within 20% on a GPU the model never saw, under the 90% target | measured | `docs/status/b.md` (leave-one-GPU-out) |
| 2,339 rows | The published corpus (corpus@0.3.0): 1,519 kernel configs, 100 Hugging Face jobs, 720 divergence runs; on Hugging Face, private until Nov 3 to 4 | cited (the datasheet) | `python/preflight/corpus/DATASHEET.md`; https://huggingface.co/datasets/chaoschia/preflight-corpus (commit `0aac218`) |
| $0.0184 | Spark Open's 4 triage calls on the first real production run | measured (recorded cost) | A's `a-triage-lora-prod`, GitHub run 41336726935, `docs/status/a.md` |
| 11 of 11 | Failed probes in the test scripts' records where Spark Open quoted a log line | measured | A's `a-triage-agreement`, GitHub run 41372018640 |
| 9 of 11 | The same failed probes where Muse Glimmer named the same category as Spark Open | measured | the same run |
| $0.33 | Muse Glimmer's 11 triage calls in that run, likely including one cold start | measured (recorded cost $0.3316) | the same run |
| A100-40GB | The GPU Muse Glimmer is hosted on, on Modal | cited | A's Glimmer runs, `docs/status/a.md`; B's app, `python/preflight/glimmer/` |
| $0.24, 411 s | One Muse Glimmer triage call, and the cold start it waited through | measured (recorded cost $0.2398) | A's `a-triage-glimmer-5`, GitHub run 41364641243 |
| AUROC 0.66 (0.56 to 0.76) | The divergence classifier after a refit on 720 runs, held out by settings family, still under its 0.85 target, so it is off and not used (lead 138 steps, 3 to 269) | measured (B's fit report) | `python/preflight/corpus/reports/divergence_nov1.json` (`1cfa689`) |
| 3,000 characters | How much of each probe's log Spark Open triage reads | cited (code) | `python/preflight/harness/triage.py`, `TAIL_CHARS` |
| 8 characters | The shortest quoted log line code accepts from triage | cited (code) | `python/preflight/harness/triage.py` |
| 30 seconds | The longest each triage call may take | cited (A's status) | `docs/status/a.md`, known limits |

Not to be used: the derived "about $6.4" for a 500-task agent run, the derived "6 minutes and
$0.09" the fine-tune crash would have cost, anything from Mon Nov 2's scoring until B's scorer has
written it into the README, and the retrieval, GNN and vision probe runs until A logs them in
`docs/status/a.md`.
