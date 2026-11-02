# Portal answers (Cerebral Valley, Indie Developer track)

Draft by D, Sun Nov 1. Every number is in `README.md`'s table with its source. Check the field
names and length limits against the live form; lengths here are counted in characters.

## Project name

Preflight

## Tagline (58 characters)

A flight simulator for ML jobs, before you pay for the GPU

## Track and theme

Indie Developer. Build for Your World.

## Links

- Live app: TODO(PM)
- Repository: https://github.com/uchuuronin/preflight (public at submission)
- Demo video (3 minutes): TODO(PM)
- X post (45 seconds): TODO(PM)
- The corpus, with its datasheet: https://huggingface.co/datasets/chaoschia/preflight-corpus (public with the repository on Nov 3 to 4)

## Team

TODO(PM): every team member, as on the registration.

## Short description (332 characters)

Preflight runs a few minutes of your real training, evaluation, serving or agent job, measures
it, and lets you fly the whole run in the browser first: memory, time, cost and risk, each with a
range and a word for how it was made. On a fine-tune with a planted bug, it caught the
out-of-memory at step 200 in 189 seconds of probing.

## Full description

**The problem.** A training or evaluation run can look fine and die hours in: out of memory at
its first evaluation, a reward that never varies, an agent that re-pays its whole history every
turn. You find out after the GPU hours or API dollars are spent. Students with one assigned GPU
and labs planning a grid of runs both lose time this way.

**What Preflight does.** It reads your script, runs a few real steps in a GPU sandbox, and
builds a model of how the job behaves. Then you fly the run in the browser:
your job drawn as a map, four readouts (memory, time, cost, risk) with ranges, and knobs that move
everything at once. Every number says how it was made: Measured, Predicted, Extrapolated, or Not
trusted when Preflight is too far from anything it measured to give one. TODO(PM): at
submission, say what the live app offers: checking in your own script needs `runner.queue`, and
the plan to take away (settings, an edit script, a Slurm batch file, the command) needs
`c.exports`; until those are on, the site opens samples and shared records.

**What it caught in our test scripts.** We wrote one script per job type with problems planted
on purpose, and ran them for real:
- A LoRA fine-tune with a planted evaluation batch that is too large: the real full run on an L4
  ran out of memory at step 200, its first evaluation. Preflight's probe caught it in 189 seconds,
  for $0.042 of GPU at list price: the evaluation reaches 20.4 of the card's 22.06 GiB.
- An agent evaluation on Muse Spark with planted uncached prompts (the prompt starts with the
  current time, so its beginning changes on every call): across 130 tasks, $1.67 of Muse Spark
  calls at list price, not one of 905,739 input tokens came from the cache.
- A GRPO run on an L4 with a planted group setting: 15 of 20 prompts had no reward spread at all,
  so they teach nothing, and 95.3% of each step went to generating rollouts.

**How it works.** Probes run in Modal GPU sandboxes; an agent proxy counts every model call's
tokens. A prediction engine (analytical models, learned corrections, conformal intervals and
abstention) turns the probes into a flight record. Everything a visitor explores afterwards is
computed in the browser from that record. A campaign mode plans a whole grid of runs on a shared
cluster's calendar.

**Honest about what it doesn't know.** Predictions are checked against real full runs, hit or
miss, in the README's validation table (scored Mon Nov 2). What misses is said: step time on a GPU
Preflight never measured lands within 20% for 60% of held-out queries, under the 90% target, and
the divergence classifier, refitted on 720 runs, scores a held-out AUROC of 0.66 (0.56 to
0.76) against a 0.85 target, so it is switched off. There are no
AMD measurements yet, and every AMD figure says so.

**The corpus published with a datasheet.** The 2,339 rows of GPU measurements behind the learned
models and the held-out checks are a dataset on Hugging Face, public with the repository from
Nov 3 to 4 (https://huggingface.co/datasets/chaoschia/preflight-corpus), with a datasheet that
says how they were made, what they cover and where they are biased.

**Built during the hackathon.** Everything in the repository was built Oct 26 to Nov 4, 2026; the
first commit is `11a0ce9`, Oct 26, 22:25 ET.

## How we used Meta's models

- **Muse Spark** reads each script with structured output and draws the job map; it is the guide
  that moves the cockpit's knobs through tool calls (the Responses API); and it is the model inside
  the agent evaluation Preflight measured.
- **Muse Code**, headless, is the fix loop: it races candidate fixes in parallel sandboxed
  worktrees, edits only, and a fix is meant to land only when Preflight's own re-probe verifies
  it, never on Muse Code's exit code. In a clean smoke run Muse Code produced the correct patch in
  19 seconds; gating on the re-probe isn't wired end to end yet.
- **Muse Spark Open** on Baseten is built to triage each probe's log (next section); it hasn't
  run on a real job yet.

TODO(PM): at submission, say which of these are on in the live app (the guide is behind
`c.guide.live`, the fix loop behind `c.fix_loop`; see the README's Status section).

## Best Build on Meta Open Weight Models

**Muse Spark Open (`meta-models/muse-spark-open-1026` on Baseten) is built to triage each
probe's log.** After a job's probes, Preflight sends Spark Open the last 3,000 characters of each probe's output
with its status. Spark Open answers with structured output, at low reasoning effort: a category
from a fixed list (out of memory, missing package, data error, NaN or divergence, timeout, script
error, infrastructure, unclear), one plain sentence, and the log line it rests on, copied exactly.
Code then checks that quoted line really is in the log (at least 8 characters) before showing it;
for a probe that didn't finish, the result appears as a finding beside the probe's own error, and
every call is kept in the record's provenance (`python/preflight/harness/triage.py`).

How it runs today: on in production since Nov 1 (`harness.spark_open_triage`, with its line in
the data-use notice). Calls run one after another after the probes, outside the job's probe time
budget, and every call is in the record's provenance. The first real run, on our LoRA fine-tune
script with its planted eval bug (`a-triage-lora-prod`, GitHub run 41336726935), made 4 calls on
Baseten for $0.0184; the eval probe's call said "unclear", because the out-of-memory was in the
probe's recorded error rather than its output. Triage now reads that error too, and on all 11
failed probes in our test scripts' records Spark Open quoted a line every time
(`a-triage-agreement`). TODO(PM): add the production re-run's ID once A posts it.

**Muse Glimmer, Meta's open weights on our own GPU, for private mode.** With its flag on, a
private job gets triage from Muse Glimmer (Apache 2.0) on an A100-40GB on Modal that Preflight
deploys, scaled to zero when idle, so the job's logs go to no outside model provider. It is built
and hosted, behind `harness.glimmer_private` (off). On the same 11 failed probes it named the same
category as Spark Open 9 times. One call that waited through a cold start cost $0.24; the 11 calls
cost $0.33 together.

## Best Agentic Use-Case

The guide is an agent operating a real tool: Muse Spark calls the same actions a visitor has
(set a knob, highlight a part of the map, compare, propose a fix) and walks a tour of the levers
that matter for the visitor's goal, one move at a time, waiting for Apply, Revert or Pin. In
smoke run it called tools in all 5 rounds. The fix loop is the other agent: Muse Code races
fixes, and Preflight's own re-probe is to pick the winner by measurement (not wired end to end
yet). TODO(PM): say which is live.

## Challenges

- Measuring a few minutes of a job honestly: profiler overhead distorts small kernels, so timings
  come from unprofiled steps; optimizer state appears only after the first step; a CUDA context
  sits outside any model's tensors.
- Saying "not trusted" instead of a number when a question is too far from what was measured.
- Running strangers' code safely: GPU sandboxes, quotas, a spend cap and an edge that proxies
  every API call.

## What's next

AMD measurements once the AMD Developer Cloud account clears; the full-run validation table from
Mon Nov 2; private mode with Muse Glimmer switched on for visitors.
