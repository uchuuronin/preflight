# Portal answers (Cerebral Valley, Indie Developer track)

Draft by D, Sun Nov 1; updated Tue Nov 3 to the product at the freeze. Every number is in `README.md`'s table with its source. Check the field
names and length limits against the live form; lengths here are counted in characters.

## Project name

Preflight

## Tagline (58 characters)

A flight simulator for ML jobs, before you pay for the GPU

## Track and theme

Indie Developer. Build for Your World.

## Links

- Live app: https://preflight-cqn.pages.dev
- Repository: https://github.com/uchuuronin/preflight (public at submission)
- Demo video (3 minutes): Chiya adds the link on Wed Nov 4
- X post (45 seconds): Chiya adds the link on Wed Nov 4
- A shared flight record: https://preflight-cqn.pages.dev/s/ICDYrfHQ2nBNbv3mlRHHMg
- The corpus, with its datasheet: https://huggingface.co/datasets/chaoschia/preflight-corpus (public with the repository)

## Team

Chiya fills this in: every team member, as on the registration.

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
measured here when Preflight is too far from anything it measured to give one. On the live site you can
check in your own repository or script (3 probe jobs a day while the site's GPU budget lasts), ask
the guide, and take the plan away: the settings, an edit script, a Slurm batch file, the command
and a Weights & Biases helper. Or open a sample or a shared record, at no cost.

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
- An evaluation sweep of reasoning models with no cap on thinking: in one probe on an A10, 8 of 16
  generated answers ran to the 2,048-token cap without finishing, so they'd be cut off and scored
  wrong.

**How it works.** Probes run in Modal GPU sandboxes; an agent proxy counts every model call's
tokens. A prediction engine (analytical models, learned corrections, conformal intervals and
abstention) turns the probes into a flight record. Everything a visitor explores afterwards is
computed in the browser from that record. A campaign mode plans a whole grid of runs on a shared
cluster's calendar.

**Honest about what it doesn't know.** Predictions are checked against real full runs, hit or
miss, in the README's validation table (scored Nov 2; final table Nov 3). What misses is said: across five GPU
types (A10, H100, L4, L40S, T4), step time on a GPU Preflight never measured lands within 20% for
57% of 6,694 held-out cases (the Nov 1 top-up included, which corpus
0.4.0 holds), under the 90% target, so it is shown with a wide band and labelled rough; and the divergence classifier, refitted on 720 runs, scores a held-out AUROC of 0.66 (0.56 to
0.76) against a 0.85 target, so it is switched off. On an MI300X,
step time from its own probe lands within 15% (2 of 2 runs), but from NVIDIA corpus rows it is 22% within 20%
(NVIDIA torch 2.8.0 to MI300X torch 2.5.1+rocm6.2: card and software differ), 254 of 1,147 pairs, so a move across vendors isn't shown.

**The corpus published with a datasheet.** The 4,255 rows of GPU measurements behind the learned
models and the held-out checks (corpus 0.4.0, an MI300X slice included) are a dataset on Hugging
Face, public with the repository (https://huggingface.co/datasets/chaoschia/preflight-corpus),
with a datasheet that says how they were made, what they cover and where they are biased. Muse
Spark Open was asked to label every row of the first 2,339 (which part of memory dominates its peak, or whether and how a run diverged) for
$8.79 of calls; set against our rule labels it agreed on 1,117 of 1,522 rows compared, and the
datasheet says where it falls short: on divergence runs it did worse than simply answering
"stable".

**Built during the hackathon.** Everything in the repository was built Oct 26 to Nov 4, 2026; the
first commit is `11a0ce9`, Oct 26, 22:25 ET.

## How we used Meta's models

- **Muse Spark** reads each script with structured output and draws the job map; it is the guide
  that moves the cockpit's knobs through tool calls (the Responses API); and it is the model inside
  the agent evaluation Preflight measured.
- **Muse Spark** also reads each finished record's memory timeline and loss plot, as images, with
  the raw probe numbers beside them, and reports where the two disagree (the diagnosis).
- **Muse Code**, headless, is the fix loop: it writes candidate fixes in sandboxes, edits only, and
  a fix lands only when Preflight's own re-probe verifies it, never on Muse Code's exit code. On
  the LoRA script's eval out-of-memory, verify rejected Muse Code's single fix (the eval still ran
  out of memory) and passed the stacked fix at the guide's eval batch of 8: eval peak 10.4 GiB,
  loss within 0.11% ($0.39 of the race at list price).
- **Muse Spark Open** on Baseten triages each failed probe's log on the live site, checks the
  numbers in model-written findings (code alone for a private check-in, within a US$0.10 cap a job; next section), and labelled the corpus.
- **Muse Glimmer**, Meta's open weights on our own GPU, triages private check-ins on the live site
  (next section).
- **All three in one agent evaluation.** On the same 26 clean programming tasks, run through
  Preflight's proxy, the task ended on a tool call (an upper bound on passes) 23 times with Muse
  Spark ($0.39 of calls), 15 with Muse Glimmer (about $0.22 of its GPU's time, cold start aside)
  and 14 with Muse Spark Open ($0.33 of calls).

On the live site, all of it is on: the script map, the guide, the diagnosis, Muse Spark's search grounding on failed
probes, the fix loop, Spark Open's triage and number check, and Glimmer's private triage.

## Best Build on Meta Open Weight Models

**Muse Spark Open (`meta-models/muse-spark-open-1026` on Baseten) triages each probe's log.** After a job's probes, Preflight sends Spark Open the last 3,000 characters of each probe's output
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
(`a-triage-agreement`). Spark Open also labelled the corpus: it was sent all 2,339 rows,
answered for 1,997, and the datasheet sets its labels against ours, misses included. The
production re-run (`a-triage-lora-prod-2`, GitHub run 41479039229, Nov 2, record
01M6EGH5YMCXPFPNY5SHPNSYV5) quotes the line: "Ran out of GPU memory during the eval pass.
OutOfMemoryError: CUDA out of memory. Tried to allocate 530.00 MiB.".

**Spark Open also checks the numbers in every model-written finding** (`harness.spark_claim_check`,
on in production since Nov 2): code finds the record fields each number could mean, and Spark Open
only picks among them. On planted claims, true numbers passed 101 of 107 times and altered ones
were caught 93 of 102 (`python/preflight/harness/claim_check.py`).

**Muse Glimmer, Meta's open weights on our own GPU, for private mode.** A check-in with "Private
triage" ticked gets its triage from Muse Glimmer (Apache 2.0) on an A100-40GB on Modal that
Preflight deploys, scaled to zero when idle, so the job's logs go to no outside model provider. It
is on in production since Nov 2 (`harness.glimmer_private`); the first triage after a quiet spell
waits 5 to 8 minutes while it starts. On the same 11 failed probes it named the same
category as Spark Open 9 times. One call that waited through a cold start cost $0.24; the 11 calls
cost $0.33 together.

## Best Agentic Use-Case

The guide is an agent operating a real tool: Muse Spark calls the same actions a visitor has
(set a knob, highlight a part of the map, compare, propose a fix) and walks a tour of the levers
that matter for the visitor's goal, one move at a time, waiting for Apply, Revert or Pin. In
smoke run it called tools in all 5 rounds, and it is on in the live site. The fix loop is the
other agent: Muse Code writes fixes, and Preflight's own re-probe picks the winner by measurement;
on the LoRA script's eval out-of-memory it rejected the fix that didn't work and passed the one
that did. Both are on in the live site.

## Challenges

- Measuring a few minutes of a job honestly: profiler overhead distorts small kernels, so timings
  come from unprofiled steps; optimizer state appears only after the first step; a CUDA context
  sits outside any model's tensors.
- Saying "not measured here" instead of a number when a question is too far from what was measured.
- Running strangers' code safely: GPU sandboxes, quotas, a spend cap and an edge that proxies
  every API call.

## What's next

A cross-vendor step-time model that holds on the MI300X (its corpus slice, measured Nov 2, shows the roofline
reading 0.63x); more full runs behind the validation table, whose rows still rest on a few runs
each; and harnessing more unseen scripts on their first try (0 of 9 so far).
