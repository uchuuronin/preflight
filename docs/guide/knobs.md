# How to fill this in: every input and knob, explained

Preflight is a flight simulator for ML jobs. You hand it your code, it flies a few minutes of the
real thing on a real GPU, and then it lets you fly the whole run in the cockpit before you pay for
it. This page is the pilot's manual: every box on **New flight**, then every knob in the cockpit, in
plain words. You don't need to have trained a model before. Where a word is jargon, the page says
what it means the first time it comes up. For the engine's own account of each knob, line by line
from the code, see [`knobs-reference.md`](knobs-reference.md).

Two words used everywhere below:

- **A probe** is a short, real run of your code: a few training steps (or a few tasks, for an agent)
  on a real GPU, timed and measured. Preflight runs a handful of them when you check in.
- **The guide** is the co-pilot in the cockpit: press **Start the tour**. It walks you through the
  knobs that matter for your aim, one move at a time, and never moves anything you've pinned or
  ruled out.

## What to write in "What are you trying to do?"

This box is your job in your own words: what it's for, what it runs on, what "done" means, and what
you can't change. A good answer is one or two sentences with the numbers you actually know.

**What happens to it.** Preflight saves your words with the job's record and hands them to the
guide: when the cockpit opens and you press **Start the tour**, they're waiting in the guide's box,
**What are you doing, and what matters most?** Press **Ask the guide** and Muse Spark reads them and
plans its moves around them. Two more things steer it, which is why each example below says what to
pick and type:

- **What matters most** on New flight: the guide's aim (lowest cost, finish soonest or most
  headroom).
- **Your limits** in the cockpit: type your hard limits in plain words ("must stay on an A100, under
  $40, done in 6 hours"), press **Read my limits**, and Muse Spark turns them into limits you
  confirm. The guide never suggests anything outside a confirmed limit.

So write your description here, and once the cockpit opens, send it to the guide and put the hard
numbers in **Your limits**. (If the cockpit says the live guide is off, the rule-based tour runs
instead: it follows **What matters most** and your limits, not your words.)

### Examples, and what the guide does with each

**A finished script you want to run for real.**

> Fine-tune Qwen2.5-1.5B with LoRA on our support tickets, one A100, done by Friday, cheapest
> possible.

Pick **Lowest cost**. In the cockpit's limits, write "only A100, done within 72 hours". The guide
first checks the run fits in memory (a predicted out-of-memory is always fixed first), then ranks
the knobs by how much they cut cost on an A100 and proposes one move at a time: say, a bigger batch
with fewer accumulation steps. For each one you press **Apply ...**, **Not this**, **Keep ...**
(your value) or **Pin ...** (the suggestion). One thing to know: the public site probes on cards up
to an L40S, so an A100's time and cost come from Preflight's cross-GPU model, read **Extrapolated**
and are rough (their range runs from about 1.55 to 1.90 times either way [derived: the engine's
held-out cross-GPU bands]). Memory on the A100 is still worked out from your probes.

**No training loop yet: you're planning.**

> I want to pretrain a small GPT on TinyStories but haven't written the loop yet. What GPU and batch
> size should I budget for?

Preflight measures real code, so a plan with no code has nothing to probe yet. Open the sample
closest to your plan instead (here **Pretrain a GPT-2 (124M) on TinyStories, measured on an A10 (2
Nov)**) and move its knobs. Batch size, sequence length, training steps and precision move memory,
time and cost. Layers and model width move memory, but not time yet: the cockpit holds time at the
probes' value, says "isn't modelled for time yet", and widens the range. When your loop exists,
check it in for numbers measured on your code.

**No GPU chosen yet.**

> Train a ResNet-50 on 20,000 leaf photos. I don't know which GPU to rent; I just want it under $10.

Leave **Your GPU** on **L4 or A10** (or press **Let Preflight pick**), and pick **Lowest cost**. In
the cockpit, write "under $10" in the limits. Then open the **GPU** knob: memory, time and cost move
for each card, and the guide only suggests cards that fit and stay under $10. A card the probes
never ran on is predicted from the one they did, reads **Extrapolated**, and its range is wide; the
readout says so. (A sample measured on one card may offer only that card for now.)

**Just "will it fit?"**

> Will a 7B LoRA fine-tune with 4,096-token sequences fit on a 24 GB card?

Pick **Most headroom**. Check in the script with the GPU you mean (the default L4 and A10 are both
24 GB cards [cited: NVIDIA's spec pages], but when neither is free the job moves to a 48 GB [cited:
NVIDIA's spec pages] L40S: check which card the memory readout names). The memory readout shows the
predicted peak against the card's memory, with its range, and the chance it runs out of memory
beside it. If it doesn't fit, the guide's first move is the one that makes it fit: 4-bit weights (a
7B model's weights alone take about 14 GB in 16-bit [derived: 7 billion x 2 bytes]), checkpointing,
a smaller eval batch, and shorter sequences only if nothing else works.

**An agent or API job.**

> An agent that answers support questions with tool calls on Muse Spark, 500 tasks, under $20.

Check in the agent's script. Instead of GPU probes, Preflight runs a sample of real tasks (20 to 50
[cited: the engine's agent cost model]) and scales their tokens, cost and time to your 500, with a
range widened for runaway tasks the sample may have missed. Pick **Lowest cost** and write "under
$20" in the limits. The knobs here are about calls, not GPUs: **Max turns** cuts runaway tasks (and
their bills), **Concurrency** finishes sooner, **Tasks** scales everything. A **Model** the sample
didn't run on can't be priced until you sample it.

**An evaluation.**

> Score pythia-160m on ARC-Easy with lm-evaluation-harness, 100 questions, as fast as possible.

Use the **Run a command** tab: `lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m
--tasks arc_easy --limit 100`. Pick **Finish soonest**. The probes measure the evaluation's memory
and time. Preflight's model of an evaluation's scoring passes is behind a setting that's off on the
public site, so read the finish time as rough and check its status word.

**Something the boxes don't cover.** Write it anyway ("must reuse our cluster's checkpoints", "the
data can't leave our bucket"). If you type it into **Your limits** and Preflight has no knob or
number for it, it says so plainly ("Not turned into a limit: ...") rather than guessing.

## The check-in form

### Where your job lives

Four ways in, as tabs:

- **Repo link:** a public GitHub repository. The branch (or the default one) is pinned to its commit
  when you check in, so later changes to the repo don't change your record.
- **Upload:** `.py` files plus requirements, configs and small text data, 5 MB in all and at most
  2,000 files. Weights and big datasets come from Hugging Face, not the upload. Pick which file to
  run.
- **Notebook:** a single notebook, checked in on its own. Its `pip` lines become its requirements,
  and the tab shows what will run.
- **Run a command:** a command Preflight knows by name (today that's `lm_eval`, for
  lm-evaluation-harness), at most 64 words [cited: the form's limit]. It runs as written without a
  shell, so no pipes (`|`), redirects (`>`), variables or second commands. The box says which word
  it can't take.

### Extra packages

PyPI names your code imports that might not be installed, separated by commas (`einops, timm`), up
to 10 [cited: the form's limit]. Names only, without versions. Preflight installs names on its own
list, ready-built (wheels only), before the run; a name it doesn't install stops the check-in before
any GPU time is spent, so a typo costs you nothing.

### What matters most

The guide's aim, and the readout it treats as your goal:

- **Lowest cost:** fewest dollars for the whole run (GPU time and tokens).
- **Finish soonest:** shortest wall-clock time.
- **Most headroom:** the most room between the memory peak and the card's memory.

You can change it in the cockpit later: press a dial to make it your goal.

### Your GPU

**L4 or A10** starts on an L4; if none is free within two minutes it moves to an A10, then an L40S,
and a script that runs out of memory may be measured on a bigger card. (From Nov 5 to 8, while the
judges try it, the default starts on an A10.) You can also pick **L40S**, **A100-SXM-80GB** or
**H100-SXM-80GB**, or press **Let Preflight pick**, which takes the cheapest card with room to spare
for a rough estimate of what your job needs. Probes on the public site run on cards up to an L40S:
pick an A100 or H100 and the probes run on a smaller card, and that card's numbers come from
Preflight's cross-GPU model. Pick the card you'll really use if you know it.

### Private triage

Tick it and your failed probes' logs are read by Muse Glimmer, Meta's open-weights model, on
Preflight's own GPU, so Meta and Baseten never see them. Muse Spark still reads your code to draw
the map, and the check of your plots against your numbers is skipped. Glimmer naps when idle: waking
it took 5 to 8 minutes on 1 Nov [measured], then a triage takes seconds.

### Check in

Free, with no account: Preflight pays for the GPU minutes, up to 3 jobs a day (a check-in, or a
probe or check from the cockpit), and up to 3 queued or running at once [derived: the queue's
limits]. Days run midnight to midnight, US Eastern. Past your 3 jobs, or once the site's daily GPU
budget is spent, the site opens samples and shared records only until midnight. A check-in takes
about 5 to 11 minutes (production's two check-ins of the LoRA sample took 4 min 59 s and 10 min 21 s
[measured]), longer if you wait for a free GPU: the boarding screen says so, and the wait is capped
at 20 minutes [derived: the runner's wait cap]. If a check-in stops at an import before measuring
anything, one resend a day is free and doesn't count against the 3 [derived: the queue's
`FREE_RESENDS_PER_DAY`].

## Reading the cockpit

The middle of the map is **Peak memory**: the predicted peak against the card's memory, marked
**Fits**, **Tight** or **Fails**. The dials beside it are **Time to finish**, **Run cost** (GPU and
tokens) and the chance the run breaks: runs out of memory for training, runs away for an agent. Each
carries a range the true value should fall inside 9 times in 10 [cited: the engine's 90% level], and
a word for where it came from:

- **Measured:** a probe ran at exactly these settings on this GPU. Time to finish and cost never
  read Measured: the step is measured, but the whole run is that step multiplied out.
- **Predicted:** worked out from nearby measured settings, inside what the probes covered. Should
  hold.
- **Extrapolated:** worked out past what was measured (another GPU, a knob off its probed values, a
  knob the model doesn't capture yet). A rough guide; the reason says why.
- **Not measured here:** nothing measured is close enough, so Preflight shows no number on purpose.
  Move back towards a measured setting, or run a probe there.

The out-of-memory words: **Fits** is under a 1% chance, **Tight** 1% to 80%, **Fails** 80% or more
[simulated: the engine's cut-offs, not yet checked against real outcomes]. A chance between 5% and
95% [derived: the engine's unverified range] is shown as a range rather than one number, because
those haven't been checked on real runs yet. **Tight: measure first** means run a probe before you
trust it. A record marked **Example numbers** shows how the cockpit works; nothing in it was
measured.

How far to trust the ranges so far: against real full runs, the 90% ranges held 10 times out of 18
[measured: Preflight's validation table, Nov 3], so treat a **Predicted** range as narrower than it
should be.

## How knobs behave

### Moving a knob

Drag it or pick a value and the map, the readouts and the timeline answer straight away. Each knob's
hint says what it moves first: memory, speed and cost, how batches are built, the model, where it
runs, the RL loop or the agent's calls.

### Knobs that only change what the model learns

Learning rate, its schedule, warm-up, weight decay, gradient clip, LoRA alpha and the RL clip ranges
change how the model learns, not how much memory or time a step takes. Move them and memory, time
and cost stay as they were, with the same status. Preflight doesn't predict learning quality.

### Isn't modelled for time yet

Some knobs move memory, or would move time, in ways Preflight's time model doesn't capture yet: the
model's shape, LoRA rank, quantisation, the optimiser, the attention kernel, data workers, packing
and others, each marked below. Move one off the value your probes ran and time stays at the probes'
value, reads **Extrapolated** with "isn't modelled for time yet; held at its probed effect", and its
range widens by half again [guess: the engine's widening factor]. If that range would pass plus or
minus 50% [cited: the project's abstain rule], the dial shows **Not measured here** instead.

### Off the probed values

Any knob set to a value no probe ran turns memory and time from **Predicted** to **Extrapolated**,
except the size knobs (batch, sequence length, image size, the fan-outs), run length, the GPU and
the GPU count, the save and eval intervals, gradient accumulation (for time) and the knobs that only
change what the model learns. Past twice the largest probe [cited: the engine's extrapolation limit]
on the job's size axis (batch times sequence length, say), memory and time show **Not measured
here**.

### Pinned

**Pin at ...** freezes a knob at its value; the guide will never move it. Unpin it to let the guide
use it again.

### The guide suggests this

A knob the guide has proposed shows its suggested value. Move it yourself and your value stands.

### Code

Some knobs can't be changed from outside your script. They carry a **Code** badge: changing one
means editing your code, which Muse Code writes and a short verify run checks before you get it.
Nothing runs until you say so, and a verify run or a probe counts toward your 3 jobs a day.

### Not predicted here yet

The knob's value is outside what the probes covered. A probe at that setting would measure it.

### Your limits

Set what the guide must respect: allowed GPUs, a budget, a deadline, a GPU-hour quota, knobs not to
change. Type them in plain words and press **Read my limits**, or set them by hand. Nothing applies
until you confirm it. A choice you turn down with **Not this** is ruled out too, and you can allow
it again.

## The knobs

Not every job has every knob: the cockpit shows the ones your code actually sets. They're grouped
here by the kind of job they usually belong to. Each ends with what Preflight does with it today.

### Training: memory first

#### Batch size

How many examples the model sees in one step. Bigger is usually faster per example and needs more
memory. The first knob to look at when a run doesn't fit, and the first to raise when it fits with
room to spare. It also changes how the model learns, unless gradient accumulation keeps batch times
accumulation the same; and with a fixed number of steps, a smaller batch means fewer examples seen
in all. *In Preflight:* moves memory, time per step and the number of steps; total time and cost go
whichever way the product goes.

#### Gradient accumulation

Runs several small batches before each weight update and averages their gradients, so batch 8 with
accumulation 4 learns like batch 32 with roughly batch 8's memory. Costs a little speed. *In
Preflight:* on its own it adds a little memory (one copy of the gradients is kept between batches);
the saving comes from lowering the batch size alongside it. Time per update goes up and updates per
epoch go down by the same factor.

#### Eval batch

The batch size for the evaluation pass, separate from training. A classic trap: training fits but
the first evaluation runs out of memory because its batch is bigger. Lower it if the memory readout
spikes at eval. *In Preflight:* moves memory when evaluation sets the peak; time isn't modelled for
it (each eval pass is counted at the probes' eval batch).

#### Eval without gradients

Runs evaluation without keeping what training needs to learn (`torch.no_grad()`), so it uses far
less memory. Almost always on; if your code forgets it, this knob is a free win. *In Preflight:*
moves memory (on, the eval pass holds one layer's in-between results; off, all of them); time isn't
modelled for it yet.

#### Sequence length

How many tokens one example holds; longer rows are cut there. Memory and time grow with it
(attention grows faster than linearly). Shorten it only if nothing else makes the run fit, because
it changes what the model can learn from long examples. *In Preflight:* moves memory and time only
while your rows are longer than the limit: when the probes scanned your dataset, raising it past
your longest row changes nothing. Past the length the dataset was scanned at, time reads
**Extrapolated**.

#### Checkpointing

Gradient checkpointing: forgets some in-between results during the forward pass and recomputes them
when learning. Saves a lot of memory for one extra forward pass of time [derived: the engine adds
one recompute pass]. Not to be confused with saving the model to disk (that's **Save every**). *In
Preflight:* turning it on moves memory down and time up; turning it off when the probes ran with it
on keeps the recompute in the time, labelled **Extrapolated**.

#### Precision

The number format for the maths: 32-bit is precise and big, `bf16` and `fp16` are half the size
[derived: 16 bits against 32] and much faster on modern GPUs. `bf16` is the safe default on cards
that support it (A10, L4, A100, H100 [cited: NVIDIA's spec pages]); `fp16` has a narrow range, so
small gradients can round to zero unless loss scaling is on. *In Preflight:* moves memory and time,
labelled **Extrapolated** whenever it differs from what the probes ran; a format the card has no
fast path for (bf16 on a T4) shows **Not measured here**.

#### Weight quantisation

Stores the frozen base model's weights in 8-bit or 4-bit NF4 instead of 16-bit, cutting their memory
by about half or three quarters (NF4 is about 0.56 bytes a weight against 2 [derived: 4 bits plus a
scale per 64 weights]). Often used with LoRA: LoRA on 4-bit weights is called QLoRA. A little slower
per step. *In Preflight:* moves memory; time isn't modelled for it yet.

#### Optimiser

The rule that turns gradients into weight updates. AdamW is the default; it keeps two extra numbers
per weight [cited: AdamW's two moving averages], which is a lot of memory for a big model. Lion
keeps one, plain SGD none, and 8-bit versions store each in a byte. *In Preflight:* moves memory (an
optimiser it doesn't know is counted as AdamW, to be safe); time isn't modelled for it yet.

#### Attention kernel

Which implementation computes attention. Fused kernels such as FlashAttention or PyTorch's SDPA are
faster and use less memory than the plain ("eager") version, which keeps a table that grows with the
square of the sequence length. *In Preflight:* moves memory; time isn't modelled for it yet.

#### Channels-last layout

For image models: stores pictures as height, width, channels instead of channels first, which lets
tensor cores work faster with mixed precision. Usually a free speed-up for CNNs. *In Preflight:* not
predicted: a verify run measures the speed-up.

### Training: what the model learns

#### Learning rate

How big a step the model takes each update. Too high and the loss jumps around or blows up (NaN);
too low and it barely moves. *In Preflight:* changes no number (see "Knobs that only change what the
model learns").

#### Learning rate schedule

How the learning rate changes over the run: constant, linear decay, cosine. Cosine with a short
warm-up is the common default. *In Preflight:* changes no number.

#### Warm-up steps

Steps at the start where the learning rate ramps up from near zero, so the first noisy updates don't
knock the model off course. *In Preflight:* changes no number.

#### Weight decay

A gentle pull of the weights towards zero each step, which helps the model generalise. Small values
are typical (0.01 to 0.1 [guess: common defaults]). *In Preflight:* changes no number.

#### Gradient clip

Scales the gradient down when its overall size passes a cap, which stops a single bad batch from
blowing the run up. Commonly 1.0 [guess: a common default]. *In Preflight:* changes no number.

#### LoRA rank

LoRA trains small add-on matrices instead of the whole model. The rank is their size: higher learns
more and costs a little more memory; 8 to 64 is common [guess]. *In Preflight:* moves memory a
little; time isn't modelled for it yet.

#### LoRA alpha

Scales how strongly the LoRA add-ons count. What matters is alpha divided by the rank, so changing
the rank alone also changes the strength. Often set to twice the rank [guess: a common habit]. *In
Preflight:* changes no number.

#### LoRA dropout

Randomly drops part of what goes into the LoRA add-ons while training, to curb overfitting on small
datasets. 0.05 to 0.1 is typical [guess]. *In Preflight:* the numbers don't change, but they read
**Extrapolated** (at 0 the library drops the dropout altogether, which saves a little memory).

#### Model

Which model the job uses. For training, a bigger model needs more memory and time; for agents and
evals, it changes token prices and answer quality. *In Preflight:* for an agent, cost, time and the
runaway chance come from the tasks sampled on that model; a model the sample didn't run on shows
**Not measured here** until you sample it.

#### Layers

How many blocks the model stacks. More layers, more parameters, more memory and time. *In
Preflight:* moves memory; time isn't modelled for it yet.

#### Model width

The size of each layer's hidden vector (`d_model`). Parameters grow roughly with its square
[derived: each weight matrix is width by width], so it moves memory and time fast. *In Preflight:*
moves memory; time isn't modelled for it yet.

#### Attention heads

How many parallel attention "views" each layer has. It must divide the model width evenly. With
plain attention, more heads means more memory for the attention scores; with a fused kernel, hardly
any. *In Preflight:* moves memory a little; time isn't modelled for it yet.

### Training: the run's length and housekeeping

#### Training steps

How many weight updates the whole run makes; when it's set, it wins over epochs. Time and cost scale
with it almost exactly. If the cockpit says **Run length unknown**, set this (or **Epochs**) so it
can. *In Preflight:* moves time and cost in proportion.

#### Epochs

How many times the run goes through the whole dataset. Another way to set the run's length. *In
Preflight:* moves time and cost in proportion, through the number of steps.

#### Eval interval

How often (in steps) the run stops to evaluate. More often tells you more and costs time. *In
Preflight:* time goes down as you raise it, by the probes' measured eval time per pass; memory
doesn't move (the eval peak comes from the eval batch, not how often it runs).

#### Save every

How often the run writes a checkpoint to disk. More often loses less to a crash and spends more time
writing. *In Preflight:* time goes down as you raise it, by the probes' measured save time (if no
probe saved, the reason says the save time isn't included).

#### Log interval

How often the run prints or logs its loss. Barely changes anything, unless it's every step on a fast
loop. *In Preflight:* time isn't modelled for it (logging is inside the measured step).

### Data

#### Packing

Joins several short examples into one full-length sequence so no tokens are wasted on padding. Often
a big speed-up on datasets of short texts; it changes which examples share a sequence. *In
Preflight:* time and cost aren't modelled for it yet (held, with a wider range); if it differs from
what the probes ran and the dataset's size isn't known, time shows **Not measured here**.

#### Data workers

How many background processes load and prepare batches. Too few and the GPU waits for data; the map
shows that as time spent in data loading. *In Preflight:* time isn't modelled for it yet.

### Where it runs

#### GPU

Which card the run uses. Each has its own memory, speed and price per hour. *In Preflight:* the knob
lists the 11 cards Preflight can rent [cited: the GPUs Modal offers] (or only the ones your limits
allow). Memory is worked out against the new card's size. Time on a card the probes didn't run on
comes from the cross-GPU model: **Extrapolated**, labelled rough, its range about 1.55 to 1.90 times
either way [derived: the engine's held-out cross-GPU bands]. Cost is that time at the card's hourly
price.

#### GPUs

How many cards the run spreads over. More cards finish sooner if the job splits well, and cost more
per hour. *In Preflight:* moves memory per card (down if the script shards the model, up a little
under plain data parallel, unchanged if the script sets no strategy), time per step (up, for the
cards talking to each other) and the number of steps (down); always **Extrapolated**, because the
probes ran on one card and the talking is estimated.

#### FSDP mode

How a multi-GPU run splits the model across cards (Fully Sharded Data Parallel). Sharding more saves
memory per card and adds talking between cards. *In Preflight:* moves memory per card with more than
one GPU; time isn't modelled for it yet.

#### Tensor parallel

Splits each layer's maths across several GPUs; for models too big for one card, mostly in serving.
*In Preflight:* moves memory per card down; time isn't modelled for it yet.

### Reinforcement learning (GRPO and friends)

#### Group size

How many answers the model writes for each prompt; GRPO compares them to each other to decide which
to reinforce. Bigger groups give a clearer signal and cost more generation time. *In Preflight:*
moves memory and rollout time up, and lowers the share of prompts that teach nothing (all answers
scoring the same).

#### Max prompt length

The longest prompt kept, in tokens. Longer prompts cost memory and time. *In Preflight:* moves
memory and time per step.

#### Max completion length

The longest answer the model may write while training. In RL most of the time goes to writing
answers, so this moves time and cost a lot. *In Preflight:* moves memory and rollout time up, and
lowers the share of answers cut off at the cap.

#### KL coefficient

How hard the run is pulled back towards the original model, so it doesn't drift into nonsense that
games the reward. Higher is safer and learns slower. *In Preflight:* the numbers don't change, but
they read **Extrapolated** (at 0 no reference model is loaded, which would save memory; that isn't
modelled yet).

#### Clip range (low)

#### Clip range (high)

How far one update may move the model's choices, below and above. Tighter ranges are steadier and
slower to learn. *In Preflight:* changes no number.

#### Loss type

Which variant of the RL loss the trainer uses. Changes how the run learns, not its memory. *In
Preflight:* the numbers don't change, but they read **Extrapolated** (some variants change how long
answers run).

#### vLLM rollouts

Writes the training answers with vLLM, a fast generation engine, instead of the training model's own
generate. Usually much faster, and it uses extra GPU memory. *In Preflight:* moves memory when a GPU
memory share is set (vLLM's reserve replaces the KV cache in the peak); time isn't modelled for it
yet.

#### GPU memory share

The fraction of the card vLLM is allowed to take. Too high leaves training no room; too low makes
generation slower. *In Preflight:* moves memory when vLLM shares the training card.

### Agents and API jobs

#### Tasks

How many tasks the whole job runs. Cost and time scale with it. *In Preflight:* scaled from the
sampled tasks.

#### Concurrency

How many tasks run at once. Higher finishes sooner, up to the API's rate limit. *In Preflight:* time
goes down with it until the API's tokens- or requests-per-minute limit binds.

#### Max turns

The most back-and-forth rounds one task may take before it's stopped. Caps runaway agents and their
bills. *In Preflight:* lowering it cuts the tokens, cost and time of the tasks it stops (they end
unfinished, and the reason counts them); raising it past a cap the sample hit leaves what those
tasks would spend unknown, and the reason says so.

#### Context limit

The most tokens the agent keeps in its conversation. Longer remembers more and pays for every token
on every turn. *In Preflight:* moves the **Context window filled** readout (how full the fullest
tasks get); if a sampled task would overflow, it says what share would. Cost and the runaway chance
aren't modelled for it.

#### Prompt caching

Reuses the unchanged start of a prompt between calls, which the API bills for less. Usually a saving
on agents that resend a long system prompt (some APIs charge a little extra to write the cache). *In
Preflight:* moved off what the sample ran, cost shows **Not measured here** until you sample with
caching on.

#### Tool retry cap

How many times a failing tool call is retried before the task gives up on it. *In Preflight:* retry
loops are counted in the sample, not simulated, so the numbers don't move yet.

#### Reasoning effort

How long the model thinks before it answers. More effort, more tokens, more cost, often better
answers. *In Preflight:* not read yet: a move changes no number, and the site doesn't say so yet.
Sample the tasks again at the effort you want.

#### Temperature

How random the model's word choice is: 0 gives nearly the same answer every time, higher is more
varied. It mostly changes quality, but it can change how long answers are, and so the cost. *In
Preflight:* the numbers don't change, but they read **Extrapolated**; a new probe or sample measures
it.

#### Max new tokens

The longest answer the model may write. It caps cost and time per call; too low cuts answers off.
*In Preflight:* moves memory when the script sets no sequence length (the model's notes on each
answer grow with it); time isn't modelled for it yet.

### Serving

Preflight's serving model is off on the public site, so most of these change only a status today.

#### Max model length

The longest prompt plus answer the server accepts. Longer needs more memory for the KV cache (the
model's notes on everything said so far). *In Preflight:* moves memory and time when the script sets
no sequence length.

#### Sequences in flight

How many requests the server works on at once. More is more throughput and more KV-cache memory. *In
Preflight:* no number moves while the serving model is off.

#### Prefix caching

Keeps the KV cache of shared prompt starts, so repeated system prompts aren't recomputed. *In
Preflight:* not modelled yet.

#### KV-cache dtype

The number format for the KV cache. 8-bit fits about twice the conversations in the same memory
[derived: half the bytes of 16-bit], with a small quality cost. *In Preflight:* fp8 halves the KV
cache's memory; time isn't modelled for it yet.

### Retrieval

#### Encode batch

How many texts the encoder turns into vectors at once. Bigger is faster until memory runs out. *In
Preflight:* moves memory and time per step, when the script has no other batch size.

#### HNSW links (M)

For a graph search index: how many neighbours each vector links to. More links, better recall, more
memory and slower building. *In Preflight:* not read yet; memory and time hold.

#### IVF lists

For a clustered search index: how many clusters the vectors are split into. More lists make each
search faster but can miss true neighbours unless you also search more lists at once (`nprobe`), and
the index needs more data to train. *In Preflight:* not read yet; time holds.

#### Dump hidden states

Saves the model's in-between vectors to disk. Useful for analysis; costs disk and time. *In
Preflight:* not modelled yet.

### Graph neural networks

#### Hidden width

The size of each node's vector inside the network. Wider learns more and uses more memory. *In
Preflight:* not read yet: the numbers hold at the probes' value, labelled **Extrapolated**.

#### GNN layers

How many hops of neighbours each node hears from. Each layer adds a hop, and memory grows fast. *In
Preflight:* not read yet, as for hidden width.

#### Neighbours at hop 1

#### Neighbours at hop 2

#### Neighbours at hop 3

#### Neighbours at hop 4

How many neighbours are sampled at each hop when building a batch. The batch's size multiplies hop
by hop, so these move memory more than almost anything else in a GNN. *In Preflight:* move memory,
time and cost up: their product sits on the job's size axis.

### Vision

#### Image size

The width and height pictures are resized to. For CNNs, memory and time grow with its square (vision
transformers grow faster): going from 224 to 448 pixels is about four times the work [derived: twice
the side, four times the pixels]. *In Preflight:* moves memory and time per step, with batch times
size squared on the size axis.
