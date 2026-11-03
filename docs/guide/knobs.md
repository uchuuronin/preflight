# How to fill this in: every input and knob, explained

Preflight is a flight simulator for ML jobs. You hand it your code, it flies a few minutes of the
real thing on a real GPU, and then it lets you fly the whole run in the cockpit before you pay for
it. This page is the pilot's manual: every box on **New flight**, then every knob in the cockpit, in
plain words. You don't need to have trained a model before. Where a word is jargon, the page says
what it means the first time it comes up.

Two words used everywhere below:

- **A probe** is a short, real run of your code: a few training steps (or a few tasks, for an agent)
  on a real GPU, timed and measured. Preflight runs a handful of them when you check in.
- **The guide** is the co-pilot in the cockpit: press **Start the tour**. It walks you through the
  knobs that matter for your aim, one move at a time, and never moves anything you've pinned or
  ruled out.

## What to write in "What are you trying to do?"

This box is your job in your own words: what it's for, what it runs on, what "done" means, and what
you can't change. A good answer is one or two sentences with the numbers you actually know.

**What happens to it today.** Preflight saves your words with the job's record, so they travel with
it, but the guide doesn't read them from there yet, whatever the box's label
says. The guide steers by three things you set, which is why each example below says what to pick
and type:

- **What matters most** on New flight: the guide's aim (lowest cost, finish soonest or most
  headroom).
- **Your limits** in the cockpit: type your hard limits in plain words ("must stay on an A100, under
  $40, done in 6 hours"), press **Read my limits**, and Muse Spark turns them into limits you
  confirm. If your check-in named GPUs, a budget or a deadline, they're waiting there marked "From
  this job's check-in": confirm the ones you mean. The guide never suggests anything outside a
  confirmed limit.
- **The guide's own box:** press **Start the tour**, paste your description into **What are you
  doing, and what matters most?** and press **Ask the guide**. Muse Spark reads it and plans its
  moves around it.

So write your description here, then, once the cockpit opens, paste it into the guide's box and put
the hard numbers in **Your limits**. (If the cockpit says the live guide is off, the rule-based tour
runs instead: it follows **What matters most** and your limits, not your words.)

### Examples, and what the guide does with each

**A finished script you want to run for real.**

> Fine-tune Qwen2.5-1.5B with LoRA on our support tickets, one A100, done by Friday, cheapest
> possible.

Pick **Lowest cost**. In the cockpit's limits, write "only A100, done within 72 hours". The guide
first checks the run fits in memory (a predicted out-of-memory is always fixed first), then ranks
the knobs by how much they cut cost on an A100 and proposes one move at a time: say, a bigger batch
with fewer accumulation steps. For each one you press **Apply**, **Not this**, **Keep** your value,
or **Pin** the suggestion.

**No training loop yet: you're planning.**

> I want to pretrain a small GPT on TinyStories but haven't written the loop yet. What GPU and batch
> size should I budget for?

Preflight measures real code, so a plan with no code has nothing to probe yet. Open the sample
closest to your plan instead (here **Pretrain a GPT-2 (124M) on TinyStories, measured on an A10 (2
Nov)**) and move its knobs: model width, layers, sequence length, batch size, GPU. The readouts
answer from that sample's measured probes. When your loop exists, check it in for numbers measured
on your code.

**No GPU chosen yet.**

> Train a ResNet-50 on 20,000 leaf photos. I don't know which GPU to rent; I just want it under $10.

Leave **Your GPU** on **L4 or A10** (or press **Let Preflight pick**), and pick **Lowest cost**. In
the cockpit, write "under $10" in the limits. Then drag the **GPU** knob: memory, time and cost move
for each card, and the guide only suggests cards that fit and stay under $10. A card the probes
never ran on is predicted, not measured, and its band is wider; the readout says so. (Some samples
measured on one card offer only that card for now.)

**Just "will it fit?"**

> Will a 7B LoRA fine-tune with 4,096-token sequences fit on a 24 GB card?

Pick **Most headroom**. Check in the script with the GPU you mean (the default L4 and A10 are both
24 GB cards [cited: NVIDIA's spec pages], but when neither is free the job moves to a 48 GB L40S:
check which card the memory readout names). The memory readout shows the predicted peak against the
card's memory, with its range, and **Chance it breaks** gives the odds of running out of memory. If
it doesn't fit, the guide's first move is the one that makes it fit: 4-bit weights (a 7B model's
weights alone take about 14 GB in 16-bit [derived: 7 billion x 2 bytes]), checkpointing, a smaller
eval batch, and shorter sequences only if nothing else works.

**An agent or API job.**

> An agent that answers support questions with tool calls on Muse Spark, 500 tasks, under $20.

Check in the agent's script. Its probes run a few real tasks through Preflight's proxy, which counts
tokens, tool calls and turns. Pick **Lowest cost** and write "under $20" in the limits. The knobs
here are about calls, not GPUs: the model, concurrency, max turns, prompt caching, the tool retry
cap. Cost is mostly tokens, so the guide looks first at the knobs that cut tokens.

**An evaluation.**

> Score pythia-160m on ARC-Easy with lm-evaluation-harness, 100 questions, as fast as possible.

Use the **Run a command** tab: `lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m
--tasks arc_easy --limit 100`. Pick **Finish soonest**. The probes time a few batches; the cockpit
scales them to your full run. The knobs that matter are the eval batch and, for generated answers,
max new tokens.

**Something the boxes don't cover.** Write it anyway ("must reuse our cluster's checkpoints", "the
data can't leave our bucket"). If you type it into **Your limits** and Preflight has no knob or
number for it, it says so plainly ("Not turned into a limit: ...") rather than guessing.

## The check-in form

### Where your job lives

Four ways in, as tabs:

- **Repo link:** a public GitHub repository. The branch (or the default one) is pinned to its commit
  when you check in, so later changes to the repo don't change your record.
- **Upload:** `.py` files plus requirements, configs and small text data, 5 MB in all. Weights and
  big datasets come from Hugging Face, not the upload. Pick which file to run.
- **Notebook:** a single notebook, checked in on its own. Its `pip` lines become its requirements,
  and the tab shows what will run.
- **Run a command:** a command Preflight knows by name, such as `lm_eval ...` for
  lm-evaluation-harness. It runs as written without a shell, so no pipes (`|`), redirects (`>`),
  variables or second commands. The box lists the commands it runs and says which word it can't.

### Extra packages

PyPI names your code imports that might not be installed, separated by commas (`einops, timm`).
Preflight installs them before the run. A name it can't install stops the check-in before any GPU
time is spent, so a typo costs you nothing.

### What matters most

The guide's aim, and the readout it treats as your goal:

- **Lowest cost:** fewest dollars for the whole run (GPU time and tokens).
- **Finish soonest:** shortest wall-clock time.
- **Most headroom:** the most room between the memory peak and the card's memory.

You can change it in the cockpit later: press a dial to make it your goal.

### Your GPU

**L4 or A10** starts on an L4; if none is free within two minutes it moves to an A10, then an L40S,
and a script that runs out of memory may be measured on a bigger card. You can also pick an L40S,
**A100-SXM-80GB**, **H100-SXM-80GB** or **MI300X** (AMD), or press **Let Preflight pick**. Pick the
card you'll really use if you know it; the probes' measured numbers are worth more than any
prediction.

### Private triage

Tick it and your probes' logs are read by Muse Glimmer, Meta's open-weights model, on Preflight's
own GPU, so Meta and Baseten never see them. Muse Spark still reads your code to draw the map, and
the check of your plots against your numbers is skipped. Glimmer naps when idle: waking it took 5 to
8 minutes on 1 Nov [measured], then a triage takes seconds.

### Check in

Free, with no account: Preflight pays for the GPU minutes, up to 3 jobs a day (a check-in, or a
probe or check from the cockpit). A measured run took under 7 minutes on an L4 (28 Oct) [measured].
If a check-in stops before measuring anything, one resend a day is free [cited: the queue's
`FREE_RESENDS_PER_DAY`].

## Reading the cockpit

The middle of the map is **memory**: the predicted peak against the card's memory, marked **Fits**,
**Tight** or **Fails**. The dials beside it are **Time to finish**, **Run cost** (GPU and tokens)
and the chance the run breaks: runs out of memory for training, runs away for an agent. Each carries
a range, not just a number, and a word for where it came from:

- **Measured:** a probe ran it.
- **Predicted:** a model fitted to the probes, inside the settings they covered.
- **Extrapolated:** beyond what the probes measured. Trust it less.
- **Not measured here:** Preflight won't guess. Run a probe at that setting to get a number.

**Tight: measure first** means the prediction sits close to the card's limit: run a probe before you
trust it. A record marked **Example numbers** shows how the cockpit works; nothing in it was
measured.

## How knobs behave

### Moving a knob

Drag it or pick a value and the map, the readouts and the timeline answer straight away. Each knob's
hint says what it moves first: memory, speed and cost, how batches are built, the model, where it
runs, the RL loop or the agent's calls.

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
here by the kind of job they usually belong to.

### Training: memory first

#### Batch size

How many examples the model sees in one step. Bigger is usually faster per example and needs more
memory. The first knob to look at when a run doesn't fit, and the first to raise when it fits with
room to spare. It also changes how the model learns, unless gradient accumulation keeps batch times
accumulation the same; and with a fixed number of steps, a smaller batch means fewer examples seen
in all.

#### Gradient accumulation

Runs several small batches before each weight update and averages their gradients, so batch 8 with
accumulation 4 learns like batch 32 but needs batch 8's memory. Costs a little speed.

#### Eval batch

The batch size for the evaluation pass, separate from training. A classic trap: training fits but
the first evaluation runs out of memory because its batch is bigger. Lower it if the memory readout
spikes at eval.

#### Eval without gradients

Runs evaluation without keeping what training needs to learn (`torch.no_grad()`), so it uses far
less memory. Almost always on; if your code forgets it, this knob is a free win.

#### Sequence length

How many tokens one example holds. Memory and time grow with it (attention grows faster than
linearly). Shorten it only if nothing else makes the run fit, because it changes what the model can
learn from long examples.

#### Checkpointing

Gradient checkpointing: forgets some in-between results during the forward pass and recomputes them
when learning. Saves a lot of memory for roughly a third more time [guess: the usual rule of thumb,
one extra forward pass]. Not to be confused with saving the model to disk (that's **Save every**).

#### Precision

The number format for the maths: 32-bit is precise and big, `bf16` and `fp16` are half the size
[derived: 16 bits against 32] and much faster on modern GPUs. `bf16` is the safe default on cards
that support it (A10, L4, A100, H100 [cited: NVIDIA's spec pages]); `fp16` has a narrow range, so
small gradients can round to zero unless loss scaling is on.

#### Weight quantisation

Stores the frozen base model's weights in 8-bit or 4-bit NF4 instead of 16-bit, cutting their memory
by about half or three quarters [derived: 8 or 4 bits against 16]. Often used with LoRA: LoRA on
4-bit weights is called QLoRA. A little slower per step.

#### Attention kernel

Which implementation computes attention. Fused kernels such as FlashAttention or PyTorch's SDPA are
faster and use less memory than the plain version, on cards that support them.

#### Channels-last layout

For image models: stores pictures as height, width, channels instead of channels first, which lets
tensor cores work faster with mixed precision. Usually a free speed-up for CNNs.

### Training: what the model learns

#### Learning rate

How big a step the model takes each update. Too high and the loss jumps around or blows up (NaN);
too low and it barely moves. It doesn't change memory or speed, only whether training works.

#### Learning rate schedule

How the learning rate changes over the run: constant, linear decay, cosine. Cosine with a short
warm-up is the common default.

#### Warm-up steps

Steps at the start where the learning rate ramps up from near zero, so the first noisy updates don't
knock the model off course.

#### Optimiser

The rule that turns gradients into weight updates. AdamW is the default; it keeps two extra numbers
per weight [cited: AdamW's two moving averages], which is a lot of memory for a big model. 8-bit and
factored optimisers keep less.

#### Weight decay

A gentle pull of the weights towards zero each step, which helps the model generalise. Small values
are typical (0.01 to 0.1 [guess: common defaults]). No effect on memory or speed.

#### Gradient clip

Scales the gradient down when its overall size passes a cap, which stops a single bad batch from
blowing the run up. Commonly 1.0 [guess: a common default].

#### LoRA rank

LoRA trains small add-on matrices instead of the whole model. The rank is their size: higher learns
more and costs a little more memory; 8 to 64 is common [guess].

#### LoRA alpha

Scales how strongly the LoRA add-ons count. What matters is alpha divided by the rank, so changing
the rank alone also changes the strength. Often set to twice the rank [guess: a common habit].

#### LoRA dropout

Randomly drops part of what goes into the LoRA add-ons while training, to curb overfitting on small
datasets. 0.05 to 0.1 is typical [guess].

#### Model

Which model the job uses. For training, a bigger model needs more memory and time; for agents and
evals, it changes token prices and answer quality.

#### Layers

How many blocks the model stacks. More layers, more parameters, more memory and time.

#### Model width

The size of each layer's hidden vector (`d_model`). Parameters grow roughly with its square
[derived: each weight matrix is width by width], so it moves memory and time fast.

#### Attention heads

How many parallel attention "views" each layer has. It must divide the model width evenly. With
plain attention, more heads means more memory for the attention scores; with a fused kernel, hardly
any.

### Training: the run's length and housekeeping

#### Training steps

How many weight updates the whole run makes. Time and cost scale with it almost exactly. If the
cockpit says **Run length unknown**, set this (or **Epochs**) so it can.

#### Epochs

How many times the run goes through the whole dataset. Another way to set the run's length.

#### Eval interval

How often (in steps) the run stops to evaluate. More often tells you more and costs time.

#### Save every

How often the run writes a checkpoint to disk. More often loses less to a crash and spends more time
writing.

#### Log interval

How often the run prints or logs its loss. Barely changes anything, unless it's every step on a fast
loop.

### Data

#### Packing

Joins several short examples into one full-length sequence so no tokens are wasted on padding. Often
a big speed-up on datasets of short texts; it changes which examples share a sequence.

#### Data workers

How many background processes load and prepare batches. Too few and the GPU waits for data; the map
shows that as time spent in data loading.

### Where it runs

#### GPU

Which card the run uses. Each has its own memory, speed and price per hour. The card the probes ran
on is measured; others are predicted from it, with a wider range.

#### GPUs

How many cards the run spreads over. More cards finish sooner if the job splits well, and cost more
per hour.

#### FSDP mode

How a multi-GPU run splits the model across cards (Fully Sharded Data Parallel). Sharding more saves
memory per card and adds talking between cards.

#### Tensor parallel

Splits each layer's maths across several GPUs; for models too big for one card, mostly in serving.

### Reinforcement learning (GRPO and friends)

#### Group size

How many answers the model writes for each prompt; GRPO compares them to each other to decide which
to reinforce. Bigger groups give a clearer signal and cost more generation time.

#### Max prompt length

The longest prompt kept, in tokens. Longer prompts cost memory and time.

#### Max completion length

The longest answer the model may write while training. In RL most of the time goes to writing
answers, so this moves time and cost a lot.

#### KL coefficient

How hard the run is pulled back towards the original model, so it doesn't drift into nonsense that
games the reward. Higher is safer and learns slower.

#### Clip range (low)

#### Clip range (high)

How far one update may move the model's choices, below and above. Tighter ranges are steadier and
slower to learn.

#### Loss type

Which variant of the RL loss the trainer uses. Changes how the run learns, not its memory.

#### vLLM rollouts

Writes the training answers with vLLM, a fast generation engine, instead of the training model's own
generate. Usually much faster, and it uses extra GPU memory.

#### GPU memory share

The fraction of the card vLLM is allowed to take. Too high leaves training no room; too low makes
generation slower.

### Agents and API jobs

#### Tasks

How many tasks the whole job runs. Cost and time scale with it.

#### Concurrency

How many tasks run at once. Higher finishes sooner, up to the API's rate limit.

#### Max turns

The most back-and-forth rounds one task may take before it's stopped. Caps runaway agents and their
bills.

#### Context limit

The most tokens the agent keeps in its conversation. Longer remembers more and pays for every token
on every turn.

#### Prompt caching

Reuses the unchanged start of a prompt between calls, which the API bills for less. Usually a saving
on agents that resend a long system prompt (some APIs charge a little extra to write the cache).

#### Tool retry cap

How many times a failing tool call is retried before the task gives up on it.

#### Reasoning effort

How long the model thinks before it answers. More effort, more tokens, more cost, often better
answers.

#### Temperature

How random the model's word choice is: 0 gives nearly the same answer every time, higher is more
varied. It mostly changes quality, but it can change how long answers are, and so the cost.

#### Max new tokens

The longest answer the model may write. It caps cost and time per call; too low cuts answers off.

### Serving

#### Max model length

The longest prompt plus answer the server accepts. Longer needs more memory for the KV cache (the
model's notes on everything said so far).

#### Sequences in flight

How many requests the server works on at once. More is more throughput and more KV-cache memory.

#### Prefix caching

Keeps the KV cache of shared prompt starts, so repeated system prompts aren't recomputed.

#### KV-cache dtype

The number format for the KV cache. 8-bit fits about twice the conversations in the same memory
[derived: half the bytes of 16-bit], with a small quality cost.

### Retrieval

#### Encode batch

How many texts the encoder turns into vectors at once. Bigger is faster until memory runs out.

#### HNSW links (M)

For a graph search index: how many neighbours each vector links to. More links, better recall, more
memory and slower building.

#### IVF lists

For a clustered search index: how many clusters the vectors are split into. More lists make each
search faster but can miss true neighbours unless you also search more lists at once (`nprobe`), and
the index needs more data to train.

#### Dump hidden states

Saves the model's in-between vectors to disk. Useful for analysis; costs disk and time.

### Graph neural networks

#### Hidden width

The size of each node's vector inside the network. Wider learns more and uses more memory.

#### GNN layers

How many hops of neighbours each node hears from. Each layer adds a hop, and memory grows fast.

#### Neighbours at hop 1

#### Neighbours at hop 2

#### Neighbours at hop 3

#### Neighbours at hop 4

How many neighbours are sampled at each hop when building a batch. The batch's size multiplies hop
by hop, so these move memory more than almost anything else in a GNN.

### Vision

#### Image size

The width and height pictures are resized to. For CNNs, memory and time grow with its square (vision
transformers grow faster): going from 224 to 448 pixels is about four times the work [derived: twice
the side, four times the pixels].