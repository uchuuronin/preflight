# How to fill this in: every input and knob, explained

Preflight runs a few minutes of your job on a real GPU, then lets you try the whole run in the
cockpit before you pay for it. This page covers every box on **New flight** and every knob in the
cockpit. You don't need to have trained a model before. For the engine's line-by-line account of
each knob, see [`knobs-reference.md`](knobs-reference.md).

Two terms used throughout:

- **Probe:** a short, real run of your code (a few training steps, or a few tasks for an agent) on a
  real GPU, timed and measured. A check-in runs a handful.
- **The guide:** the co-pilot in the cockpit. Press **Start the tour** and it suggests one move at a
  time for your aim. It never touches a knob you've pinned or ruled out.

## What to write in "What are you trying to do?"

Describe the job in one or two sentences: what it's for, what it runs on, when it must finish, and
what can't change. Use the numbers you know.

**Where it goes.** Preflight saves it with the record. When you press **Start the tour**, it's
already in the guide's box (**What are you doing, and what matters most?**). Press **Ask the
guide** and Muse Spark plans around it.

Two other settings steer the guide:

- **What matters most** on New flight sets its aim: lowest cost, finish soonest or most headroom.
- **Your limits** in the cockpit holds hard rules. Type them ("only A100, under $40, done in 6
  hours"), press **Read my limits**, and confirm what Muse Spark reads. The guide never suggests
  anything outside a confirmed limit.

If the cockpit says the live guide is off, the rule-based tour runs instead. It follows **What
matters most** and your limits, not your description.

### Examples, and what the guide does with each

**A finished script you want to run for real.**

> Fine-tune Qwen2.5-1.5B with LoRA on our support tickets, one A100, done by Friday, cheapest
> possible.

Pick **Lowest cost**. In **Your limits**, write "only A100, done within 72 hours". The guide fixes
memory first if the run would run out, then suggests moves that cut cost, one at a time. For each,
press **Apply**, **Not this**, **Keep** (your value) or **Pin** (its value). Probes on the public
site run on cards up to an L40S, so A100 time and cost are estimates from another card: they read
**Extrapolated**, and the true value can be about 1.6 to 1.9 times higher or lower. A100 memory is
recalculated from your probes for the bigger card, and also reads **Extrapolated**.

**No training loop yet.**

> I want to pretrain a small GPT on TinyStories but haven't written the loop yet. What GPU and batch
> size should I budget for?

Preflight measures code, so there's nothing to probe yet. Open the closest sample instead (**Pretrain
a GPT-2 (124M) on TinyStories, measured on an A10 (2 Nov)**) and move its knobs. Batch size,
sequence length and precision change memory, time and cost (precision reads **Extrapolated**);
training steps change time and cost. Layers and model width change memory only; time stays at the
probes' value and is marked "isn't modelled for time yet". This sample offers only the A10, so it
can't compare GPUs: check in your loop once it exists, then use the **GPU** knob.

**No GPU chosen yet.**

> Train a ResNet-50 on 20,000 leaf photos. I don't know which GPU to rent; I just want it under $10.

Leave **Your GPU** on **L4 or A10**, or press **Let Preflight pick**. Pick **Lowest cost** and add
"under $10" to **Your limits**. In the cockpit, open the **GPU** knob to see memory, time and cost
for each card. The guide only suggests cards that fit and stay under $10. Cards the probes didn't
run on read **Extrapolated**, with a wide range. A sample measured on one card may offer only that
card.

**Just "will it fit?"**

> Will a 7B LoRA fine-tune with 4,096-token sequences fit on a 24 GB card?

Pick **Most headroom** and check in with the card you mean. The L4 and A10 both have 24 GB; if
neither is free, the job moves to a 48 GB L40S, so check which card the memory readout names. The
readout shows the predicted peak against the card's memory and the chance of running out. If it
doesn't fit, the guide's first move fixes that: 4-bit weights (a 7B model's weights take about 14 GB
in 16-bit), then checkpointing, a smaller eval batch, and shorter sequences only as a last resort.

**An agent or API job.**

> An agent that answers support questions with tool calls on Muse Spark, 500 tasks, under $20.

Check in the agent's script. Preflight runs 20 to 50 real tasks and scales their tokens, cost and
time to your 500, with a wider range for runaway tasks the sample may have missed. Pick **Lowest
cost** and add "under $20" to **Your limits**. Use **Max turns** to stop runaway tasks,
**Concurrency** to finish sooner, and **Tasks** to scale the job. A **Model** the sample didn't use
can't be priced until you sample it.

**An evaluation.**

> Score pythia-160m on ARC-Easy with lm-evaluation-harness, 100 questions, as fast as possible.

Use the **Run a command** tab:
`lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks arc_easy --limit 100`.
Pick **Finish soonest**. The probes measure memory and step time. Preflight's model of evaluation
passes is off on the public site, so the finish time may show **Not measured here**.

**Something the boxes don't cover.** Write it anyway ("must reuse our cluster's checkpoints"). If
you type it into **Your limits** and Preflight has no knob for it, it says "Not turned into a limit"
instead of guessing.

## The check-in form

### Where your job lives

Pick one tab:

- **Repo link:** a public GitHub repository. The branch (or the default one) is pinned to its
  current commit, so later pushes don't change your record. Two optional fields say what to run
  (see **Entry point and arguments** below).
- **Upload:** `.py` files plus requirements, configs and small text data. Up to 5 MB and 2,000
  files. Weights and large datasets come from Hugging Face. Choose which file to run.
- **Notebook:** one notebook on its own. Its `pip` lines become its requirements.
- **Run a command:** a tool Preflight knows by name (today, `lm_eval`), up to 64 words and 8,192
  characters. There's no
  shell, so no pipes (`|`), redirects (`>`), variables or chained commands. The box names any word
  it can't take.

### Entry point and arguments

For a **Repo link**, two optional fields under the repository:

- **Entry point:** the script to run, as its path in the repo (`examples/train.py`). Leave it empty
  and Preflight picks `train.py`, `main.py`, `run.py`, `finetune.py` or `pretrain.py` at the top of
  the repo, or the only `.py` file. Fill it in whenever the repo holds more than one script, and
  always for a large repo (over 2,000 files or 25 MB of text), where Preflight doesn't guess.
- **Arguments:** the flags you'd type after `python <entry point>`, such as `--epochs 6 --lr 3e-4`.
  They run as written, without a shell: no pipes (`|`), redirects (`>`), variables or chained
  commands. Quote an argument that has a space in it. Up to 64 arguments.

Example: [pytorch/examples](https://github.com/pytorch/examples) holds dozens of scripts. To check in
its word-level language model:

- Repo link: `github.com/pytorch/examples`
- Entry point: `word_language_model/main.py`
- Arguments: `--accel --epochs 6 --data word_language_model/data/wikitext-2`

Preflight reads the repo (for a large one, the entry point and what it uses: the repo's modules it
imports, its requirements and files named in the arguments), then probes the script with those
flags. If the entry point isn't in the repo, the check-in stops before any GPU time, but it still
uses one of the day's check-ins, so check the path first.

Limits to know:

- **Paths from the top folder.** The script runs from the repo's top folder, so give data paths from
  there: `--data word_language_model/data/wikitext-2`, not `--data ./data/wikitext-2`.
- **One script, run with `python`.** The entry point must do its work when run as
  `python <entry point>`. A file that only defines functions, or a module meant to be imported, runs
  nothing; Preflight stops and says so.
- **No two-step flows.** "Download, then train" in two commands isn't supported: the script has to
  fetch what it needs itself, or read it from Hugging Face.
- **No tools launched by name,** such as `litgpt pretrain` or `tune run`. Only `lm_eval` runs by name,
  on the **Run a command** tab.
- **Notebooks in a repo** go through the **Notebook** tab (download the `.ipynb` and add it there).
  That check-in isn't pinned to a commit.

### Extra packages

PyPI names your code imports that may not be installed, separated by commas (`einops, timm`), up to
10. Names only, no versions. Before the run, Preflight installs the names on its list from ready-built wheels.
Any other name stops the check-in before GPU time is spent, so a typo costs nothing.

### What matters most

The guide's aim:

- **Lowest cost:** fewest dollars for the whole run, GPU time and tokens.
- **Finish soonest:** shortest wall-clock time.
- **Most headroom:** the most room between peak memory and the card's memory.

To change it later, press a dial in the cockpit.

### Your GPU

**L4 or A10** starts on an L4. If none is free within two minutes, it moves to an A10, then an L40S.
A script that runs out of memory may be measured on a bigger card. From Nov 5 to 8 the default starts
on an A10.

You can also pick **L40S**, **A100-SXM-80GB** or **H100-SXM-80GB**, or press **Let Preflight
pick** for the cheapest card with room to spare. Probes on the public site run on cards up to an
L40S: if you pick an A100 or H100, the probes run on a smaller card and its numbers are converted.
If you know the card you'll use, pick it.

### Private triage

Tick it and Muse Glimmer, Meta's open-weights model on Preflight's own GPU, reads your failed
probes' logs, so Meta and Baseten never see them. Muse Spark still reads your code to draw the map,
and the check of your plots against your numbers is skipped. Glimmer sleeps when idle: waking it
took 5 to 8 minutes on Nov 1; after that, a triage takes seconds.

### Check in

Free, no account. Limits:

- 3 jobs a day (a check-in, or a probe or check from the cockpit), and 3 queued or running at once.
  Days run midnight to midnight, US Eastern.
- Past your 3 jobs, or once the site's daily GPU budget is spent, only samples and shared records
  open until midnight.
- If a check-in stops at a missing import before measuring anything, one resend a day is free.

A check-in of the LoRA sample on an L4 takes about 5 to 11 minutes (its two production check-ins
took 4 min 59 s and 10 min 21 s); other jobs vary. Add any wait for a free GPU: the boarding screen
shows it. If no GPU frees up within 20 minutes, the probe stops with "no GPU could be scheduled".

## Reading the cockpit

The centre of the map is **Peak memory**: the predicted peak against the card's memory, marked
**Fits**, **Tight** or **Fails**. Beside it are **Time to finish**, **Run cost** (GPU and tokens)
and the chance the run breaks: out of memory for training, runaway for an agent.

Each readout has a range the true value should fall inside 9 times in 10, and a status word:

- **Measured:** a probe ran at exactly these settings on this GPU. Time to finish and cost are never
  Measured: the step is measured, then multiplied out to the whole run.
- **Predicted:** worked out from nearby measured settings. Should hold.
- **Extrapolated:** worked out beyond what was measured (another GPU, an unprobed value, a knob the
  model doesn't cover yet). Rough; the reason says why.
- **Not measured here:** nothing measured is close enough, so no number is shown. Move back towards
  a measured setting, or run a probe there.

Out-of-memory chance: **Fits** is under 1%, **Tight** 1% to 80%, **Fails** 80% or more. These
cut-offs haven't been checked against real runs yet. A chance between 5% and 95% shows as a range,
because chances in that band haven't been verified either. **Tight: measure first** means run a probe before you trust it.

How much to trust the ranges: against real full runs, the 90% ranges held 10 times out of 18 (Nov
3). Treat a **Predicted** range as narrower than it should be.

## How knobs behave

### Moving a knob

Drag it or pick a value. The map, readouts and timeline update at once. Each knob's hint says what
it affects first, for example memory, speed and cost, or where it runs.

### Knobs that only change what the model learns

Learning rate, its schedule, warm-up, weight decay, gradient clip, LoRA alpha and the RL clip ranges
change how the model learns, not a step's memory or time. Moving them leaves memory, time and cost
as they were. Preflight doesn't predict learning quality.

### Isn't modelled for time yet

Some knobs affect time in ways Preflight doesn't model yet: model shape, LoRA rank, quantisation,
the optimiser, the attention kernel, data workers, packing and others (marked below). Move one off
its probed value and time stays at the probes' value, reads **Extrapolated** with "isn't modelled
for time yet", and its range widens by half. If the range would pass ±50%, the dial shows **Not
measured here** instead.

### Off the probed values

A knob at a value no probe ran turns memory and time from **Predicted** to **Extrapolated**. Not
covered by this rule: size knobs (batch, sequence length, image size, fan-outs), run length, save
and eval intervals, gradient accumulation (for time), the knobs that only change learning, and GPU
and GPU count (those read **Extrapolated** for their own reasons; see **GPU** and **GPUs**). The job's size axis is the
product of its size knobs (for a language model, batch × sequence length). Past twice the largest
probe on it, memory and time show **Not measured here**.

### Pinned

**Pin at ...** fixes a knob at its value; the guide won't move it. Unpin to release it.

### The guide suggests this

A knob the guide proposed shows the suggested value. Move it yourself and your value wins.

### Code

Knobs with a **Code** badge can only change by editing your script. Muse Code writes the change and
a short verify run checks it before you get it. Nothing runs until you say so, and each verify run
or probe counts toward your 3 jobs a day.

### Not predicted here yet

The value is outside what the probes covered. A probe at that setting would measure it.

### Your limits

What the guide must respect: allowed GPUs, a budget, a deadline, a GPU-hour quota, knobs not to
change. Type them and press **Read my limits**, or set them by hand. Nothing applies until you
confirm. A suggestion you turn down with **Not this** is ruled out too; you can allow it again.

## The knobs

The cockpit shows only the knobs your code sets. They're grouped here by job type. Each entry ends
with what Preflight does with it today.

### Training: memory first

#### Batch size

Examples per step. Bigger is usually faster per example and uses more memory. Lower it first when a
run doesn't fit; raise it when there's room. It also changes learning unless gradient accumulation
keeps batch × accumulation the same, and with fixed steps a smaller batch sees fewer examples in
total. *In Preflight:* moves memory, step time and step count; total time and cost follow their
product.

#### Gradient accumulation

Runs several small batches before each update, so batch 8 with accumulation 4 learns like batch 32
at about batch 8's memory. Slightly slower. *In Preflight:* on its own it adds up to one copy of the
trainable gradients (small for LoRA, large for full fine-tuning); the saving comes from lowering
batch size with it. Time per update rises and
updates per epoch fall by the same factor.

#### Eval batch

Batch size for evaluation, separate from training. A common trap: training fits, then the first
eval runs out of memory. Lower it if memory spikes at eval. *In Preflight:* moves memory when eval
sets the peak; time and its range don't change (each eval pass is counted at the probes' eval
batch).

#### Eval without gradients

Runs evaluation under `torch.no_grad()`, which uses far less memory. Almost always on; if your code
misses it, turning it on is free. *In Preflight:* moves memory; time isn't modelled yet.

#### Sequence length

Tokens per example; longer rows are cut. Memory and time grow with it, attention faster than the
rest. Shorten it only if nothing else fits, because the model loses long examples. *In Preflight:*
moves memory and time only while rows are longer than the limit. Past the length the dataset was
scanned at, time reads **Extrapolated**.

#### Checkpointing

Gradient checkpointing: drops some intermediate results and recomputes them during the backward
pass. Saves a lot of memory for about one extra forward pass of time. Not the same as saving the
model to disk (that's **Save every**). *In Preflight:* on lowers memory and raises time; turning it
off when the probes ran with it on keeps the recompute in the time, marked **Extrapolated**.

#### Precision

The number format. 32-bit is precise and large; `bf16` and `fp16` are half the size and much faster on
modern GPUs. Use `bf16` on cards that support it (A10, L4, A100, H100). `fp16` needs loss scaling or
small gradients round to zero. *In Preflight:* moves memory and time, **Extrapolated** when it
differs from the probes; a format the card can't run fast (bf16 on a T4) shows **Not measured
here**.

#### Weight quantisation

Stores the frozen base weights in 8-bit or 4-bit (NF4) instead of 16-bit, cutting their memory by
about half or three quarters. LoRA on 4-bit weights is QLoRA. Slightly slower per step. *In
Preflight:* moves memory; time isn't modelled yet.

#### Optimiser

Turns gradients into weight updates. AdamW, the default, keeps two extra numbers per weight, which
is a lot for a big model. Lion keeps one, plain SGD none, and 8-bit versions store each in a byte.
*In Preflight:* moves memory (an unknown optimiser counts as AdamW); time isn't modelled yet.

#### Attention kernel

How attention is computed. Fused kernels (FlashAttention, PyTorch SDPA), on cards that support them, are
faster and lighter than plain "eager" attention, whose memory grows with the square of sequence length. *In Preflight:*
moves memory; time isn't modelled yet.

#### Channels-last layout

For image models: stores pictures as height, width, channels, which speeds up mixed precision on
tensor cores. Usually a free speed-up for CNNs. *In Preflight:* not predicted; a verify run measures
it.

### Training: what the model learns

#### Learning rate

Step size per update. Too high and the loss jumps or turns NaN; too low and it barely moves. *In
Preflight:* changes no number.

#### Learning rate schedule

How the learning rate changes over the run: constant, linear decay or cosine. Cosine with a short
warm-up is the usual default. *In Preflight:* changes no number.

#### Warm-up steps

Steps at the start where the learning rate ramps up from near zero, so early noisy updates don't
derail the model. *In Preflight:* changes no number.

#### Weight decay

Pulls weights slightly towards zero each step, which helps the model generalise. Typical values:
0.01 to 0.1. *In Preflight:* changes no number.

#### Gradient clip

Scales the gradient down when it passes a cap, so one bad batch can't wreck the run. Commonly 1.0.
*In Preflight:* changes no number.

#### LoRA rank

LoRA trains small add-on matrices instead of the whole model; the rank is their size. Higher learns
more and uses a little more memory. 8 to 64 is common. *In Preflight:* moves memory slightly; time
isn't modelled yet.

#### LoRA alpha

How strongly the LoRA add-ons count. What matters is alpha divided by rank, so changing the rank
alone also changes strength. Often twice the rank. *In Preflight:* changes no number.

#### LoRA dropout

Randomly drops part of the LoRA input during training to curb overfitting on small datasets. 0.05
to 0.1 is typical. *In Preflight:* numbers don't change but read **Extrapolated** (at 0 the library
skips dropout and saves a little memory).

#### Model

Which model the job uses. For training, bigger means more memory and time; for agents and evals, it
changes token prices and answer quality. *In Preflight:* for an agent, numbers come from tasks
sampled on that model; an unsampled model shows **Not measured here**.

#### Layers

How many blocks the model stacks. More layers, more memory and time. *In Preflight:* moves memory;
time isn't modelled yet.

#### Model width

Size of each layer's hidden vector (`d_model`). Parameters grow with roughly its square. *In
Preflight:* moves memory; time isn't modelled yet.

#### Attention heads

Parallel attention "views" per layer; must divide model width evenly. With plain attention, more
heads use more memory; with a fused kernel, hardly any. *In Preflight:* moves memory slightly; time
isn't modelled yet.

### Training: the run's length and housekeeping

#### Training steps

Total weight updates; overrides epochs when set. Time and cost scale with it almost exactly. If the
cockpit says **Run length unknown**, set this or **Epochs**. *In Preflight:* moves time and cost in
proportion.

#### Epochs

Passes over the whole dataset; another way to set run length. *In Preflight:* moves time and cost in
proportion, through the step count.

#### Eval interval

Steps between evaluations. More often tells you more and costs time. *In Preflight:* raising it
lowers time by the probes' measured time per eval; memory doesn't change.

#### Save every

Steps between checkpoints written to disk. More often loses less to a crash and spends more time
writing. *In Preflight:* raising it lowers time by the probes' measured save time (if no probe
saved, the reason says so).

#### Log interval

Steps between loss logs. Rarely matters unless it's every step on a fast loop. *In Preflight:* time
isn't modelled (logging is inside the measured step).

### Data

#### Packing

Joins short examples into full-length sequences so no tokens go to padding. Often a big speed-up on
short texts. *In Preflight:* time and cost aren't modelled yet (held, with a wider range); if it
differs from the probes and the dataset size isn't known, time shows **Not measured here**.

#### Data workers

Background processes that load batches. Too few and the GPU waits for data, which the map shows as
time in data loading. *In Preflight:* time isn't modelled yet.

### Where it runs

#### GPU

The card the run uses; each has its own memory, speed and hourly price. *In Preflight:* lists the 11
cards Preflight can rent (or those your limits allow). Memory is recalculated for the card. Time on
a card the probes didn't use is converted from the probed card: **Extrapolated**, rough, with a
range of about 1.6 to 1.9 times either way. Cost is that time at the card's hourly price.

#### GPUs

How many cards the run uses. More finish sooner if the job splits well, and cost more per hour. *In
Preflight:* moves memory per card (down if the script shards the model, up by as much as one copy of the
trainable gradients under plain data parallel, unchanged with no strategy), step time (up, for communication) and step count (down).
Always **Extrapolated**: the probes ran on one card.

#### FSDP mode

How a multi-GPU run shards the model (Fully Sharded Data Parallel). More sharding saves memory per
card and adds communication. *In Preflight:* moves memory per card with more than one GPU; time
isn't modelled yet.

#### Tensor parallel

Splits each layer's maths across GPUs, mostly for serving models too big for one card. *In
Preflight:* lowers memory per card; time isn't modelled yet.

### Reinforcement learning (GRPO and friends)

#### Group size

Answers generated per prompt; GRPO compares them to decide what to reinforce. Bigger groups give a
clearer signal and take longer to generate. *In Preflight:* raises memory and rollout time, and
lowers the share of prompts where every answer scores the same (they teach nothing).

#### Max prompt length

Longest prompt kept, in tokens. *In Preflight:* moves memory and step time.

#### Max completion length

Longest answer the model may write in training. In RL most time goes to generating, so this moves
time and cost a lot. *In Preflight:* raises memory and rollout time, and lowers the share of answers
cut off at the cap.

#### KL coefficient

How hard the run is pulled back towards the original model so it doesn't drift into gaming the
reward. Higher is safer and learns slower. *In Preflight:* numbers don't change but read
**Extrapolated** (at 0 no reference model loads, which would save memory; not modelled yet).

#### Clip range (low)

#### Clip range (high)

How far one update may move the model's choices, below and above. Tighter is steadier and slower.
*In Preflight:* changes no number.

#### Loss type

Which RL loss variant the trainer uses. Changes learning, not memory. *In Preflight:* numbers don't
change but read **Extrapolated** (some variants change answer length).

#### vLLM rollouts

Generates training answers with vLLM, a fast generation engine, instead of the model's own
`generate`. Usually much faster, at the cost of extra GPU memory. *In Preflight:* moves memory when a
GPU memory share is set; time isn't modelled yet.

#### GPU memory share

The fraction of the card vLLM may take. Too high leaves training no room; too low slows generation.
*In Preflight:* moves memory when vLLM shares the training card.

### Agents and API jobs

#### Tasks

How many tasks the job runs. Cost and time scale with it. *In Preflight:* scaled from the sampled
tasks.

#### Concurrency

Tasks running at once. Higher finishes sooner, up to the API's rate limit. *In Preflight:* lowers
time until the API's tokens- or requests-per-minute limit binds.

#### Max turns

Most rounds one task may take before it's stopped. Caps runaway agents and their bills. *In
Preflight:* lowering it cuts tokens, cost and time of the tasks it stops (they end unfinished and
the reason counts them); raising it past a cap the sample hit leaves their cost unknown, and the
reason says so.

#### Context limit

Most tokens the agent keeps in its conversation. Longer remembers more and pays for every token on
every turn. *In Preflight:* moves the **Context window filled** readout. If any sampled task would
overflow, it shows no number and says what share would. Cost and runaway chance aren't modelled.

#### Prompt caching

Reuses the unchanged start of a prompt between calls, billed at a lower rate. Usually saves money
when an agent resends a long system prompt (some APIs charge a little extra to write the cache). *In Preflight:* off the sampled setting, cost shows
**Not measured here** until you sample with caching on.

#### Tool retry cap

How many times a failing tool call is retried before the task gives up on it. *In Preflight:*
retries are counted in the sample, not simulated, so numbers don't move yet.

#### Reasoning effort

How long the model thinks before answering. More effort, more tokens and cost, often better answers.
*In Preflight:* not read yet: a move changes no number, and the site doesn't say so yet. Sample the
tasks again at the effort you want.

#### Temperature

How random word choice is: 0 gives nearly the same answer each time. Mostly affects quality, but can
change answer length and cost. *In Preflight:* numbers don't change but read **Extrapolated**; a new
sample measures it.

#### Max new tokens

Longest answer the model may write. Caps cost and time per call; too low cuts answers off. *In
Preflight:* moves memory when the script sets no sequence length; time isn't modelled yet.

### Serving

Preflight's serving model is off on the public site, so most of these change only a status today.

#### Max model length

Longest prompt plus answer the server accepts. Longer needs more KV-cache memory (the model's notes
on everything said so far). *In Preflight:* moves memory and time when the script sets no sequence
length.

#### Sequences in flight

Requests the server handles at once. More throughput, more KV-cache memory. *In Preflight:* no
number moves while the serving model is off.

#### Prefix caching

Keeps the KV cache of shared prompt starts so repeated system prompts aren't recomputed. *In
Preflight:* not modelled yet.

#### KV-cache dtype

Number format for the KV cache. 8-bit fits about twice the conversations in the same memory, at a
small quality cost. *In Preflight:* fp8 halves KV-cache memory; time isn't modelled yet.

### Retrieval

#### Encode batch

Texts the encoder turns into vectors at once. Bigger is faster until memory runs out. *In
Preflight:* moves memory and step time when the script has no other batch size.

#### HNSW links (M)

For a graph search index: neighbours each vector links to. More links, better recall, more memory,
slower build. *In Preflight:* not read yet; memory and time hold.

#### IVF lists

For a clustered search index: how many clusters the vectors split into. More lists make each search
faster but can miss neighbours unless you search more lists (`nprobe`), and need more data to train.
*In Preflight:* not read yet; time holds.

#### Dump hidden states

Saves the model's intermediate vectors to disk for analysis; costs disk and time. *In Preflight:*
not modelled yet.

### Graph neural networks

#### Hidden width

Size of each node's vector. Wider learns more and uses more memory. *In Preflight:* not read yet;
numbers hold at the probes' value, marked **Extrapolated**.

#### GNN layers

How many hops of neighbours each node hears from. Each layer adds a hop, and memory grows fast. *In
Preflight:* not read yet, as for hidden width.

#### Neighbours at hop 1

#### Neighbours at hop 2

#### Neighbours at hop 3

#### Neighbours at hop 4

Neighbours sampled at each hop when building a batch. Batch size multiplies hop by hop, so these move
memory more than almost anything else in a GNN. *In Preflight:* raise memory, time and cost; their
product is on the job's size axis.

### Vision

#### Image size

Width and height images are resized to. For CNNs, memory and time grow with its square (vision
transformers grow faster): 224 to 448 pixels is about four times the work. *In Preflight:* moves
memory and step time, with batch × size² on the size axis.
