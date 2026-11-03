# Preflight: a one-page brief

**Link:** <https://preflight-cqn.pages.dev>

## What it does

Preflight is a flight simulator for training jobs. You give it your training script; it runs a few
minutes of the real job on a rented GPU, builds a model of how the whole run will behave, and shows
it as a map you can fly: how much memory it needs, how long it takes, what it costs and what could
go wrong. Move a setting and the map moves with it. When something is wrong, such as an
evaluation step that will run out of memory, it says so before you spend a full run finding out.

It's built for people who train or evaluate neural networks in PyTorch: fine-tuning, pretraining,
RL post-training, evaluation, retrieval, graph networks, vision and agent runs.

## What to try

1. **Open a sample.** No code needed. Samples open straight from the page; each says whether its
   numbers are illustrative or measured.
2. **Check in your own job.** A public GitHub repo, a `.py` file or a Jupyter notebook (the
   Notebook tab). Preflight measures it on a real GPU and opens the map. If your repo has no
   `requirements.txt`, it installs what the repo declares; if it still misses a package, it says
   which, and you can name extra packages and send it again.
3. **Act on one finding.** Open something Preflight flagged and change what it points to: move the
   setting yourself, ask the guide, or let Muse Code write the fix and Preflight check it.

## How long it takes

- The sample: a couple of minutes.
- Your own job: a few minutes of waiting while it measures. One measured run went from upload to
  its map in 6 min 45 s (Oct 28; one run, so treat it as an example, not a promise).
- The whole session, with the short form at the end: about 40 minutes.

## What happens to your code

Before you check in, the desk shows exactly what is sent and kept, and you tick "Got it" (in the
"Where your code goes" panel, below the list of samples) to go on. In short:

- The code runs sandboxed on a rented Modal GPU with none of our keys.
- Muse Spark (Meta) reads it to build the map, and later reads plots of your run's memory and
  loss with the numbers behind them.
- Muse Spark Open (Meta's open model, on Baseten) reads the end of each probe's log, which can
  show lines of your code, and checks the numbers in model-written findings.
- Tick "Private triage" and the logs go to Muse Glimmer on Preflight's own GPU instead.
- The live guide sends your note and messages to Muse Spark, and fixes are written by Muse Code.
- The result becomes a flight record that anyone with its link can open.
- Records and uploads are deleted about 30 days after they're made or last used.

Keep keys and passwords out of the code you bring.

## Limits on the public site

Please check in no more than 3 jobs: the whole site shares a daily GPU budget. (If a check-in
stops at a missing package before measuring anything, one resend of the same code that day
is free.) If the site says
its GPUs are busy or the day's budget is gone, that's a limit of the demo, not something you did;
just carry on with a sample.

## What we'd like from you

Think aloud as you go: what you expect, what surprises you, where you're unsure. We're testing the
product, not you; if something is confusing, that is exactly what we need to hear.
