# What the "New to this" level shows (D1)

Revised Tue Oct 27, night, after Chiya's four conditions for turning `d.expertise` on. Kept
honest by `web/src/addons/expertise/dial.zoo.test.ts`, `tiers.test.ts` and
`tests/test_d_knob_census.py`.

## Rule

At "New to this" the view picks at most 5 knobs by itself, claiming places in this order:

1. **Must stay** (never dropped for the cap): the knobs of every error or critical finding, on the
   grid or not (`eval_no_grad` included); and, while C's dock isn't listing the warnings, the
   knobs of every warning too, so a newcomer never loses a warning entirely.
2. **The goal's top 2**: C's `ranked`, from `sensitivity()` on the visitor's current objective.
3. **The baseline**: the lens's most common settings in training examples, most common first.

On top of the five, the view keeps any knob the visitor opened or moved this visit; C's dock also
keeps pinned, guide-highlighted, open and moved knobs, and offers "Show N more".

Once C's dock lists the warnings under the row ("3 things worth checking", from
`worthChecking.ts`) and says so on the context (`findingsShown`), warnings stop claiming places;
a warning's knob shows when the visitor opens it and stays for the visit. Request in
`docs/requests/from-d.md`.

## Basis for each decision

| Decision | Basis | Test that fails if it's wrong |
| --- | --- | --- |
| At most 5 knobs picked | cited: Cowan 2001, about 4 chunks in working memory; 5 is the upper end of 4 ± 1 | `tiers.test.ts` "picks at most five"; `dial.zoo.test.ts` "at most five" |
| Failures first, never dropped | Chiya and the Oct 27 review | `tiers.test.ts` "never drops a failure's knob"; `dial.zoo.test.ts` failure knobs in the dock for every goal |
| Warnings stay in the dock until the list is on screen | Chiya, Oct 27 night | `tiers.test.ts`, `knobView.test.ts` "keeps a warning's knob ... while C's list isn't showing"; `dial.zoo.test.ts` on screen, every planted problem in the dock for every goal |
| A warning's knob shows once opened, and stays | Chiya, Oct 27 | `knobView.test.ts` reveal test; `dial.zoo.test.ts` "shows a listed warning's knob once the visitor opens it" |
| Goal's top knobs before the baseline, and how many (2) | guess; kept unless B's grids show a version that beats it by a clear margin (Wilson intervals not overlapping), see below | `dial.zoo.test.ts` "keeps the configured policy unless another beats it by a clear margin"; `wilson.test.ts` for the margin rule |
| Baseline for finetune, pretrain, rl, vision | measured: the census below | `tiers.test.ts` "takes each training lens's baseline from the census ranking"; `test_d_knob_census.py` consistency and engine-id checks |
| Baseline for agent, eval, serving | guess: from the Users and views tab; replace with what first-time visitors open, from session telemetry after Nov 4 | `tiers.test.ts` "names only knob ids the app knows" (ids only) |

## Order of the five and the top count

`dial.zoo.test.ts` compares three orders (failures, top, baseline; failures, baseline, top;
failures, top only) with the goal's top 1, 2 or 3, by how often the planted problems' knobs are
in the dock with the list on screen. Each record is counted once, under its own goal (the
fine-tune's is fastest, the pretrain's cheapest): one count per planted problem, pooled over
records (Chiya, Oct 28). Until Oct 27 it counted each problem once per goal, three times per
record, which made the intervals look narrower than the evidence is.

These are the two zoo records the rule is tuned on, so taking the best of nine would test on its
own training data (Chiya, Oct 27). Today's version (failures, top, baseline; top 2) stays unless
another beats it by a clear margin: its Wilson 95% interval wholly above today's. The comparison
is recorded here either way. Problems in the same record share a script, so even per record the
counts aren't fully independent.

| Predictions | Today's version | Best challenger | Clear margin? | Kept |
| --- | --- | --- | --- | --- |
| `golden-fixture@0`, Oct 27, counted per goal (superseded) | 9 of 18, 50% [29, 71] | all eight tie | no | failures, top, baseline; top 2 |
| `golden-fixture@0`, Oct 28, counted per record | 3 of 6, 50% [19, 81] | all eight tie at 3 of 6 | no | failures, top, baseline; top 2 |
| B's grids (Wed Oct 28) | to run | | | |

How much the comparison can tell: none of it yet. With 6 planted problems, today's 3 of 6 has
an interval of [19, 81], and even a perfect 6 of 6 has a lower bound of 61, so no challenger can
clear it [derived: Wilson 95%]. A clear win needs at least 12 planted problems if today's version
scores half (12 of 12 gives a lower bound of 76 against 75 for 6 of 12), so 4 zoo records at 3
problems each. B's grids change the predictions, not the number of records, so they alone won't
make it decide; the test reports "can't decide" until it can.

Why 2 and not 1 or 3: not yet shown by data; on the fixtures all three counts tie. Two keeps
room for at least one basic beside two failures within the five. If B's grids also tie or show
no clear margin, 2 stays as a guess, and the held-out run on Nov 2 is its test.

## Census: the newcomer basics

A unit is one example: a script with its documented launch commands, an override config, or a
zoo script. A setting counts once per example that sets it, mapped to knob ids with A's harness
vocabulary (`preflight.harness.static`), plus `--bf16` and `--fp16` for precision. Built by
`python -m preflight.addons.knob_census`, output in `web/src/addons/expertise/census.json`;
rebuilding from the pinned commits reproduces it byte for byte (checked Oct 27).

Sources, read 27 Oct 2026:

| Source | Commit | What counts | Examples |
| --- | --- | --- | --- |
| Bug zoo (A) | uchuuronin/preflight `ceb166a` | settings A's static reader finds in the script | 2 |
| [TRL scripts and examples](https://github.com/huggingface/trl/tree/c886cf2) | `c886cf2` | settings in code (A's reader), declared flags, docstring commands; sft, grpo, gspo, rloo | 22 |
| [transformers PyTorch examples](https://github.com/huggingface/transformers/tree/v5.19.0/examples/pytorch) | `v5.19.0` (`c4c4605`), the last release that ships them | README launch commands | 20 |
| [PEFT examples](https://github.com/huggingface/peft/tree/7a05ae5/examples) | `7a05ae5` | README launch commands and .sh files | 8 |
| [nanoGPT configs](https://github.com/karpathy/nanoGPT/tree/3adf61e/config) (popular, not Hugging Face) | `3adf61e` | names each config overrides | 3 |

Left out: full config files (nanotron, litgpt, torchtitan) list every field, so they can't show
which settings people choose; TRL's DPO, KTO, reward and distillation examples and transformers'
speech and generation examples have no matching lens; diffusion examples in PEFT likewise. Hardware
(`gpu_type`) isn't set in training scripts, so the census can't rank it; it shows when a failure
or the goal points at it.

**finetune**, 34 examples

| Setting | Examples that set it | Share, Wilson 95% |
| --- | --- | --- |
| `batch_size` | 28 of 34 | 82% [66, 92] |
| `epochs` | 20 of 34 | 59% [42, 74] |
| `lr` | 20 of 34 | 59% [42, 74] |
| `seq_len` | 18 of 34 | 53% [37, 69] |
| `eval_batch_size` | 11 of 34 | 32% [19, 49] |
| `lora_alpha` | 8 of 34 | 24% [12, 40] |
| `grad_accum` | 7 of 34 | 21% [10, 37] |
| `log_every` | 7 of 34 | 21% [10, 37] |

**pretrain**, 3 examples

| Setting | Examples that set it | Share, Wilson 95% |
| --- | --- | --- |
| `batch_size` | 3 of 3 | 100% [44, 100] |
| `max_steps` | 3 of 3 | 100% [44, 100] |
| `seq_len` | 3 of 3 | 100% [44, 100] |
| `d_model` | 2 of 3 | 67% [21, 94] |
| `eval_every` | 2 of 3 | 67% [21, 94] |
| `grad_accum` | 2 of 3 | 67% [21, 94] |
| `lr` | 2 of 3 | 67% [21, 94] |
| `n_head` | 2 of 3 | 67% [21, 94] |

**rl**, 15 examples

| Setting | Examples that set it | Share, Wilson 95% |
| --- | --- | --- |
| `max_completion_len` | 14 of 15 | 93% [70, 99] |
| `grad_accum` | 10 of 15 | 67% [42, 85] |
| `log_every` | 9 of 15 | 60% [36, 80] |
| `lr` | 9 of 15 | 60% [36, 80] |
| `batch_size` | 8 of 15 | 53% [30, 75] |
| `max_steps` | 8 of 15 | 53% [30, 75] |
| `group_size` | 7 of 15 | 47% [25, 70] |
| `use_vllm` | 6 of 15 | 40% [20, 64] |

**vision**, 3 examples

| Setting | Examples that set it | Share, Wilson 95% |
| --- | --- | --- |
| `batch_size` | 3 of 3 | 100% [44, 100] |
| `epochs` | 3 of 3 | 100% [44, 100] |
| `lr` | 3 of 3 | 100% [44, 100] |
| `eval_batch_size` | 2 of 3 | 67% [21, 94] |
| `grad_accum` | 1 of 3 | 33% [6, 79] |
| `log_every` | 1 of 3 | 33% [6, 79] |
| `num_workers` | 1 of 3 | 33% [6, 79] |
| `precision` | 1 of 3 | 33% [6, 79] |

Pretrain and vision rest on 3 examples each, so their intervals are wide; treat them as a first
reading. The dock shows each lens's ranking in order, skipping knobs the record doesn't declare,
until the cap. On the golden fine-tune today that is the two failure knobs (eval batch size, eval
without no_grad) and the packing warning, then batch size and sequence length from the baseline.

The unseen scripts stay out of the census entirely (Chiya, Oct 27): they are the held-out check
(below), and adding them after Nov 2 would change the basics after the held-out score was
recorded, so that score would describe an older version than the one judged.

## How it is checked

Ground truth is A's zoo, not the dial: each zoo script's docstring lists its planted problems
and which make the run fail. For each golden record built from a zoo script (matched by
`job.source.entrypoint`), a newcomer arrives at the record's own setting with one of three goals
(cheapest, fastest, fits in memory); the test runs C's `applyKnobViews` with D's view.

On A's fixture predictions (`golden-fixture@0`), Oct 28, each record under its own goal, one
count per planted problem. Safety checks (failure knobs in the dock, nothing lost) run for every
goal and pass.

| Record | On screen today (in-sample, 2 records) | Design check with the list (in-sample, 2 records, counts a list not yet on screen) |
| --- | --- | --- |
| finetune.json (zoo/finetune_lora_hf), fastest | 3 of 3 in the dock, 100% [44, 100]; at most 5 knobs | 3 of 3 visible, 100% [44, 100]; 2 of 3 in the dock |
| pretrain.json (zoo/pretrain_nanogpt), cheapest | 3 of 3 in the dock, 100% [44, 100]; at most 5 knobs | 3 of 3 visible, 100% [44, 100]; 1 of 3 in the dock |

Neither column is a result: both are in-sample on the two records the rule was written against.
The result is the held-out run on the unseen scripts (Mon Nov 2): their planted fixes as ground
truth, reported per record with Wilson 95% intervals, with the policy frozen on B's grids
beforehand and not retuned afterwards.

## Superseded

The Oct 27 morning Monte Carlo (random settings over gridded knobs, scored on three goals while
ranking by cost) is withdrawn: its top-2 rule covered both movable knobs by construction, so it
could not fail, and its 44 misses came from ranking by one goal while scoring others. Its
figures were calibration: simulated and are not quoted anywhere. The Oct 27 evening baseline
(gpu_type, batch size, sequence length for fine-tuning) was a guess and is replaced by the census.
