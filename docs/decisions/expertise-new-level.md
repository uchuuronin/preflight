# What the "New to this" level shows (D1)

Revised Tue Oct 27 after the Oct 27 review. Kept honest by
`web/src/addons/expertise/dial.zoo.test.ts`.

## Rule

At "New to this" the knob dock shows:

- the lens baseline (knobs a newcomer recognises) [guess: chosen by D from the Users and views
  tab; replace with which knobs first-time visitors open, from session telemetry after Nov 4];
- the top 2 knobs for the visitor's current goal (C's `ranked`, from `sensitivity()` on the
  active objective) [guess: replace with the zoo coverage below once B's real grids land];
- the knobs of every error or critical finding, on the grid or not (`eval_no_grad` included);
- any knob the visitor has touched this visit, even if put back.

C's dock also keeps pinned, guide-highlighted, open and moved knobs, and offers "Show N more".

## Test

Ground truth is A's zoo, not the dial: each zoo script's docstring lists its planted problems
and which make the run fail. For each golden record built from a zoo script (matched by
`job.source.entrypoint`), a newcomer arrives at the record's own setting with one of three goals
(cheapest, fastest, fits in memory); the test runs C's `applyKnobViews` with D's view and checks
which planted knobs are on screen.

- Asserted: every knob behind a planted failure is shown for every goal. The test fails if the
  failure rule is removed (checked: 0 of 6 on the fine-tune).
- Reported, Wilson 95%, on A's fixture predictions (`golden-fixture@0`), Oct 27:

| Record | Planted problems shown | Hidden |
| --- | --- | --- |
| finetune.json (zoo/finetune_lora_hf) | 6 of 9, 67% [35, 88] | packing, for all three goals |
| pretrain.json (zoo/pretrain_nanogpt) | 3 of 9, 33% [12, 65] | precision and checkpoint interval, for all three goals |

Both failure knobs on the fine-tune show every time; the hidden ones are warnings (padding,
fp32, sparse checkpoints). Whether "New to this" should show warning knobs too is open with
Chiya. Rerun on B's grids once `make_golden.py` replaces the fixtures.

## Superseded

The Oct 27 morning Monte Carlo (random settings over gridded knobs, scored on three goals while
ranking by cost) is withdrawn: its top-2 rule covered both movable knobs by construction, so it
could not fail, and its 44 misses came from ranking by one goal while scoring others. Its
figures were calibration: simulated and are not quoted anywhere.
