# What the "New to this" level shows (D1)

Revised Tue Oct 27, evening, after Chiya's decision on warnings. Kept honest by
`web/src/addons/expertise/dial.zoo.test.ts`.

## Rule

At "New to this" the view picks at most 5 knobs by itself [cited: Cowan 2001, about 4 chunks;
5 is the upper end of 4 ± 1], claiming places in this order:

1. the knobs of every error or critical finding, on the grid or not (`eval_no_grad` included);
   these are never dropped to stay under the cap;
2. the top 2 knobs for the visitor's current goal (C's `ranked`, from `sensitivity()` on the
   active objective);
3. the lens baseline (knobs a newcomer recognises) [guess: chosen by D from the Users and views
   tab; replace with which knobs first-time visitors open, from session telemetry after Nov 4].

The order of 2 and 3 is a guess: on A's fixtures the goal ranking is empty for most goals, so
failures-first orders with the goal's top knobs before, after or in place of the baseline all
show the same knobs (checked Oct 27). Rerun the comparison on B's real grids.

Warnings (`warn` findings) don't take a place. They are listed as findings under the dock
("3 things worth checking", `worthChecking.ts`), and a warning's knob shows once the visitor
opens it, staying for the rest of the visit. On top of the five, the view keeps any knob the
visitor opened or moved this visit; C's dock also keeps pinned, guide-highlighted, open and moved
knobs, and offers "Show N more". The list waits on C (docs/requests/from-d.md).

## Test

Ground truth is A's zoo, not the dial: each zoo script's docstring lists its planted problems
and which make the run fail. For each golden record built from a zoo script (matched by
`job.source.entrypoint`), a newcomer arrives at the record's own setting with one of three goals
(cheapest, fastest, fits in memory); the test runs C's `applyKnobViews` with D's view and checks
which planted knobs are on screen.

- Asserted, for every goal: every knob behind a planted failure is in the dock; at most 5 knobs
  before the visitor does anything; every planted problem is visible (knob in the dock, or its
  warning in the list); and each listed warning's knob stays in the dock after the visitor opens
  and closes it. Checked that the test fails when the list is emptied or the reveal removed.
- Reported, Wilson 95%, on A's fixture predictions (`golden-fixture@0`), Oct 27:

| Record | Visible | In the dock | Most knobs |
| --- | --- | --- | --- |
| finetune.json (zoo/finetune_lora_hf) | 9 of 9, 100% [70, 100] | 6 of 9, 67% [35, 88] | 4 |
| pretrain.json (zoo/pretrain_nanogpt) | 9 of 9, 100% [70, 100] | 3 of 9, 33% [12, 65] | 4 |

Packing, fp32 and the sparse checkpoints are visible as listed warnings. "Visible" counts the
list, which shows once C renders it. Rerun on B's grids once `make_golden.py` replaces the
fixtures. Then score it once on the unseen scripts (Mon Nov 2) with their planted fixes as ground
truth, as a held-out check: the constants are set on B's grids first and not retuned on them.

## Superseded

The Oct 27 morning Monte Carlo (random settings over gridded knobs, scored on three goals while
ranking by cost) is withdrawn: its top-2 rule covered both movable knobs by construction, so it
could not fail, and its 44 misses came from ranking by one goal while scoring others. Its
figures were calibration: simulated and are not quoted anywhere.
