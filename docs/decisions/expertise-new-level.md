# How many knobs the "New to this" level shows (D1)

Decided Tue Oct 27 by D. Kept honest by `web/src/addons/expertise/tiers.test.ts`.

## Question

At the "New to this" detail level the dock should show few knobs, but never hide the one
that matters. The rule is: the lens baseline (knobs a newcomer recognises), plus the top K
knobs by `sensitivity()` at the current setting, plus anything pinned or under the guide.
What should K be?

## Research

- Working memory holds about four chunks (Cowan 2001, "The magical number 4"); Hick's law
  makes choice time grow with the log of the number of options. Both argue for five or
  fewer knobs on screen for a newcomer.
- The failure Preflight exists to catch must never be hidden: the knob that clears a
  predicted failure has to be visible every time.

## Monte Carlo

Four sample records (C's dev fine-tune, A's golden fine-tune, pretrain and agent), 300
random settings each over the knobs the grids cover (others held at the record's value, as
C's provider requires), three goals (cheapest, fastest, fits in memory), C's GridProvider.
963 goal cases with a best move, 303 with a predicted failure to clear.

| New level shows | Best knob for any goal visible | Failure fix visible | Knobs shown, mean / max |
| --- | --- | --- | --- |
| Baseline only | 82.6% | 61.7% | 2.75 / 3 |
| Baseline + top 1 | 95.1% | 99.0% | 2.79 / 4 |
| **Baseline + top 2** | **95.4%** | **100%** | **2.81 / 5** |
| Baseline + top 3 | 96.7% | 100% | 2.85 / 6 |
| Baseline + top 4 | 97.9% | 100% | 2.89 / 7 |

## Decision

K = 2: the smallest K that always shows the failure fix, with at most five knobs on screen.
K = 3 buys 1.3 points of coverage for a sixth knob, which matters more once B's engine
grids more knobs.

## Caveats

The grids are A's fixture predictions (`golden-fixture@0`), the agent record has none, and
only two to four knobs per record are gridded. The test reruns the same simulation in CI;
when B's engine replaces the fixtures, a failing threshold means revisit K.
