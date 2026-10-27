# When "finishes in one free session" (D4)

Decided Tue Oct 27 by D. Kept honest by `web/src/addons/freetier/check.montecarlo.test.ts`,
which runs the same simulation against `checkPreset` itself.

## Question

The free-GPU band tells a visitor whether their plan finishes inside one Colab, Kaggle or
Lightning session. A wrong "one session" costs them a whole session, so the rule should almost
never say it falsely, while still saying it for most runs that do fit.

## Facts used

- Session caps: Colab "at most 12 hours, depending on availability" (Colab FAQ); Kaggle 12 hours
  per GPU session (Kaggle docs, via GPU Per Hour, 21 Sep 2026); Lightning AI's free Studio
  restarts every 4 hours (plan page captured 22 Oct 2026).
- The records' 90% time intervals put `hi` at 1.15x the point for training runs and 1.40x for the
  agent run.
- Each session also spends time installing packages and downloading the model; not measured yet,
  taken as 5 to 30 minutes.

## Monte Carlo

200,000 runs with predicted times from 2 to 20 hours on a 12-hour cap; true time drawn around
the prediction as the intervals claim (and again 1.5x wider, in case they are too narrow).

| Rule for "one session" | Right when it says so | Runs that fit, found | Right if intervals 1.5x too narrow |
| --- | --- | --- | --- |
| point <= cap | 96.6% | 98.0% | 95.3% |
| hi <= cap | 99.7% | 90.3% | 99.0% |
| point + 15 min <= cap | 97.3% | 97.5% | 95.9% |
| **hi + 15 min <= cap** | **99.8%** | **89.2%** | **99.2%** |
| hi + 30 min <= cap | 99.9% | 88.0% | 99.4% |

## Decision

`hi + 15 min <= cap`. Wrong about 1 time in 500, still under 1 in 100 if the intervals are too
narrow, and it counts the setup that every session really spends. When it doesn't fit, the band
says how many sessions and to save a checkpoint at least every `cap - 15 min`.

## Caveats

Colab's cap is a ceiling, not a promise: sessions end sooner under load, and the band says so.
Time is judged only when the run was predicted for a GPU the tier offers; otherwise the band
checks memory and asks for a probe on that GPU.
