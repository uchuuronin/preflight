# When "finishes in one free session" (D4)

Decided Tue Oct 27 by D. Kept honest by `web/src/addons/freetier/check.montecarlo.test.ts`,
which runs the same simulation against `checkPreset` itself.

## Question

The free-GPU band tells a visitor whether their plan finishes inside one Colab, Kaggle or
Lightning session. A wrong "one session" costs them a whole session, so the rule should almost
never say it falsely, while still saying it for most runs that do fit.

## Facts used

- Session caps [cited]: Colab "at most 12 hours, depending on availability" (Colab FAQ); Kaggle
  12 hours per GPU session (Kaggle docs, via GPU Per Hour, 21 Sep 2026); Modal's Starter plan, a
  Function runs at most 24 hours, with $30 of free credit a month (Modal timeouts and pricing,
  read 27 Oct 2026); Lightning AI's free Studio restarts every 4 hours (plan page, 22 Oct 2026).
- The records' 90% time intervals put `hi` at 1.15x the point for training runs and 1.40x for the
  agent run.
- Each session also spends time installing packages and downloading the model; not measured yet,
  taken as 5 to 30 minutes.

## Monte Carlo (calibration: simulated)

The simulated true times come from the same interval form the rule reads, so this shows which
rule is consistent with the intervals, not how often it is right on real runs. These figures are
not quoted in the README, the video or the UI.


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

`hi + 15 min <= cap`: in the simulation it is the cheapest rule that holds when the intervals are
too narrow, and it counts the setup that every session really spends. When it doesn't fit, the band
says how many sessions and to save a checkpoint at least every `cap - 15 min`.

## Replacing the simulation

Once A runs the zoo scripts to completion on a GPU, compare each one-session verdict with the
measured wall-clock plus measured setup, and report the share right with a Wilson 95% interval.
SETUP_S (15 minutes) is a guess until A's job events give fetch, install and model-load times.

## The real check, Sun Nov 1: not possible yet

Tried against A's measured L4 records: retrieval (`a-smoke-retrieval-st-l4`, GitHub run
41299583623), GNN (41299933798) and vision (41300131134) on `origin/remote-runs`, and the Oct 28
LoRA smoke record. None can score a one-session verdict:
- on all three new records the engine abstains on `time.total`: "the run's length isn't known:
  no max_steps, and no dataset size to divide by the batch" (the script maps leave
  `data.n_examples` empty), so D4 shows no session count, as it should;
- the LoRA smoke record has no estimates at all;
- no run on these was timed to the end on any GPU: the LoRA full run (`a-gt-lora-l4-5`) runs out
  of memory at step 200, and the others are probes only.

So the simulation stays, labelled as it is. What the records do check, now in
`check.realruns.test.ts`: on real input D4 never claims one session or a session count without a
trusted time, judges memory on the record's own peak (the GNN's measured 5.0 GiB, 4.5 to 5.6, fits
every free tier's GPU) and abstains where the engine gave no trusted memory (vision) or no
estimates (LoRA).

What would make the real check possible: (1) the harness filling `data.n_examples` (A, asked in
`docs/requests/from-d.md`), so the engine predicts `time.total`; (2) a zoo script run to the end
with its wall-clock and setup time measured, ideally on a free tier's GPU (T4 or L4).

## Caveats

Colab's cap is a ceiling, not a promise: sessions end sooner under load, and the band says so.
Time is judged only when the run was predicted for a GPU the tier offers; otherwise the band
checks memory and asks for a probe on that GPU.
