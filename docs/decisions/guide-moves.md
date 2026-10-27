# Guide moves: which P(OOM) a proposed setting may have (C, Oct 27)

**Question.** The guide (rule-based tour and Muse Spark's `set_knob`) proposes one setting at a
time. How risky may that setting be? B's words are fixed (engine-settings.md S2b): fits below
1% P(OOM), fails at 80% or more, tight between.

**Method.** `web/scripts/mc_guide_moves.py` (stdlib only, `python3 web/scripts/mc_guide_moves.py`).
20,000 simulated jobs per seed, seeds 7, 11, 23. Each job has a batch ladder 1 to 64 where
cheaper settings use more memory; the engine's prediction carries a log-normal error at B's
calibrated widths (0.107 / 0.160 / 0.241 half-width, 60 / 30 / 10% of jobs) and turns into
P(OOM) with `engine.memory.p_oom`. The guide picks the cheapest setting its rule allows. A crash
loses 5 to 30% of that run (up to the first eval) and retries one step safer, paying again on
each crash. Compared with an oracle that knows the true peak.

| Rule for a proposed move | Runs that crash | Mean extra cost vs oracle | p95 extra cost |
|---|---|---|---|
| A: fits only (P < 1%) | 0.06 to 0.11% | 42% | 171 to 189% |
| B: P < 5% | 0.5% | 27% | 85% |
| C: P < 10% (C's first build) | 1.1% | 20% | 83% |
| D: anything not "fails" (P < 80%) | 18 to 19% | 3.7% | 18% |
| **E: apply only "fits"; when a "tight" setting beats the best fitting move by 10% or more of the aim, offer to measure it first (the person's click, ~2% of the run)** | **0.06 to 0.11%** | **5.4 to 5.8%** | **22 to 39%** |

**Chosen: E.** It crashes as rarely as fits-only and costs a tenth of fits-only's extra.
D is cheaper on paper but one run in five crashes, which is the night Preflight exists to save.

**In code.** `GridProvider.moves()` never returns a move into "fails"; it returns "tight" moves
only with `includeTight`, flagged `fit: "tight"`. The tour applies only `fit: "fits"` moves and
turns a tight one into a `probe_offer` step ("Measure it", which queues `run_probe` for
approval) when `|relative|` beats the best fitting move by `PROBE_WORTH = 0.1`. While the run
itself is at risk (P >= 1%), fixing it comes first and no saving is offered; if no fitting fix
exists, "nothing safe" offers to measure the move that lowers the risk most. The knob-specific
"the run would stop at the first eval" lines appear only at P >= 80%; between 1% and 80% the
guide uses the generic "makes the run fit" line.

**Caveats.** The cost curve and crash loss are assumptions (diminishing returns on batch,
crash at the first eval); E stays ahead of A, B and C under every seed, and the 10% offer cut
is a choice of how often to ask, not a safety limit.
