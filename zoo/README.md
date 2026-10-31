# zoo/ (owner: A)

One script per job type with planted problems, plus the ground-truth full runs B validates against. See `zoo/README.md` sections per script.

## grpo_trl (RL post-training, Oct 31)

GRPO with TRL 0.14 on GSM8K: Qwen2.5-0.5B-Instruct, eight completions per prompt, two prompts a
micro-step, four micro-steps a step. Public model (Apache-2.0) and data (MIT), written for Preflight.

Planted, for Preflight to catch from the probes:

1. `max_completion_length=512` with a prompt that asks for step-by-step reasoning first: long
   answers are cut before `<answer>`, score 0, and show as the truncation rate (`rl.truncation_rate`,
   `generation.truncation_rate`) rising with question length.
2. Eight completions per prompt on short questions: many groups all pass or all fail, so their
   advantage is zero (`rl.pass_rate_per_prompt`, `rl.zero_spread_fraction`; B's model gives
   p^G + (1 - p)^G for other group sizes).
3. `beta=0.04`: a reference-model forward on every step (`step_time.by_phase.reference`).

What the recorder splits a step into: `rollout` (generate), `reference` (the reference log-probs,
a forward with no gradient), `reward` (the reward functions), and the update as `forward`,
`backward` and `optimizer`. Weight sync is zero here (no vLLM).

Ground truth: not run yet; the L4 run and its cost are asked for separately.
