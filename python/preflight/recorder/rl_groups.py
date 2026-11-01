"""GRPO groups from the reward calls the recorder saw: pass rate per prompt and zero spread.

B's zero-variance model needs the pass rate p of each prompt: a group of G completions teaches
nothing when all of them pass or all fail, which happens with probability p^G + (1 - p)^G. TRL
lays a prompt's G completions out consecutively in every reward call, so the calls group by G.
Pure Python, tested without torch.
"""

import statistics

MAX_PROMPTS = 256  # pass rates kept per probe: enough for B's fit, small in the record


def reward_groups(calls: list[dict], g: int | None) -> dict:
	"""Pass rate per prompt and the zero-spread fraction from GRPO's reward calls, grouped by the
	trainer's group size (TRL lays a prompt's completions out consecutively). The pass/fail signal is
	the first reward function whose values are all 0 or 1; the spread is of the summed rewards."""
	if not calls or not g or g < 2:
		return {}
	out: dict = {"group_size": g}
	by_func: dict[int, list[list[float]]] = {}
	for c in calls:
		by_func.setdefault(c["func"], []).append(c["values"])
	binary = next(
		(f for f, vs in sorted(by_func.items()) if all(v in (0.0, 1.0) for row in vs for v in row)), None
	)
	if binary is not None:
		rates = []
		for vals in by_func[binary]:
			rates += [statistics.fmean(vals[i : i + g]) for i in range(0, len(vals) - g + 1, g)]
		out["pass_rate_per_prompt"] = [round(r, 4) for r in rates[:MAX_PROMPTS]]
	# summed over functions, call by call (one call per function per batch, in the same order)
	n_calls = min(len(v) for v in by_func.values())
	spread = []
	for k in range(n_calls):
		rows = [by_func[f][k] for f in by_func]
		if len({len(r) for r in rows}) != 1:
			continue
		total = [sum(col) for col in zip(*rows, strict=True)]
		spread += [max(total[i : i + g]) == min(total[i : i + g]) for i in range(0, len(total) - g + 1, g)]
	if spread:
		out["zero_spread_fraction"] = sum(spread) / len(spread)
		# the counts behind it, and how many reward functions the spread sums (for a Wilson interval,
		# and to tell all-wrong from all-right only when the pass signal is the whole reward)
		out["_counts"] = {"rl_zero_spread_groups": float(sum(spread)), "rl_groups": float(len(spread)),
			"rl_reward_functions": float(len(by_func))}  # fmt: skip
	return out
