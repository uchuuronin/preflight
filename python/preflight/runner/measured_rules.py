"""Findings the probes measure, as opposed to those read from the code (harness/skeleton.py).

Agent runs, from the record's task sample:

- agent.no_prompt_caching: no input token was served from cache across a sample of multi-turn
  tasks, so every turn re-pays the whole history
- agent.context_overflow: the history grows by a measured number of tokens a turn and passes the
  script's context limit at a turn inside max_turns, or a sampled task already passed it

Each carries the numbers it rests on as evidence, with how many tasks and turns are behind them.
"""

import math
import statistics

from preflight.schema import Evidence, Finding

MIN_INPUT_TOKENS = 50_000  # guess: below this a sample is too small to say caching never happened
MIN_TURN_PAIRS = 10  # guess: fewer consecutive-turn pairs than this give no stable growth rate


def _knob(rec, kid):
	k = next((k for k in rec.knobs if k.id == kid), None)
	return k.current if k is not None and isinstance(k.current, int | float) else None


def agent_findings(rec) -> list[Finding]:
	s = rec.agent_sample
	if s is None or not s.tasks:
		return []
	out = []
	tasks = s.tasks
	inp = sum(t.input_tokens for t in tasks)
	cached = sum(t.cached_input_tokens for t in tasks)
	multi = sum(1 for t in tasks if t.turns >= 2)
	if cached == 0 and inp >= MIN_INPUT_TOKENS and multi * 2 >= len(tasks):
		out.append(Finding(id="measured.agent.no_prompt_caching", severity="warn", code="agent.no_prompt_caching",
			title="No prompt caching: every turn re-pays the whole history",
			detail=f"0 of {inp:,} input tokens were served from cache over {len(tasks)} sampled tasks "
			f"({multi} of them multi-turn). A prefix that changes every call (a timestamp, a reordered "
			"tool list) defeats the provider's cache.",
			node_id="llm", source="rule", fix_template="agent.stable_prompt_prefix", knob_id="prompt_caching",
			evidence=[Evidence(metric="agent.cached_input_fraction", value=0.0),
				Evidence(metric="agent.input_tokens", value=float(inp))]))  # fmt: skip
	pairs = [
		b - a
		for t in tasks
		for a, b in zip(t.context_tokens_by_turn, t.context_tokens_by_turn[1:], strict=False)
	]
	firsts = [t.context_tokens_by_turn[0] for t in tasks if t.context_tokens_by_turn]
	limit = _knob(rec, "context_limit")
	max_turns = _knob(rec, "max_turns")
	seen_max = max((max(t.context_tokens_by_turn) for t in tasks if t.context_tokens_by_turn), default=0)
	if limit and seen_max > limit:
		out.append(Finding(id="measured.agent.context_overflow", severity="error", code="agent.context_overflow",
			title="Sampled tasks overflowed the context limit",
			detail=f"a sampled task reached {seen_max:,} input tokens against a limit of {int(limit):,}",
			node_id="context", source="rule", knob_id="context_limit",
			evidence=[Evidence(metric="agent.context_tokens_max", value=float(seen_max), threshold=float(limit))]))  # fmt: skip
	elif limit and len(pairs) >= MIN_TURN_PAIRS and firsts:
		g = statistics.median(pairs)
		t1 = statistics.median(firsts)
		if g > 0:
			turn = 2 + math.floor(
				(limit - t1) / g
			)  # the first turn past the limit (turn n holds t1 + (n-1)g)
			if max_turns is None or turn <= max_turns:
				out.append(Finding(id="measured.agent.context_overflow", severity="warn", code="agent.context_overflow",
					title=f"The history passes the context limit at about turn {turn}",
					detail=f"measured growth {g:,.0f} tokens a turn (median of {len(pairs)} turn pairs, "
					f"{min(pairs):,}–{max(pairs):,}) from {t1:,.0f} at turn 1 passes context_limit "
					f"{int(limit):,} at turn {turn} [derived]; the sample's longest task was "
					f"{max(t.turns for t in tasks)} turns, so no sampled task got there",
					node_id="context", source="rule", knob_id="context_limit",
					evidence=[Evidence(metric="agent.context_growth_tokens_per_turn", value=float(g)),
						Evidence(metric="agent.context_overflow_turn", value=float(turn), threshold=float(limit))]))  # fmt: skip
	return out
