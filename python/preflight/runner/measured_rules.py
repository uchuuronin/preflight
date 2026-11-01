"""Findings the probes measure, as opposed to those read from the code (harness/skeleton.py).

Agent runs, from the record's task sample:

- agent.no_prompt_caching: no input token was served from cache across a sample of multi-turn
  tasks, so every turn re-pays the whole history
- agent.context_overflow: the history grows by a measured number of tokens a turn and passes the
  script's context limit at a turn inside max_turns, or a sampled task already passed it

GPU lenses (hygiene_findings): old checkpoints shipped with the code, a cache loaded from a fixed
path, generations cut at their cap (with a Wilson interval), different weights generating
identical outputs, and checkpoints that a resume can't match the run from (no optimizer state,
schedule, RNG or step). The last is what the checkpoints hold, not a resumed run compared step
for step.

Each carries the numbers it rests on as evidence, with how many tasks and turns are behind them.
"""

import math
import re
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


# ---------------------------------------------------------------- run hygiene (GPU lenses)

ZERO_SPREAD_WARN = 0.5  # guess: past half the prompts teaching nothing, the batch is mostly wasted
TRUNCATION_WARN = 0.10  # guess: a tenth of generations hitting the cap cuts answers often enough to matter
_CKPT = re.compile(r"(?:^|/)(checkpoint-\d+|checkpoints?|ckpt)/")
# a file a cache would be: data saved for reuse (a config.json or a model directory isn't one)
_CACHE_EXT = re.compile(r"\.(pt|pth|pkl|pickle|npy|npz|arrow|parquet|feather|h5|cache|joblib)$")
# a resume or pretrained weights loaded when present: not a cache
_RESUME = re.compile(
	r"(?i)(^|[_\-.])(ckpt|checkpoints?|last|latest|resume|pretrained)([_\-.]|$)"
)  # on the file's stem
_RESUME_LOADERS = {"load_state_dict", "from_pretrained", "load_from_checkpoint"}
_STATE_SUFFIXES = (".json", ".yaml", ".yml", ".toml", ".cfg")  # a checkpoint's state, not a README or script
_WRITERS = {"save", "save_to_disk", "dump", "to_parquet", "to_feather", "savez", "write_bytes"}
_LOADERS = {"load", "load_from_disk", "load_state_dict", "from_pretrained", "load_dataset"}


def _wilson(k: int, n: int) -> tuple[float, float]:
	from preflight.engine.conformal import wilson

	return wilson(k, n)


def hygiene_findings(rec, files: dict[str, str]) -> list[Finding]:
	"""Stale checkpoints and caches in the code as sent, and from the probes: generations cut at
	their cap, and different weights generating identical outputs."""
	out: list[Finding] = []
	# files holds the text files sent (weights aren't among them): a checkpoint folder's state,
	# config or hparams, not code in a package named ckpt/
	ckpts = sorted({m.group(0).strip("/") for f in files if f.endswith(_STATE_SUFFIXES)
		for m in [_CKPT.search(f)] if m})  # fmt: skip
	if ckpts:
		out.append(Finding(id="hygiene.stale_checkpoint", severity="warn", code="hygiene.stale_checkpoint",
			title="Old checkpoints came with the code",
			detail=f"{', '.join(ckpts[:3])} sit beside the script: a trainer told to resume (HF Trainer's "
			"resume_from_checkpoint=True, Lightning's ckpt_path='last') starts from them instead of fresh, "
			"and a run that saves to the same place mixes old and new.",
			node_id=None, source="rule", fix_template="hygiene.fresh_output_dir"))  # fmt: skip
	# the graph's data node where it has one (a vision graph's is "images"), else none: a finding
	# pointing at a node the graph lacks makes the record unloadable
	data_node = next((n.id for n in rec.graph.nodes if n.kind == "data"), None)
	for where, detail in _unkeyed_caches(files):
		out.append(Finding(id=f"hygiene.cache_not_keyed.{len(out)}", severity="warn", code="hygiene.cache_not_keyed",
			title="A cache is reused whatever produced it", detail=f"{where}: {detail}", node_id=data_node,
			source="rule", fix_template="hygiene.key_the_cache"))  # fmt: skip
	reward_node = next((n.id for n in rec.graph.nodes if n.id == "reward"), None)
	for p in rec.probes:
		out += _zero_spread(
			p, reward_node, "group_size" if any(k.id == "group_size" for k in rec.knobs) else None
		)
		g = p.generation
		if g is not None and g.truncation_rate is not None and g.sequences:
			n = int(p.extras.get("generation_ending_known") or g.sequences)  # the rate's own total
			k = round(g.truncation_rate * n)
			lo, hi = _wilson(k, n)
			if g.truncation_rate > TRUNCATION_WARN:
				out.append(Finding(id=f"hygiene.truncation.{p.id}", severity="warn", code="generation.truncated",
					title=f"{g.truncation_rate:.0%} of generations hit their length cap",
					detail=f"{k} of {n} generated sequences ran to "
					+ (f"max_new_tokens={g.max_new_tokens}" if g.max_new_tokens else "their length cap")
					+ " without "
					f"ending (95% interval {lo:.0%}–{hi:.0%}) [measured]: their answers are cut off",
					node_id=None, source="rule", knob_id="max_new_tokens",
					evidence=[Evidence(metric="generation.truncation_rate", value=g.truncation_rate, probe_id=p.id,
						threshold=TRUNCATION_WARN)]))  # fmt: skip
		if p.extras.get("checkpoint_saves_seen"):
			missing = [w for k, w in (("optimizer", "optimizer state"), ("scheduler", "the LR schedule"),
				("rng", "RNG state"), ("step", "the step count")) if not p.extras.get(f"checkpoint_has_{k}")]  # fmt: skip
			if missing:
				sev = "warn" if "optimizer state" in missing or "the step count" in missing else "info"
				out.append(Finding(id=f"hygiene.resume_mismatch.{p.id}", severity=sev, code="hygiene.resume_mismatch",
					title="A resume from this run's checkpoints won't match the run",
					detail=f"its checkpoints hold the weights but not {', '.join(missing)} [measured: what torch.save was "
					"given]; a resumed run restarts "
					+ ("Adam's moments, " if "optimizer state" in missing else "")
					+ ("from the wrong step, " if "the step count" in missing else "")
					+ "so its losses won't match an unbroken run's",
					node_id=None, source="rule", fix_template="hygiene.full_checkpoint"))  # fmt: skip
		n, same = int(p.extras.get("outputs_compared", 0)), int(p.extras.get("outputs_identical", 0))
		if n >= 2 and same == n:
			out.append(Finding(id=f"hygiene.identical_outputs.{p.id}", severity="warn", code="hygiene.identical_outputs",
				title="Different weights gave identical outputs",
				detail=f"all {n} inputs generated by two different sets of weights (another adapter, model or "
				"checkpoint) came out token for token the same [measured]: the eval may be reading a cache, or the "
				"weights it means to compare were never loaded",
				node_id=None, source="rule",
				evidence=[Evidence(metric="hygiene.outputs_identical", value=float(same), probe_id=p.id,
					threshold=float(n))]))  # fmt: skip
	return out


def _zero_spread(p, node_id, knob_id) -> list[Finding]:
	"""GRPO: groups whose completions' summed rewards are all equal have zero advantage, so no
	gradient. Counted by the recorder over the sum of every reward function (what GRPO's advantage
	uses), with a Wilson interval over the groups; all wrong vs all right comes from the pass rates
	only when the pass/fail function is the whole reward."""
	rl, ex = p.rl, p.extras
	n, k = int(ex.get("rl_groups", 0)), int(ex.get("rl_zero_spread_groups", 0))
	if rl is None or n < 2 or not rl.group_size or k / n <= ZERO_SPREAD_WARN:
		return []
	lo, hi = _wilson(k, n)
	g = rl.group_size
	split = ""
	if ex.get("rl_reward_functions") == 1 and len(rl.pass_rate_per_prompt) == n:
		wrong, right = (
			sum(r == 0 for r in rl.pass_rate_per_prompt),
			sum(r == 1 for r in rl.pass_rate_per_prompt),
		)
		split = (
			" ("
			+ ", ".join(
				x for x in (f"{wrong} all wrong" if wrong else "", f"{right} all right" if right else "") if x
			)
			+ ")"
		)
	return [Finding(id=f"measured.rl.zero_spread.{p.id}", severity="warn", code="rl.zero_spread",
		title=f"{k / n:.0%} of prompts teach nothing: every completion scored the same",
		detail=f"{k} of {n} prompt groups{split} had all {g} completions score alike, summed over the reward "
		f"functions, so their advantage is zero and they add no gradient (95% interval {lo:.0%}–{hi:.0%}, "
		f"{n * g} completions) [measured]. Easier or harder prompts, a larger group, or a partial-credit "
		"reward give them spread.",
		node_id=node_id, source="rule", knob_id=knob_id,
		evidence=[Evidence(metric="rl.zero_spread_fraction", value=k / n, probe_id=p.id, threshold=ZERO_SPREAD_WARN)])]  # fmt: skip


def _unkeyed_caches(files: dict[str, str]) -> list[tuple[str, str]]:
	"""`if os.path.exists(P): x = load(...)` (or `Path(P).exists()`) with P a fixed path (not built
	from the model, the tokenizer, the length or a hash) that is a cache: the file saves to P too,
	or P has a data file's extension. The cache is reused after any of its inputs change; a config
	file or a model directory loaded when present is not a cache."""
	import ast

	hits = []
	for name, text in files.items():
		if not name.endswith(".py"):
			continue
		try:
			tree = ast.parse(text)
		except SyntaxError:
			continue
		consts: dict[str, str] = {}
		for node in ast.walk(tree):
			if (
				isinstance(node, ast.Assign)
				and len(node.targets) == 1
				and isinstance(node.targets[0], ast.Name)
				and isinstance(node.value, ast.Constant)
				and isinstance(node.value.value, str)
			):
				consts[node.targets[0].id] = node.value.value
		written = set()
		for c in ast.walk(tree):
			if (
				isinstance(c, ast.Call)
				and (getattr(c.func, "attr", None) or getattr(c.func, "id", None)) in _WRITERS
			):
				for a in [*c.args, *(k.value for k in c.keywords)]:
					if isinstance(a, ast.Constant) and isinstance(a.value, str):
						written.add(a.value)
					elif isinstance(a, ast.Name) and a.id in consts:
						written.add(consts[a.id])
		for node in ast.walk(tree):
			if not isinstance(node, ast.If):
				continue
			path = _exists_arg(node.test, consts)
			if path is None or not (path in written or _CACHE_EXT.search(path)):
				continue
			if _RESUME.search(path.rsplit("/", 1)[-1].rsplit(".", 1)[0]) or _CKPT.search(path):
				continue  # named as a checkpoint (its stem), or kept in a checkpoint folder
			loaded = _loaded_names(node.body, path, consts)
			if any(
				_feeds_weights(c, path, consts, loaded)
				for b in node.body
				for c in ast.walk(b)
				if isinstance(c, ast.Call)
			):
				continue  # the file is loaded into the model: a resume or pretrained weights, not a cache
			loads = [c for b in node.body for c in ast.walk(b) if isinstance(c, ast.Call)
				and (getattr(c.func, "attr", None) or getattr(c.func, "id", None)) in _LOADERS]  # fmt: skip
			if loads:
				hits.append((f"{name}:{node.lineno}", f"loads {path!r} when it exists, and the path doesn't change "
					"with the model, tokenizer or settings that produced it"))  # fmt: skip
	return hits[:3]


_WEIGHTS_KW = {"state_dict", "pretrained_model_name_or_path", "checkpoint_path"}


def _feeds_weights(call, path: str, consts, loaded: set[str]) -> bool:
	"""A weights loader (load_state_dict, from_pretrained, ...) whose weights argument (the first
	positional, or state_dict= / pretrained_model_name_or_path= / checkpoint_path=) is this path, a
	load of it (`torch.load(P)`), or a name the if-body loaded it into (`sd`, `sd["model"]`,
	`sd.get("model", sd)`), found anywhere inside that argument. A
	loaded name used only for a setting (`vocab_size=len(tokens)`) is not weights."""
	import ast

	if (getattr(call.func, "attr", None) or getattr(call.func, "id", None)) not in _RESUME_LOADERS:
		return False

	def is_path(a) -> bool:
		return (isinstance(a, ast.Constant) and a.value == path) or (
			isinstance(a, ast.Name) and consts.get(a.id) == path
		)

	given = call.args[:1] + [k.value for k in call.keywords if k.arg in _WEIGHTS_KW]
	# anywhere inside the weights argument itself (ckpt.get("model", ckpt), torch.load(open(P, "rb"))),
	# never in the loader's other arguments
	return any(is_path(x) or (isinstance(x, ast.Name) and x.id in loaded) for a in given for x in ast.walk(a))


def _loaded_names(body, path: str, consts) -> set[str]:
	"""Names assigned in the body from a call that takes the path (`sd = torch.load(P, ...)`)."""
	import ast

	out = set()
	for b in body:
		for n in ast.walk(b):
			if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
				args = [*n.value.args, *(k.value for k in n.value.keywords)]
				if any((isinstance(a, ast.Constant) and a.value == path) or (isinstance(a, ast.Name)
					and consts.get(a.id) == path) for a in args):  # fmt: skip
					out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
	return out


def _exists_arg(test, consts) -> str | None:
	"""The fixed path in `os.path.exists(P)` / `Path(P).exists()` / `P.exists()`, if it is one."""
	import ast

	call = test.operand if isinstance(test, ast.UnaryOp) else test
	if not isinstance(call, ast.Call) or isinstance(test, ast.UnaryOp):
		return None
	arg = None
	if getattr(call.func, "attr", None) in ("exists", "isdir", "isfile"):
		if call.args:
			arg = call.args[0]
		elif isinstance(call.func.value, ast.Call) and call.func.value.args:
			arg = call.func.value.args[0]
	if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
		return arg.value
	if isinstance(arg, ast.Name) and arg.id in consts:
		return consts[arg.id]
	return None
