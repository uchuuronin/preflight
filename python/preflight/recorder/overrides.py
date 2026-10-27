"""Apply knob values to a copy of the user's files by rewriting the literal at each knob's location.

Only knobs with a verified location (a module constant, a keyword argument
holding a literal, one element of a literal list such as num_neighbors, or a
number in a launcher script) are rewritten. Knobs that need code (packing,
no_grad) go through patch templates instead, not here.

Written values are always literals produced by repr/json, never spliced text,
so a value can't smuggle code into the script. A dtype written as
torch.bfloat16 stays an attribute.
"""

import ast
import json
import re

from preflight.harness.static import PRECISION
from preflight.schema.script_map import ScriptKnob, ScriptMap


class OverrideError(ValueError):
	pass


# canonical precision -> torch dtype name
_TORCH_DTYPE = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
_SAFE_ATTR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _literal_text(value) -> str:
	if isinstance(value, str):
		return json.dumps(value, ensure_ascii=True)  # a valid, fully escaped Python string literal
	if isinstance(value, bool | int | float) or value is None:
		return repr(value)
	raise OverrideError(f"unsupported override value {value!r}")


def _line_starts(src: str) -> list[int]:
	"""Offsets of each line start, splitting on \\n only (as the tokenizer does), not \\x0c or \\u2028."""
	starts, pos = [0], 0
	while True:
		pos = src.find("\n", pos)
		if pos < 0:
			return starts
		pos += 1
		starts.append(pos)


def _offset(src: str, starts: list[int], line: int, col: int) -> int:
	"""Character offset of an AST (line, col); ast columns count UTF-8 bytes."""
	begin = starts[line - 1]
	end = starts[line] if line < len(starts) else len(src)
	return begin + len(src[begin:end].encode("utf-8")[:col].decode("utf-8", errors="ignore"))


def _value_node(tree: ast.Module, k: ScriptKnob) -> tuple[ast.AST, str | None]:
	line = k.location.line
	name = (k.location.expr or "").split("=", 1)[0].strip()
	hop = re.fullmatch(r"fanout\.hop(\d+)", k.id)
	for node in ast.walk(tree):
		if (
			hop
			and isinstance(node, ast.keyword)
			and node.arg == "num_neighbors"
			and isinstance(node.value, ast.List)
		) and node.value.lineno == line:
			i = int(hop.group(1)) - 1
			if i < len(node.value.elts):
				return node.value.elts[i], None
		if (
			isinstance(node, ast.Assign)
			and node.lineno == line
			and len(node.targets) == 1
			and isinstance(node.targets[0], ast.Name)
			and node.targets[0].id == name
		):
			return node.value, None
		if isinstance(node, ast.keyword) and node.value.lineno == line and node.arg == name:
			return node.value, node.arg if node.arg in ("bf16", "fp16") else None
	raise OverrideError(f"knob {k.id}: no literal found at {k.location.file}:{line}")


def _new_text(k: ScriptKnob, node: ast.AST, value, flag: str | None) -> str | None:
	"""Replacement source for the node, or None when the script already says this."""
	if k.id == "precision":
		if flag in ("bf16", "fp16"):
			if value == flag:
				new = True
			elif value == "fp32":
				new = False
			else:
				raise OverrideError(
					f"precision {value!r}: the script sets {flag}=True; switching to {value} needs a code change"
				)
			return None if isinstance(node, ast.Constant) and node.value is new else repr(new)
		dtype = _TORCH_DTYPE.get(value, value)
		if isinstance(node, ast.Attribute):
			if not _SAFE_ATTR.match(str(dtype)):
				raise OverrideError(f"precision {value!r} is not a torch dtype")
			return None if node.attr == dtype else f"torch.{dtype}"
		cur = node.value if isinstance(node, ast.Constant) else None
		if isinstance(cur, str):
			torch_style = cur.lower() in _TORCH_DTYPE.values()
			new = dtype if torch_style else value
			if PRECISION.get(cur.lower()) == value and cur == new:
				return None
			return _literal_text(new)
		return _literal_text(value)
	try:
		cur = ast.literal_eval(node)
	except (ValueError, SyntaxError, TypeError):
		cur = object()
	if cur == value and type(cur) is type(value):
		return None
	return _literal_text(value)


def _text_override(src: str, k: ScriptKnob, value) -> str:
	"""Launcher files (.sh, .sbatch, .yaml): replace the number inside the recorded expression."""
	if not isinstance(value, int) or isinstance(value, bool):
		raise OverrideError(f"knob {k.id}: launcher files take integer overrides")
	lines = src.split("\n")
	i = k.location.line - 1
	expr = k.location.expr or ""
	if not (0 <= i < len(lines)) or not expr or expr not in lines[i]:
		raise OverrideError(f"knob {k.id}: {expr!r} not found at {k.location.file}:{k.location.line}")
	new_expr = re.sub(r"\d+(?!.*\d)", str(value), expr, count=1)
	lines[i] = lines[i].replace(expr, new_expr, 1)
	return "\n".join(lines)


def apply_overrides(files: dict[str, str], sm: ScriptMap, values: dict[str, object]) -> dict[str, str]:
	out = dict(files)
	by_id = {k.id: k for k in sm.knobs}
	edits: dict[str, list[tuple[int, int, str]]] = {}
	for kid, value in values.items():
		k = by_id.get(kid)
		if k is None or k.location is None or k.location.line is None:
			raise OverrideError(f"knob {kid} has no code location to override")
		if k.needs_code:
			raise OverrideError(f"knob {kid} needs a code patch, not a value override")
		path = k.location.file
		if path not in out:
			raise OverrideError(f"knob {kid} points at {path}, which is not among the files")
		if not path.endswith(".py"):
			out[path] = _text_override(out[path], k, value)
			continue
		src = out[path]
		node, flag = _value_node(ast.parse(src), k)
		text = _new_text(k, node, value, flag)
		if text is None:
			continue
		starts = _line_starts(src)
		start = _offset(src, starts, node.lineno, node.col_offset)
		end = _offset(src, starts, node.end_lineno, node.end_col_offset)
		edits.setdefault(path, []).append((start, end, text))
	for path, spans in edits.items():
		src = out[path]
		for start, end, text in sorted(spans, reverse=True):
			src = src[:start] + text + src[end:]
		try:
			ast.parse(src)  # never hand the probe a broken file
		except SyntaxError as exc:
			raise OverrideError(f"overriding {path} would break it: {exc.msg} at line {exc.lineno}") from exc
		out[path] = src
	return out
