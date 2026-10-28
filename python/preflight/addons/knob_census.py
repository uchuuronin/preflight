"""D1: which settings training examples actually set, per lens, for the "New to this" basics.

A unit is one example (a script with its documented launch commands, a config file, or a zoo
script). A setting counts once per unit if the unit sets it. Settings map to knob ids through
A's harness vocabulary (preflight.harness.static), plus the command-line precision flags the
harness reads from code instead (--bf16, --fp16). The ranking per lens is by units, ties by id.

Sources (pinned in census.json with their commits):
- the bug zoo, read by A's static reader (map_static);
- TRL's scripts and examples (settings in code, read by map_static, plus docstring launch
  commands): sft -> finetune; grpo, gspo and rloo -> rl; other objectives skipped;
- transformers' PyTorch examples (README launch commands), last release that ships them;
- PEFT's examples (README launch commands and .sh files);
- nanoGPT's configs (each file sets only what it overrides): train -> pretrain, finetune.
Full config files (nanotron, litgpt, torchtitan) are left out: they list every field, so they
can't show which settings people choose.

Run: python -m preflight.addons.knob_census <checkouts dir> <out json>
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from preflight.harness.static import ARGS_KNOBS, CONST_KNOBS, ScriptParseError, map_static

# Command-line flags the harness reads from code, not from *Arguments kwargs [derived: HF
# TrainingArguments exposes precision as --bf16 / --fp16 booleans and --torch_dtype].
CLI_KNOBS: dict[str, str] = {
	"bf16": "precision",
	"fp16": "precision",
	"torch_dtype": "precision",
	"dtype": "precision",
	"mixed_precision": "precision",
}

RUNNERS = ("python", "python3", "accelerate", "torchrun", "deepspeed", "trl", "CUDA_VISIBLE_DEVICES")


@dataclass(frozen=True)
class Unit:
	name: str
	lens: str
	knobs: frozenset[str]


def knob_for(flag: str) -> str | None:
	flag = flag.strip().lower().replace("-", "_")
	return ARGS_KNOBS.get(flag) or CONST_KNOBS.get(flag) or CLI_KNOBS.get(flag)


def commands_in(text: str) -> list[str]:
	"""Launch commands in fenced shell blocks (```bash, ```sh, ```shell, ```), continuations joined."""
	out: list[str] = []
	for block in re.findall(r"```(?:bash|sh|shell|console)?\n(.*?)```", text, flags=re.S):
		joined = re.sub(r"\\\s*\n", " ", block)
		for line in joined.splitlines():
			line = line.strip().lstrip("$ ").strip()
			if (line.split(" ", 1)[0].split("=", 1)[0] in RUNNERS and ".py" in line) or line.startswith(
				"trl "
			):
				out.append(line)
	return out


def flags_in(command: str) -> list[str]:
	try:
		parts = shlex.split(command, comments=True)
	except ValueError:
		parts = command.split()
	flags: list[str] = []
	for p in parts:
		if p.startswith("--") and len(p) > 2:
			flags.append(p[2:].split("=", 1)[0])
	return flags


def assigned_knobs(text: str) -> set[str]:
	"""Module-level `name = value` lines (nanoGPT's override configs)."""
	return {
		k
		for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", text, flags=re.M)
		if (k := knob_for(m.group(1)))
	}


def literal_flags(code: str) -> list[str]:
	"""Flags an example declares as string literals: argparse options and built command lines."""
	return re.findall(r"[\"']--([A-Za-z][\w-]*)[\"']", code)


def python_knobs(path: Path) -> set[str]:
	"""What A's static reader finds set in the code, the flags the code declares, and the
	docstring's launch commands."""
	text = path.read_text(errors="ignore")
	try:
		found = {k.id for k in map_static({path.name: text}, entrypoint=path.name).knobs}
	except (ScriptParseError, SyntaxError, ValueError):
		found = set()
	found |= {k for f in literal_flags(text) if (k := knob_for(f))}
	for cmd in commands_in(text):
		found |= command_knobs(cmd)
	return found


def command_knobs(command: str) -> set[str]:
	return {k for f in flags_in(command) if (k := knob_for(f))}


def script_of(command: str) -> str:
	m = re.search(r"([\w./-]+\.py)", command)
	return Path(m.group(1)).name if m else "?"


def rank(units: list[Unit]) -> dict[str, dict]:
	out: dict[str, dict] = {}
	for lens in sorted({u.lens for u in units}):
		mine = [u for u in units if u.lens == lens]
		counts = Counter(k for u in mine for k in u.knobs)
		ranking = sorted(counts, key=lambda k: (-counts[k], k))
		out[lens] = {
			"units": len(mine),
			"counts": {k: counts[k] for k in ranking},
			"ranking": ranking,
			"units_list": sorted(u.name for u in mine),
		}
	return out


# ------------------------------------------------------------------ the sample


TRANSFORMERS_LENS = {
	"text-classification": "finetune",
	"token-classification": "finetune",
	"question-answering": "finetune",
	"summarization": "finetune",
	"translation": "finetune",
	"multiple-choice": "finetune",
	"image-classification": "vision",
	"image-pretraining": "vision",
	"semantic-segmentation": "vision",
	"object-detection": "vision",
	"instance-segmentation": "vision",
	"contrastive-image-text": "vision",
}


def trl_lens(name: str) -> str | None:
	stem = name.removesuffix(".py")
	if stem.startswith(("grpo", "gspo", "rloo", "async_grpo")):
		return "rl"
	if stem.startswith("sft"):
		return "finetune"
	return None


PEFT_VISION = ("image_classification", "semantic_segmentation")
PEFT_SKIP = ("dreambooth", "stable_diffusion", "controlnet", "dna_language_models")


def _add(units: dict[tuple[str, str], set[str]], name: str, lens: str, knobs: set[str]) -> None:
	units.setdefault((name, lens), set()).update(knobs)


def collect(root: Path, zoo: Path) -> list[Unit]:
	units: dict[tuple[str, str], set[str]] = {}
	for script in sorted(zoo.glob("*/train.py")):
		sm = map_static({script.name: script.read_text()}, entrypoint=script.name)
		_add(units, f"zoo/{script.parent.name}/train.py", sm.lens, {k.id for k in sm.knobs})
	trl = root / "trl"
	for path in sorted([*(trl / "trl/scripts").glob("*.py"), *(trl / "examples").rglob("*.py")]):
		lens = trl_lens(path.name)
		if lens:
			_add(units, f"trl/{path.relative_to(trl)}", lens, python_knobs(path))
	for task, lens in TRANSFORMERS_LENS.items():
		readme = root / "tf/examples/pytorch" / task / "README.md"
		for cmd in commands_in(readme.read_text()):
			_add(units, f"transformers/examples/pytorch/{task}/{script_of(cmd)}", lens, command_knobs(cmd))
	lm = root / "tf/examples/pytorch/language-modeling/README.md"
	for cmd in commands_in(lm.read_text()):
		flags = set(flags_in(cmd))
		lens = "finetune" if "model_name_or_path" in flags else "pretrain"
		_add(
			units,
			f"transformers/examples/pytorch/language-modeling/{script_of(cmd)}",
			lens,
			command_knobs(cmd),
		)
	for path in sorted((root / "peft/examples").rglob("*")):
		if path.suffix not in (".md", ".sh") or any(s in str(path) for s in PEFT_SKIP):
			continue
		lens = "vision" if any(s in str(path) for s in PEFT_VISION) else "finetune"
		text = path.read_text(errors="ignore")
		cmds = commands_in(text) if path.suffix == ".md" else commands_in(f"```bash\n{text}\n```")
		for cmd in cmds:
			rel = path.relative_to(root / "peft").parent
			_add(units, f"peft/{rel}/{script_of(cmd)}", lens, command_knobs(cmd))
	for path in sorted((root / "nanoGPT/config").glob("*.py")):
		lens = (
			"pretrain"
			if path.name.startswith("train")
			else "finetune"
			if path.name.startswith("finetune")
			else None
		)
		if lens:
			_add(units, f"nanoGPT/config/{path.name}", lens, assigned_knobs(path.read_text()))
	return [Unit(n, lens, frozenset(k)) for (n, lens), k in units.items() if k]


def main(argv: list[str]) -> None:
	root, out = Path(argv[1]), Path(argv[2])
	zoo = Path(__file__).resolve().parents[3] / "zoo"
	sources = json.loads((root / "sources.json").read_text())
	census = {
		"about": "Settings set per example, by lens; built by python/preflight/addons/knob_census.py",
		"sources": sources,
		"lenses": rank(collect(root, zoo)),
	}
	out.write_text(json.dumps(census, indent="\t") + "\n")


if __name__ == "__main__":
	main(sys.argv)
