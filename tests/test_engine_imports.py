"""The engine stays importable in the browser (Review, Oct 27; contract: api.py imports only numpy,
scipy, scikit-learn and lightgbm). No I/O, no processes, no torch: Pyodide can't run them."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "python" / "preflight" / "engine"

STDLIB = {
	"__future__", "math", "re", "dataclasses", "typing", "collections", "functools", "itertools",
	"json", "hashlib", "enum", "statistics", "bisect", "warnings", "base64", "datetime",
}  # fmt: skip
THIRD_PARTY = {"numpy", "scipy", "sklearn", "lightgbm"}
OURS = {"preflight.engine", "preflight.schema"}


def _imports(path: Path) -> list[str]:
	out = []
	for node in ast.walk(ast.parse(path.read_text(), str(path))):
		if isinstance(node, ast.Import):
			out += [a.name for a in node.names]
		elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
			out.append(node.module)
	return out


def test_engine_imports_only_the_allowlist():
	bad = {}
	for f in sorted(ENGINE.rglob("*.py")):
		for mod in _imports(f):
			top = mod.split(".")[0]
			if top in STDLIB or top in THIRD_PARTY or any(mod == o or mod.startswith(o + ".") for o in OURS):
				continue
			bad.setdefault(str(f.relative_to(ROOT)), []).append(mod)
	assert not bad, bad


def test_importing_the_api_pulls_in_nothing_heavy():
	code = (
		"import sys; import preflight.engine.api; "
		"heavy = [m for m in ('torch', 'transformers', 'modal', 'requests', 'httpx') if m in sys.modules]; "
		"print(','.join(heavy))"
	)
	out = subprocess.run(
		[sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "python")}
	)
	assert out.returncode == 0, out.stderr
	assert out.stdout.strip() == ""
