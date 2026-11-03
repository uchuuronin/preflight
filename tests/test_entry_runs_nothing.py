"""runner.entry_runs_nothing (off; D's walkthrough, Nov 3): an entry file that runs nothing is refused before any GPU
time with C's code entry.runs_nothing."""

import pytest

from preflight.runner.sources import runs_nothing

LITGPT_LIKE = '''"""Pretrain a model."""
import torch
from typing import Optional
try:
    import lightning as L
except ImportError:
    L = None
DEFAULTS = {"lr": 4e-4}
def setup(model_name: Optional[str] = None):
    fit()
def fit():
    pass
class Cfg:
    n = 1
'''


@pytest.mark.parametrize("code", [LITGPT_LIKE, "def main():\n    pass\n", "import torch\nx: int = 3\n"])
def test_files_that_only_define_things_run_nothing(code):
	assert runs_nothing(code)


@pytest.mark.parametrize(
	"code",
	[
		LITGPT_LIKE + 'if __name__ == "__main__":\n    setup()\n',
		"import torch\ntrain()\n",
		"model = build()\n",
		"for i in range(3):\n    pass\n",
		"with open('x') as f:\n    pass\n",
		"import fire\nfire.Fire(main)\n",
		"x = (\n",  # doesn't parse: the map says so, not this check
	],
)
def test_anything_that_runs_or_might_isnt_refused(code):
	assert not runs_nothing(code)


def test_the_pipeline_refuses_it_only_with_the_flag(tmp_path, monkeypatch):
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).parent))
	import preflight.runner.pipeline as P
	from preflight.probes.local import InvalidSubmission
	from preflight.runner.pipeline import _Ctx, run_pipeline
	from preflight.runner.worker import error_code
	from test_pipeline import FakeRunner, deps

	src = {"source": {"kind": "upload", "files": {"pretrain.py": LITGPT_LIKE}, "entrypoint": "pretrain.py"}}
	real = P._flag
	monkeypatch.setattr(P, "_flag", lambda n: n == "runner.entry_runs_nothing" or real(n))
	runner = FakeRunner()
	with pytest.raises(InvalidSubmission) as exc:
		run_pipeline(src, _Ctx(), deps(tmp_path, runner))
	assert error_code(exc.value) == "entry.runs_nothing" and "defines functions but runs nothing" in str(
		exc.value
	)
	assert not getattr(runner, "calls", None)  # no probe ran
	monkeypatch.setattr(P, "_flag", real)
	try:  # off: no refusal of this kind
		run_pipeline(src, _Ctx(), deps(tmp_path, FakeRunner()))
	except InvalidSubmission as e:
		assert error_code(e) != "entry.runs_nothing"
	except Exception:
		pass


@pytest.mark.parametrize("code", ["from experiments.run_cifar import *\n",
	"import atexit\n@atexit.register\ndef main():\n    pass\n"])  # fmt: skip
def test_a_star_import_or_a_decorated_definition_might_run(code):
	assert not runs_nothing(code)  # review, Nov 3: never refuse a maybe
