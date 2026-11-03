"""probes.eval_units (off; Nov 2, after freeze): lm_eval's scoring tasks make no generate call and no optimizer
step, so an eval probe measured nothing; a forward under no_grad is now a unit, and an eval that finishes after
measured generate calls is kept. Also probes.repo_install's script folder on the probe's path (OGB arxiv)."""

import pytest

pytest.importorskip("torch")

from preflight.probes.local import ProbeSpec, run_probe

pytestmark = pytest.mark.torch

# a stand-in transformers: the recorder hooks it by module name, as it does the real one
FAKE_TF = {
	"transformers/__init__.py": "",
	"transformers/modeling_utils.py": "import torch\n\nclass PreTrainedModel(torch.nn.Module):\n    pass\n",
	"transformers/generation/__init__.py": "",
	"transformers/generation/utils.py": (
		"import torch\n\nclass GenerationMixin:\n"
		"    def generate(self, input_ids=None, max_length=None, **kw):\n"
		"        new = torch.full((input_ids.shape[0], 5), 7, dtype=torch.long)\n"
		"        return torch.cat([input_ids, new], dim=1)\n"
	),
}
SCORE = """
import torch
from transformers.modeling_utils import PreTrainedModel

class M(PreTrainedModel):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32, 8)
    def forward(self, ids):
        return self.emb(ids).sum(-1)

m = M()
with torch.no_grad():
    for _ in range(50):  # thousands in lm_eval: the probe stops after a few
        m(torch.ones(4, 16, dtype=torch.long))
print("scored")
"""
GENERATE_TWICE = """
import torch
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

class M(PreTrainedModel, GenerationMixin):
    def forward(self, ids):
        return ids

m = M()
for _ in range(2):  # gsm8k at --limit 16, batch 8: two calls and the script is done
    m.generate(input_ids=torch.ones(8, 4, dtype=torch.long), max_length=64)
"""


def _probe(script, **kw):
	return run_probe(ProbeSpec(probe_id="p1", files={"run.py": script, **FAKE_TF}, entrypoint="run.py", warmup=1,
		timed=2, timeout_s=120, generate_calls=2, **kw)).probe  # fmt: skip


def test_scoring_batches_are_units_and_the_probe_stops_after_them():
	p = _probe(SCORE, eval_units=3)
	assert p.status == "ok", p.error
	assert p.extras["eval_forward_batches"] == 3.0 and p.extras["eval_forward_tokens_mean"] == 64.0
	assert p.extras["eval_forward_rows_mean"] == 4.0 and p.extras["eval_forward_s_p50"] >= 0


def test_without_the_flag_scoring_measures_nothing_as_before():
	p = _probe(SCORE)
	assert p.status == "error" and p.error_code == "too_few_steps"
	assert "eval_forward_batches" not in p.extras


def test_an_eval_that_finishes_after_measured_generate_calls_is_kept():
	on = _probe(GENERATE_TWICE, eval_units=8)
	assert on.status == "ok", on.error
	assert on.generation.calls == 2 and on.generation.sequences == 16 and on.generation.new_tokens_mean == 5.0
	off = _probe(GENERATE_TWICE)
	assert off.status == "error" and off.error_code == "too_few_steps"


def test_with_repo_install_the_scripts_own_folder_is_on_its_path():
	files = {
		"examples/arxiv/gnn.py": (
			"import torch\nimport logger\nm = torch.nn.Linear(2, 2)\nopt = torch.optim.SGD(m.parameters(), 0.1)\n"
			"for _ in range(8):\n    opt.zero_grad()\n    m(torch.ones(1, 2)).sum().backward()\n    opt.step()\n"
		),
		"examples/arxiv/logger.py": "X = 1\n",
	}
	spec = {"probe_id": "p1", "files": files, "entrypoint": "examples/arxiv/gnn.py", "warmup": 1, "timed": 2,
		"timeout_s": 120}  # fmt: skip
	on = run_probe(ProbeSpec(**spec, install_repo=True)).probe
	assert on.status == "ok", on.error
	off = run_probe(ProbeSpec(**spec)).probe
	assert off.status == "error" and off.error_code == "missing_package"  # as before: logger isn't found
