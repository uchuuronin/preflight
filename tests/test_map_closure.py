"""harness.map_closure (off; the re-validation, Nov 3): the map from the entry point's import closure, not every file."""

import json
import os
from pathlib import Path

import pytest

from preflight.harness.static import map_static
from preflight.runner import flags as F

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def closure(monkeypatch):
	def on(v: bool):
		monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.map_closure": v}))
		F.reload()

	yield on
	monkeypatch.delenv("PREFLIGHT_FLAGS", raising=False)
	F.reload()


def _same(sm):
	d = sm.model_dump(mode="json")
	return {k: d.get(k) for k in ("lens", "framework", "model", "loop", "knobs")}


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "zoo").iterdir() if (p / "train.py").is_file()
	or (p / "run.py").is_file() or (p / "serve.py").is_file()))  # fmt: skip
def test_every_zoo_sample_maps_the_same(name, closure):
	from preflight.recorder.ground_truth import entrypoint_of
	from preflight.runner.sources import _read_tree

	files = _read_tree(ROOT / "zoo" / name)
	entry = entrypoint_of(files, None)
	closure(False)
	before = _same(map_static(files, entrypoint=entry))
	closure(True)
	assert _same(map_static(files, entrypoint=entry)) == before


REPO = {
	"examples/word/main.py": "import torch\nfrom model import Net\nimport argparse\np = argparse.ArgumentParser()\n"
	"p.add_argument('--lr', type=float, default=20)\na = p.parse_args()\nm = Net()\n"
	"opt = torch.optim.SGD(m.parameters(), lr=a.lr)\nfor x in range(3):\n    m(x).sum().backward()\n    opt.step()\n",
	"examples/word/model.py": "import torch.nn as nn\nclass Net(nn.Module):\n    pass\n",
	"examples/other/t5.py": "from transformers import Trainer, AutoModelForSeq2SeqLM\n"
	"m = AutoModelForSeq2SeqLM.from_pretrained('t5-base')\nt = Trainer(model=m, learning_rate=1e-4)\nt.train()\n",
	"tests/test_x.py": "import lightning\ntrainer = lightning.Trainer(max_epochs=5)\ntrainer.fit(None)\n",
}


def test_a_repo_maps_from_what_the_entry_point_imports(closure):
	closure(False)
	before = map_static(REPO, entrypoint="examples/word/main.py")
	closure(True)
	after = map_static(REPO, entrypoint="examples/word/main.py")
	assert after.loop.location.file == "examples/word/main.py" and after.loop.kind == "plain"
	assert {k.id: k.location.file for k in after.knobs}.get("lr") == "examples/word/main.py"
	assert all(k.location.file.startswith("examples/word/") for k in after.knobs)
	assert after.model == before.model  # the model from the full read, as before (B's review, Nov 3)
	assert before.loop.location.file != "examples/word/main.py"  # before: another example's Trainer


def test_an_api_method_passed_not_called_is_an_agent_loop(closure):
	files = {"run.py": "from lib.models import M\nM().run()\n",
		"lib/__init__.py": "", "lib/models.py": "class M:\n    def run(self):\n"
		"        return self.retry(self.client.chat.completions.create, model='x')\n"}  # fmt: skip
	closure(True)
	sm = map_static(files, entrypoint="run.py")
	assert sm.framework == "openai_client" and sm.loop.kind == "agent_loop"
	assert sm.loop.location.file == "lib/models.py" and sm.loop.location.line == 3
	closure(False)
	assert map_static(files, entrypoint="run.py").framework != "openai_client"  # off: as before


def test_the_flag_is_off_by_default():
	assert json.loads((ROOT / "flags.json").read_text())["harness.map_closure"] is False
	assert "harness.map_closure" not in os.environ.get("PREFLIGHT_FLAGS", "")


def test_a_docstring_that_mentions_an_api_call_changes_nothing(closure):
	files = {"train.py": "import timm, torch\nfrom utils import helper\nm = timm.create_model('resnet18')\n"
		"opt = torch.optim.SGD(m.parameters(), lr=0.1)\nfor x in range(3):\n    m(x).sum().backward()\n    opt.step()\n",
		"utils.py": '"""Unlike client.chat.completions.create(...), this helper is local."""\ndef helper():\n    pass\n'}  # fmt: skip
	closure(False)
	before = map_static(files, entrypoint="train.py")
	closure(True)
	after = map_static(files, entrypoint="train.py")
	assert after.framework == before.framework == "timm" and after.loop.kind == before.loop.kind


def test_the_model_and_the_rl_inputs_come_from_the_full_read(closure):
	"""B's review (Nov 3): LitGPT gained a model shape from a module it imports, and open-r1 lost vLLM and its group
	size, which other files set; with the flag on, those still come from the whole source, as before."""
	files = {"grpo.py": "from trl import GRPOTrainer, GRPOConfig\nfrom shapes import Cfg\n"
		"t = GRPOTrainer(model='Qwen/Qwen2.5-0.5B', reward_funcs=[f], args=GRPOConfig(learning_rate=1e-6))\nt.train()\n",
		"shapes.py": "class Cfg:\n    n_layer = 16\n",
		"gen.py": "import vllm\nfrom trl import GRPOConfig\nc = GRPOConfig(num_generations=4, use_vllm=True)\n"}  # fmt: skip
	closure(False)
	before = map_static(files, entrypoint="grpo.py")
	closure(True)
	after = map_static(files, entrypoint="grpo.py")
	assert after.model == before.model and after.rl == before.rl and after.lens == before.lens == "rl"
