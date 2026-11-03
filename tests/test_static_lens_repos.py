"""The lens comes from what the job imports, not every file of the repo (Nov 2, the unseen runs).

Two of the nine unseen scripts were read as serving jobs because their repos import vLLM somewhere: BEIR's
dense-retrieval example (its models package imports a vLLM embedder beside SentenceBERT) and a smolagents example (the
library's VLLMModel imports vLLM inside the class that uses it). The second sent an API agent to a GPU. These are
minimal copies of each repo's import pattern (pinned commits: beir-cellar/beir ef83d29, huggingface/smolagents
96f33fa).
"""

from preflight.harness.static import map_static

BEIR = {
	"examples/retrieval/evaluation/dense/evaluate_sbert.py": (
		"from beir import util\n"
		"from beir.retrieval import models\n"
		"from beir.retrieval.evaluation import EvaluateRetrieval\n"
		"from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES\n"
		"model = DRES(models.SentenceBERT('msmarco-distilbert-base-tas-b'), batch_size=256)\n"
		"retriever = EvaluateRetrieval(model, score_function='dot')\n"
	),
	"beir/__init__.py": "from .logging import LoggingHandler\n",
	"beir/logging.py": "import logging\n",
	"beir/util.py": "import requests\n",
	"beir/retrieval/__init__.py": "",
	"beir/retrieval/evaluation.py": "import pytrec_eval\n",
	"beir/retrieval/search/__init__.py": "",
	"beir/retrieval/search/dense/__init__.py": "from .exact_search import DenseRetrievalExactSearch\n",
	"beir/retrieval/search/dense/exact_search.py": "import torch\n",
	"beir/retrieval/models/__init__.py": "from .sentence_bert import SentenceBERT\nfrom .vllm import VLLMEmbed\n",
	"beir/retrieval/models/sentence_bert.py": "from sentence_transformers import SentenceTransformer\n",
	"beir/retrieval/models/vllm.py": "from vllm import LLM\n\nclass VLLMEmbed:\n\tdef __init__(self, m):\n\t\tself.llm = LLM(m)\n",
}

SMOLAGENTS = {
	"examples/agent_from_any_llm.py": (
		"import os\n"
		"from smolagents import CodeAgent, OpenAIServerModel, tool\n"
		"model = OpenAIServerModel(model_id='muse-spark-1.3', api_base=os.environ['META_API_BASE'],\n"
		"\tapi_key=os.environ['META_API_KEY'])\n"
		"agent = CodeAgent(tools=[], model=model)\n"
		"agent.run('What is 2 + 2?')\n"
	),
	"src/smolagents/__init__.py": "from .agents import *\nfrom .models import *\nfrom .tools import *\n",
	"src/smolagents/agents.py": "import json\n",
	"src/smolagents/tools.py": "import inspect\n",
	"src/smolagents/models.py": (
		"import json\n\n"
		"class OpenAIServerModel:\n"
		"\tdef __init__(self, model_id, api_base=None, api_key=None):\n"
		"\t\timport openai\n"
		"\t\tself.client = openai.OpenAI(base_url=api_base, api_key=api_key)\n\n"
		"\tdef generate(self, messages):\n"
		"\t\treturn self.client.chat.completions.create(model=self.model_id, messages=messages)\n\n"
		"class VLLMModel:\n"
		"\tdef __init__(self, model_id):\n"
		"\t\tfrom vllm import LLM\n"
		"\t\tself.model = LLM(model=model_id)\n"
	),
}


def test_beirs_dense_example_is_retrieval_not_serving():
	sm = map_static(BEIR, entrypoint="examples/retrieval/evaluation/dense/evaluate_sbert.py")
	assert sm.lens == "retrieval" and sm.framework == "sentence_transformers"


def test_a_smolagents_example_is_an_api_agent_not_a_gpu_job():
	sm = map_static(SMOLAGENTS, entrypoint="examples/agent_from_any_llm.py")
	assert sm.lens == "agent" and sm.framework == "openai_client"


def test_a_script_that_imports_vllm_itself_is_still_serving():
	files = {
		"bench.py": "from vllm import LLM, SamplingParams\nllm = LLM('m')\nllm.generate(['hi'], SamplingParams())\n"
	}
	assert map_static(files, entrypoint="bench.py").lens == "serving"


# airbench94 (KellerJordan/cifar10-airbench 4c1b6d1): CIFAR-10 with torchvision's transforms and its own conv net. It
# read as pretraining, so "No sequence packing" and "Every document is seen 8 times" fired on an image dataset.
AIRBENCH = {
	"airbench94_muon.py": (
		"import torch\nfrom torch import nn\nimport torchvision\nimport torchvision.transforms as T\n"
		"net = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU())\n"
		"opt = torch.optim.SGD(net.parameters(), lr=0.1)\n"
		"for epoch in range(8):\n\tloss = net(torch.randn(2000, 3, 32, 32)).mean()\n\tloss.backward()\n\topt.step()\n"
	),
}


def test_airbench_is_a_vision_job():
	assert map_static(AIRBENCH, entrypoint="airbench94_muon.py").lens == "vision"


def _findings(lens, packing, padding, epochs):
	from preflight.harness.skeleton import read_time_findings, skeleton_graph
	from preflight.schema.script_map import DataInfo, ScriptMap, TrainInfo

	sm = ScriptMap(lens=lens, framework="plain_pytorch", entrypoint="train.py",
		data=DataInfo(batch_size=2000, packing=packing, padding=padding), training=TrainInfo(epochs=epochs))  # fmt: skip
	return {f.code for f in read_time_findings(sm, skeleton_graph(sm))}


def test_packing_and_document_repetition_fire_only_on_text():
	# as Muse Spark mapped airbench: packing False, padding "none", 8 epochs
	assert not {"data.no_packing", "data.repetition"} & _findings("vision", False, "none", 8.0)
	assert "data.no_packing" not in _findings(
		"pretrain", False, "none", 1.0
	)  # nothing padded: nothing wasted
	assert {"data.no_packing", "data.repetition"} <= _findings("pretrain", False, "longest", 8.0)


def test_a_serving_repo_whose_own_package_wraps_vllm_is_serving():
	files = {
		"serve.py": "from mypkg import Server\nServer('m').run()\n",
		"mypkg/__init__.py": "from .server import Server\n",
		"mypkg/server.py": (
			"from transformers import AutoTokenizer\nfrom vllm import LLM\n\nclass Server:\n"
			"\tdef __init__(self, m):\n\t\tself.tok = AutoTokenizer.from_pretrained(m)\n\t\tself.llm = LLM(m)\n\n"
			"\tdef run(self):\n\t\treturn self.llm.generate(['hi'])\n"
		),
	}
	assert map_static(files, entrypoint="serve.py").lens == "serving"


def test_a_module_beside_the_entrypoint_is_found_first():
	files = {
		"examples/train.py": "from model import Net\nimport torch\nnet = Net()\n",
		"examples/model.py": "import torch_geometric\nfrom torch_geometric.nn import SAGEConv\n\nclass Net:\n\tpass\n",
	}
	assert map_static(files, entrypoint="examples/train.py").lens == "gnn"


def test_channels_last_is_read_from_its_use_not_from_a_comment_and_from_either_spelling():
	"""B, Nov 2: vision_timm's header names the finding, which hid it from the static reader, and Muse Spark's map
	writes "channels-last", which the rule didn't match: both made the zoo miss vision.no_channels_last."""
	from preflight.harness.skeleton import skeleton_record
	from preflight.harness.static import map_static
	from preflight.schema import Source

	body = (
		"import timm, torch\nfrom torchvision import transforms\n"
		"model = timm.create_model('resnet50')\nx = torch.randn(8, 3, 224, 224)\n"
	)
	for text, want in (
		('"""Planted: vision.no_channels_last."""\n' + body, True),
		(body + "model = model.to(memory_format=torch.channels_last)\n", False),
	):
		sm = map_static({"train.py": text}, entrypoint="train.py")
		assert sm.lens == "vision"
		codes = {f.code for f in skeleton_record(sm, Source(kind="upload")).findings}
		assert ("vision.no_channels_last" in codes) is want, text[:40]
	sm = map_static({"train.py": body}, entrypoint="train.py")
	sm = sm.model_copy(
		update={"risks": ["Model and inputs stay in default NCHW layout with no channels-last"]}
	)
	assert "vision.no_channels_last" in {f.code for f in skeleton_record(sm, Source(kind="upload")).findings}
	# a map that mentions the script's own channels-last loader isn't a missing format (airbench's Spark map)
	sm = sm.model_copy(
		update={"risks": ["custom CUDA GPU dataloader with channels_last fp16", "converted to NHWC"]}
	)
	assert "vision.no_channels_last" not in {
		f.code for f in skeleton_record(sm, Source(kind="upload")).findings
	}
