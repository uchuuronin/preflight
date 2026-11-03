"""harness.map_closure before and after: each source fetched as production reads it, mapped with the flag off, then on."""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
prod = json.loads((ROOT / "flags.production.json").read_text())
prod = prod.get("flags", prod) if isinstance(prod, dict) else prod


def setflags(extra):
	from preflight.runner import flags as F

	os.environ["PREFLIGHT_FLAGS"] = json.dumps(
		{**{k: v for k, v in prod.items() if isinstance(v, bool)}, **extra}
	)
	F.reload()


UNSEEN = [
	(
		"unseen-1 torchtune",
		"pytorch/torchtune",
		"bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1",
		"recipes/lora_finetune_single_device.py",
	),
	(
		"unseen-2 litgpt",
		"Lightning-AI/litgpt",
		"4738993a5e0190b851162a52173729827d3884e7",
		"litgpt/pretrain.py",
	),
	(
		"unseen-3 open-r1",
		"huggingface/open-r1",
		"5b6ff22b3fb7aa069c54866e517f39dfc3160e09",
		"src/open_r1/grpo.py",
	),
	(
		"unseen-4 smolagents",
		"huggingface/smolagents",
		"96f33faaf028479119ec8d34507b47694cf14e34",
		"examples/agent_from_any_llm.py",
	),
	(
		"unseen-5 lm-eval",
		"EleutherAI/lm-evaluation-harness",
		"ddd67220430a2470529f25fd5c05a576ca1057a0",
		"lm_eval/__main__.py",
	),
	(
		"unseen-6 vllm bench",
		"vllm-project/vllm",
		"5df63a7203cdb9328d824a27b682e77c7b9928be",
		"vllm/benchmarks/throughput.py",
	),
	(
		"unseen-7 beir",
		"beir-cellar/beir",
		"ef83d29307061c65d04b035b4f4e7c18bd8374af",
		"examples/retrieval/evaluation/dense/evaluate_sbert.py",
	),
	(
		"unseen-8 ogb",
		"snap-stanford/ogb",
		"61e9784ca76edeaa6e259ba0f836099608ff0586",
		"examples/nodeproppred/arxiv/gnn.py",
	),
	(
		"unseen-9 airbench",
		"KellerJordan/cifar10-airbench",
		"4c1b6d1e3889b037efadcfd5c0ea65b246592362",
		"airbench94_muon.py",
	),
	(
		"F1 word_language_model",
		"pytorch/examples",
		"acc295dc7b90714f1bf47f06004fc19a7fe235c4",
		"word_language_model/main.py",
	),
	("F3 pytorch-cifar", "kuangliu/pytorch-cifar", "49b7aa97b0c12fe0d4054e670403a16b6b834ddd", "main.py"),
]


def summary(sm):
	d = sm.model_dump(mode="json")
	model = d.get("model") or {}
	loop = d.get("loop") or {}
	return {
		"lens": d.get("lens"),
		"framework": d.get("framework"),
		"model": {k: model.get(k) for k in ("name", "hf_id", "kind", "params") if model.get(k) is not None},
		"loop": {k: v for k, v in loop.items() if v not in (None, [], {})}
		if isinstance(loop, dict)
		else loop,
		"knobs": {k["id"]: k["current"] for k in d.get("knobs") or []},
		"_where": {k["id"]: (k.get("location") or {}).get("file") for k in d.get("knobs") or []},
	}


def compare(name, fetched):
	from preflight.harness.static import map_static

	out = {"name": name, "entry": fetched.entrypoint, "files": len(fetched.files)}
	for label, on in (("before", False), ("after", True)):
		setflags({"harness.map_closure": on})
		try:
			t = time.monotonic()
			out[label] = summary(map_static(fetched.files, entrypoint=fetched.entrypoint))
			out[label + "_s"] = round(time.monotonic() - t, 2)
		except Exception as exc:
			out[label] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

	def strip(d):
		return {k: v for k, v in d.items() if k != "_where"} if isinstance(d, dict) else d

	out["same"] = strip(out["before"]) == strip(out["after"])
	return out


def main(which):
	from preflight.runner.sources import fetch

	res = []
	setflags({})
	if which in ("zoo", "all"):
		for d in sorted((ROOT / "zoo").iterdir()):
			if not d.is_dir():
				continue
			from preflight.recorder.ground_truth import entrypoint_of
			from preflight.runner.sources import _read_tree

			files = _read_tree(d)
			try:
				entry = entrypoint_of(files, None)
			except Exception:
				nb = [p.name for p in d.glob("*.ipynb")]
				files = {n: (d / n).read_text() for n in nb}
				if not nb:
					res.append({"name": f"zoo/{d.name}", "error": "no entry"})
					continue
				setflags({})
				fetched = fetch({"kind": "notebook", "files": {nb[0]: files[nb[0]]}})
				res.append(compare(f"zoo/{d.name}", fetched))
				continue
			Fet = type("F", (), {})()
			Fet.files, Fet.entrypoint = files, entry
			res.append(compare(f"zoo/{d.name}", Fet))
	if which in ("held", "all"):
		for name, repo, commit, entry in UNSEEN:
			setflags({})
			try:
				fetched = fetch(
					{
						"kind": "repo",
						"uri": f"https://github.com/{repo}",
						"commit": commit,
						"entrypoint": entry,
					}
				)
			except Exception as exc:
				res.append({"name": name, "error": f"fetch: {type(exc).__name__}: {str(exc)[:200]}"})
				continue
			res.append(compare(name, fetched))
		nb = ROOT / "validation/fresh/f2/text_classification.ipynb"
		setflags({})
		res.append(
			compare(
				"F2 text_classification (notebook)",
				fetch({"kind": "notebook", "files": {nb.name: nb.read_text()}}),
			)
		)
	print(json.dumps(res, indent=1))


main(sys.argv[1])
