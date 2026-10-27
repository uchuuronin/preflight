"""The engine calls the parity test compares between CPython and Pyodide (same code, both runtimes).

Kept free of pytest and of file I/O beyond reading the golden records, so it runs as is in Pyodide.
"""

import json

WHAT_IFS = {
	"finetune": [
		{},
		{"batch_size": 2},
		{"batch_size": 1, "eval_batch_size": 1, "eval_no_grad": True},
		{"gpu_type": "H100-SXM-80GB"},
		{"gpu_type": "MI300X"},
		{"precision": "fp32"},
		{"gpus": 8, "batch_size": 3},
	],
	"pretrain": [{}, {"precision": "bf16"}, {"batch_size": 64}, {"max_steps": 1000}],
	"agent": [{}, {"concurrency": 16}, {"max_turns": 2}],
}


def run(root: str) -> dict:
	from preflight.engine import api
	from preflight.schema import load_record
	from preflight.schema.prediction import Objective

	out = {}
	for name, cases in WHAT_IFS.items():
		with open(f"{root}/schema/examples/{name}.json") as f:
			rec = load_record(f.read())
		filled = api.fill_predictions(rec)
		out[name] = {
			"predictions": json.loads(filled.predictions.model_dump_json()),
			"models": [m.model_dump(mode="json") for m in filled.models],
			"what_ifs": {
				json.dumps(kv, sort_keys=True): json.loads(api.predict(rec, kv).model_dump_json())
				for kv in cases
			},
			"sensitivity": [
				r.model_dump(mode="json")
				for r in api.sensitivity(rec, Objective(kind="min_cost"), pinned=["precision"])
			],
		}
	return out
