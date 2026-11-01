"""runner/measured_rules.py on the first measured agent record (zoo/agent_eval_spark, Oct 31)."""

from pathlib import Path

from preflight.runner.measured_rules import agent_findings
from preflight.schema import load_record

REC = Path(__file__).resolve().parents[1] / "zoo" / "agent_eval_spark" / "record_oct31.json"


def test_the_agent_record_shows_no_caching_and_an_overflow_at_turn_11():
	rec = load_record(REC.read_text())
	fs = {f.code: f for f in agent_findings(rec)}
	assert "0 of 246,463 input tokens" in fs["agent.no_prompt_caching"].detail
	ov = fs["agent.context_overflow"]
	assert ov.severity == "warn" and "turn 11" in ov.title
	assert {e.metric: e.value for e in ov.evidence}["agent.context_growth_tokens_per_turn"] == 2745


def test_a_cached_sample_raises_no_caching_finding():
	rec = load_record(REC.read_text())
	tasks = [t.model_copy(update={"cached_input_tokens": 100}) for t in rec.agent_sample.tasks]
	rec = rec.model_copy(update={"agent_sample": rec.agent_sample.model_copy(update={"tasks": tasks})})
	assert "agent.no_prompt_caching" not in {f.code for f in agent_findings(rec)}


def test_no_projection_without_a_limit_or_growth():
	rec = load_record(REC.read_text())
	no_limit = rec.model_copy(update={"knobs": [k for k in rec.knobs if k.id != "context_limit"]})
	assert "agent.context_overflow" not in {f.code for f in agent_findings(no_limit)}
	flat = [
		t.model_copy(update={"context_tokens_by_turn": [500] * max(1, t.turns)})
		for t in rec.agent_sample.tasks
	]
	flat_rec = rec.model_copy(update={"agent_sample": rec.agent_sample.model_copy(update={"tasks": flat})})
	assert "agent.context_overflow" not in {f.code for f in agent_findings(flat_rec)}


# ---------------------------------------------------------------- run hygiene

from preflight.runner.measured_rules import hygiene_findings  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "schema" / "examples" / "finetune.json"


def _gpu_rec(**probe_update):
	rec = load_record(GOLDEN.read_text())
	p = rec.probes[0].model_copy(update=probe_update)
	return rec.model_copy(update={"probes": [p]})


def test_old_checkpoints_shipped_with_the_code_are_flagged():
	files = {"train.py": "print(1)\n", "out/checkpoint-500/trainer_state.json": "{}"}
	codes = [f.code for f in hygiene_findings(_gpu_rec(), files)]
	assert "hygiene.stale_checkpoint" in codes
	assert "hygiene.stale_checkpoint" not in [
		f.code for f in hygiene_findings(_gpu_rec(), {"train.py": "x=1\n"})
	]


def test_a_cache_loaded_from_a_fixed_path_is_flagged_and_a_keyed_one_isnt():
	fixed = "import os, torch\nCACHE = 'tokens.pt'\nif os.path.exists(CACHE):\n    data = torch.load(CACHE)\n"
	keyed = "import os, torch\nmodel = 'gpt2'\nif os.path.exists(f'tokens-{model}.pt'):\n    data = torch.load('x')\n"
	assert "hygiene.cache_not_keyed" in [f.code for f in hygiene_findings(_gpu_rec(), {"train.py": fixed})]
	assert "hygiene.cache_not_keyed" not in [
		f.code for f in hygiene_findings(_gpu_rec(), {"train.py": keyed})
	]


def test_truncated_generations_carry_a_wilson_interval():
	from preflight.schema import GenerationStats

	g = GenerationStats(calls=4, sequences=40, truncation_rate=0.25, max_new_tokens=512, backend="hf")
	f = next(f for f in hygiene_findings(_gpu_rec(generation=g), {}) if f.code == "generation.truncated")
	assert "10 of 40" in f.detail and "95% interval" in f.detail and f.knob_id == "max_new_tokens"


def test_identical_outputs_from_different_weights_are_flagged():
	rec = _gpu_rec(extras={"outputs_compared": 3.0, "outputs_identical": 3.0})
	assert "hygiene.identical_outputs" in [f.code for f in hygiene_findings(rec, {})]
	rec = _gpu_rec(extras={"outputs_compared": 3.0, "outputs_identical": 1.0})
	assert "hygiene.identical_outputs" not in [f.code for f in hygiene_findings(rec, {})]


def test_a_weights_only_checkpoint_is_a_resume_that_wont_match():
	rec = _gpu_rec(extras={"checkpoint_saves_seen": 4.0, "checkpoint_has_optimizer": 0.0,
		"checkpoint_has_scheduler": 0.0, "checkpoint_has_rng": 0.0, "checkpoint_has_step": 0.0})  # fmt: skip
	f = [x for x in hygiene_findings(rec, {}) if x.code == "hygiene.resume_mismatch"]
	assert len(f) == 1 and f[0].severity == "warn" and "optimizer state" in f[0].detail
	full = {
		k: 1.0
		for k in (
			"checkpoint_has_optimizer",
			"checkpoint_has_scheduler",
			"checkpoint_has_rng",
			"checkpoint_has_step",
		)
	}
	rec = _gpu_rec(extras={"checkpoint_saves_seen": 4.0, **full})
	assert not [x for x in hygiene_findings(rec, {}) if x.code == "hygiene.resume_mismatch"]


def test_code_in_a_ckpt_package_and_a_config_loaded_when_present_are_not_flagged():
	files = {
		"ckpt/__init__.py": "x = 1\n",
		"train.py": "import os, json\nif os.path.exists('config.json'):\n    c = json.load(open('config.json'))\n",
	}
	codes = [f.code for f in hygiene_findings(_gpu_rec(), files)]
	assert "hygiene.stale_checkpoint" not in codes and "hygiene.cache_not_keyed" not in codes
	saved = "import os, torch\nP = 'feats'\nif os.path.exists(P):\n    x = torch.load(P)\nelse:\n    torch.save(1, P)\n"
	assert "hygiene.cache_not_keyed" in [f.code for f in hygiene_findings(_gpu_rec(), {"train.py": saved})]


def test_a_cache_finding_on_a_graph_without_a_data_node_still_loads():
	from preflight.schema import load_record

	rec = _gpu_rec()
	keep = {n.id for n in rec.graph.nodes if n.kind != "data"}
	g = rec.graph.model_copy(update={"nodes": [n for n in rec.graph.nodes if n.id in keep],
		"edges": [e for e in rec.graph.edges if e.source in keep and e.target in keep]})  # fmt: skip
	knobs = [k.model_copy(update={"affects": [a for a in k.affects if a in keep]}) for k in rec.knobs]
	rec = rec.model_copy(update={"graph": g, "knobs": knobs, "findings": []})
	files = {
		"train.py": "import os, torch\nif os.path.exists('tokens.pt'):\n    d = torch.load('tokens.pt')\n"
	}
	f = next(f for f in hygiene_findings(rec, files) if f.code == "hygiene.cache_not_keyed")
	assert f.node_id is None
	load_record(rec.model_copy(update={"findings": [f]}).model_dump_json())


def test_a_lightning_checkpoint_folder_is_stale_and_a_resume_is_not_a_cache():
	files = {"train.py": "x = 1\n", "checkpoints/hparams.yaml": "lr: 0.1\n"}
	assert "hygiene.stale_checkpoint" in [f.code for f in hygiene_findings(_gpu_rec(), files)]
	resume = ("import os, torch\nif os.path.exists('state.pt'):\n    model.load_state_dict(torch.load('state.pt'))\n"
		"torch.save(model.state_dict(), 'state.pt')\n")  # fmt: skip
	named = "import os, torch\nif os.path.exists('checkpoints/last.pt'):\n    s = torch.load('checkpoints/last.pt')\n"
	for code in (resume, named):
		assert "hygiene.cache_not_keyed" not in [
			f.code for f in hygiene_findings(_gpu_rec(), {"train.py": code})
		]


def test_truncation_counts_only_the_sequences_whose_ending_is_known():
	from preflight.schema import GenerationStats

	g = GenerationStats(calls=12, sequences=12, truncation_rate=1.0, max_new_tokens=64, backend="hf")
	rec = _gpu_rec(generation=g, extras={"generation_ending_known": 2.0})
	f = next(f for f in hygiene_findings(rec, {}) if f.code == "generation.truncated")
	assert "2 of 2" in f.detail


def test_caches_with_model_or_last_in_their_names_and_beside_a_weights_load_still_fire():
	for code in ("import os, torch\nif os.path.exists('model_embeddings_cache.pt'):\n    e = torch.load('model_embeddings_cache.pt')\n",
		"import os, pickle\nif os.path.exists('cache/elastic_index.pkl'):\n    i = pickle.load(open('cache/elastic_index.pkl', 'rb'))\n",
		"import os, torch\nif os.path.exists('feats.pt'):\n    f = torch.load('feats.pt')\n    model.load_state_dict(torch.load('init.pt'))\n"):  # fmt: skip
		assert "hygiene.cache_not_keyed" in [
			f.code for f in hygiene_findings(_gpu_rec(), {"train.py": code})
		], code
	readme = {"train.py": "x = 1\n", "checkpoints/README.md": "download weights here\n"}
	assert "hygiene.stale_checkpoint" not in [f.code for f in hygiene_findings(_gpu_rec(), readme)]


def test_a_two_step_resume_and_a_keyword_load_are_not_caches():
	two_step = ("import os, torch\nP = 'out/model.pt'\nif os.path.exists(P):\n    sd = torch.load(P, map_location='cpu')\n"
		"    model.load_state_dict(sd['model'])\ntorch.save({'model': model.state_dict()}, P)\n")  # fmt: skip
	in_folder = "import os, torch\nif os.path.exists('checkpoints/latest.pt'):\n    s = torch.load('checkpoints/latest.pt')\n"
	keyword = ("import os\nif os.path.exists('enc.pt'):\n"
		"    m = AutoModel.from_pretrained(pretrained_model_name_or_path='enc.pt')\n")  # fmt: skip
	for code in (two_step, in_folder, keyword):
		assert "hygiene.cache_not_keyed" not in [
			f.code for f in hygiene_findings(_gpu_rec(), {"train.py": code})
		], code


def test_a_cache_used_only_for_a_setting_of_a_weights_load_still_fires():
	code = ("import os, torch\nif os.path.exists('tokens.pt'):\n    tokens = torch.load('tokens.pt')\n"
		"    m = AutoModel.from_pretrained('gpt2', vocab_size=len(tokens))\n")  # fmt: skip
	assert "hygiene.cache_not_keyed" in [f.code for f in hygiene_findings(_gpu_rec(), {"train.py": code})]
	direct = "import os, torch\nP = 'out/state.pt'\nif os.path.exists(P):\n    model.load_state_dict(torch.load(P))\n"
	assert "hygiene.cache_not_keyed" not in [
		f.code for f in hygiene_findings(_gpu_rec(), {"train.py": direct})
	]
