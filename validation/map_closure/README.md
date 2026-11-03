# harness.map_closure: before and after (Nov 3, A; for B's review)

The static map with `harness.map_closure` off (before) and on (after), on every zoo sample and the 12 held-out scripts
(the nine unseen and the three fresh), each source fetched as production reads it (`flags.production.json`, so the
big repos already arrive as their closure under `runner.read_closure`). `compare.py` here reruns it:
`PYTHONPATH=python python validation/map_closure/compare.py all` (clones the repos at their pinned commits);
`results.json` is its output, with each knob's file under `_where`.

With the flag on, a repo whose Python fits the 2 MB budget is mapped from the entry point and the Python files it
imports (`harness/closure.py`), not every file; the other text files stay. Two changes go with it, both only with the
flag on: an API client's method passed rather than called (`retryer(self.client.chat.completions.create, ...)`,
read from the code's syntax tree, so a comment or docstring that mentions one doesn't count) counts as an API call, and the agent loop is placed there. Without it, smolagents read as an eval job once
another example's API call was no longer in the map.

**Zoo: all 11 map the same** (job type, framework, model, loop, knobs). **Held out: the job type is the same on all
12 except F1, which changes from finetune to pretrain (right: the first attempt's record noted the lens miss).** The loop
moves into the job's own code for litgpt, OGB, airbench and F1 (before: a test file, another example, a helper).
Every dropped knob was read from a file the job doesn't import (tests, other examples, other scripts), and every
changed value now comes from the job's own file.

**Not fixed by it (both before and after are wrong):** the model for litgpt (`stablelm-zephyr-3b` from a test, then
`LoRALayer` from `litgpt/lora.py`; the real one is named in its YAML config), BEIR (unicoil from another example, then
`SpladeNaver` from a module its package imports; the real one is a string argument) and F1 (t5-base from another
example, then `PositionalEncoding`, a class in its own model.py). lm-eval and torchtitan_debug have no loop either way.

| Source | Map | Job type | Loop | Model | Knobs |
| --- | --- | --- | --- | --- | --- |
| zoo/agent_eval_spark | same | agent → agent | agent_loop at `run.py:154` → agent_loop at `run.py:154` | none → none | 6 → 6 |
| zoo/eval_sweep_reasoning | same | eval → eval | eval_loop at `run.py:85` → eval_loop at `run.py:85` | none → none | 4 → 4 |
| zoo/finetune_lora_hf | same | finetune → finetune | hf_trainer at `train.py:109` → hf_trainer at `train.py:109` | Qwen/Qwen2.5-1.5B → Qwen/Qwen2.5-1.5B | 14 → 14 |
| zoo/gnn_pyg | same | gnn → gnn | plain at `train.py:131` → plain at `train.py:131` | SAGEConv → SAGEConv | 6 → 6 |
| zoo/grpo_trl | same | rl → rl | trl_trainer at `train.py:116` → trl_trainer at `train.py:116` | Qwen/Qwen2.5-0.5B-Instruct → Qwen/Qwen2.5-0.5B-Instruct | 11 → 11 |
| zoo/notebook_tiny | same | pretrain → pretrain | plain at `tiny_train.py:20` → plain at `tiny_train.py:20` | none → none | 3 → 3 |
| zoo/pretrain_nanogpt | same | pretrain → pretrain | plain at `train.py:156` → plain at `train.py:156` | GPT → GPT | 13 → 13 |
| zoo/retrieval_st | same | retrieval → retrieval | hf_trainer at `train.py:69` → hf_trainer at `train.py:69` | sentence-transformers/all-MiniLM-L6-v2 → sentence-transformers/all-MiniLM-L6-v2 | 6 → 6 |
| zoo/serving_vllm | same | serving → serving | eval_loop at `serve.py:66` → eval_loop at `serve.py:66` | Qwen/Qwen2.5-7B-Instruct → Qwen/Qwen2.5-7B-Instruct | 6 → 6 |
| zoo/torchtitan_debug | same | pretrain → pretrain | none → none | none → none | 0 → 0 |
| zoo/vision_timm | same | vision → vision | plain at `train.py:96` → plain at `train.py:96` | resnet50.a1_in1k → resnet50.a1_in1k | 5 → 5 |
| unseen-1 torchtune | same | finetune → finetune | plain at `recipes/lora_finetune_single_device.py:652` → plain at `recipes/lora_finetune_single_device.py:652` | Qwen3Attention → Qwen3Attention | 2 → 2 |
| unseen-2 litgpt | differs | pretrain → pretrain | lightning at `tests/test_trainer_support.py:66` → plain at `litgpt/pretrain.py:361` | stabilityai/stablelm-zephyr-3b → LoRALayer | 20 → 12; dropped attn, epochs, eval_every, lr, max_new_tokens, temperature, tensor_parallel, weight_decay; now from the job's own file: checkpoint_every 200→1000, lora_alpha 8→1, lora_dropout 0.1→0.0, num_workers 4→8 |
| unseen-3 open-r1 | differs | rl → rl | trl_trainer at `src/open_r1/grpo.py:132` → trl_trainer at `src/open_r1/grpo.py:132` | none → none | 5 → 1; dropped batch_size, group_size, max_new_tokens, temperature |
| unseen-4 smolagents | differs | agent → agent | agent_loop at `examples/open_deep_research/scripts/mdconvert.py:756` → agent_loop at `src/smolagents/models.py:1611` | none → none | 3 → 1; dropped concurrency, temperature |
| unseen-5 lm-eval | same | eval → eval | none → none | none → none | 0 → 0 |
| unseen-6 vllm bench | same | serving → serving | agent_loop at `vllm/benchmarks/throughput.py:289` → agent_loop at `vllm/benchmarks/throughput.py:289` | Qwen/Qwen3-0.6B → Qwen/Qwen3-0.6B | 4 → 4 |
| unseen-7 beir | differs | retrieval → retrieval | other at `beir/retrieval/models/tldr.py:61` → other at `beir/retrieval/models/tldr.py:61` | castorini/unicoil-d2q-msmarco-passage → SpladeNaver | 7 → 2; dropped attn, encode_batch_size, epochs, temperature, warmup_steps; now from the job's own file: batch_size 80→128 |
| unseen-8 ogb | differs | gnn → gnn | hf_trainer at `examples/lsc/mag240m/gnn.py:249` → plain at `examples/nodeproppred/arxiv/gnn.py:86` | GCNConv → GCNConv | 9 → 3; dropped batch_size, eval_batch_size, eval_every, max_steps, num_workers, seq_len; now from the job's own file: epochs 25→500, lr 0.0001→0.01 |
| unseen-9 airbench | differs | vision → vision | plain at `airbench/utils.py:317` → plain at `airbench94_muon.py:405` | CifarNet → CifarNet | 2 → 2 |
| F1 word_language_model | differs | finetune → pretrain | hf_trainer at `distributed/ddp-tutorial-series/multigpu.py:91` → plain at `word_language_model/main.py:191` | t5-base → PositionalEncoding | 16 → 5; dropped checkpoint_every, d_model, eval_every, image_size, max_steps, n_head, n_layer, pin_memory, seq_len, temperature, weight_decay; now from the job's own file: batch_size 2→20, epochs 2→40, lr 0.25→20 |
| F3 pytorch-cifar | same | vision → vision | plain at `main.py:154` → plain at `main.py:154` | VGG → VGG | 3 → 3 |
| F2 text_classification (notebook) | same | finetune → finetune | hf_trainer at `text_classification.py:135` → hf_trainer at `text_classification.py:135` | distilbert-base-uncased → distilbert-base-uncased | 5 → 5 |
