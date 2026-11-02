# Preflight unseen scripts (held-out validation set)

salt: 825b2dc98d9dcfbd1dfd1f6af700e426
written: 2026-11-01, by PM. Private to Chiya until the validation runs. The repo holds only the
SHA-256 of this exact file (docs/validation/unseen-hash.txt), so anyone can check later that the
list was fixed before the engine saw these scripts.

## Rules

- None of these comes from D's census sources (TRL c886cf2, transformers v5.19.0 examples,
  PEFT 7a05ae5, nanoGPT 3adf61e) or from our zoo/.
- Each is pinned to a commit. Models and data are public and ungated.
- Run each as published. The only allowed change is the one in its "Change" line (single-GPU
  flag or model endpoint). Anything else that has to change is logged as a finding, not fixed.
- Preflight probes each one for a few minutes, then predicts memory, time, cost and P(OOM).
  The full run is the ground truth. Hit rule: the existing one (at least 90% of runs within
  tolerance, Wilson interval printed). An OOM that Preflight called is a hit for risk.
- Budget: at most about $6 on Modal for all nine at the run lengths below; Muse Spark tokens
  under $0.10. Chiya says yes before any run that spends.
- If a script won't install or run at its pinned commit, use its listed fallback. Fallbacks are
  fixed here too, so swapping one in isn't a choice we make after seeing results.

## The nine

### 1. Fine-tune: torchtune LoRA
- Repo: pytorch/torchtune @ bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1 (2026-04-23)
- Entry: recipes/lora_finetune_single_device.py
- Command (L4): `tune download Qwen/Qwen2.5-1.5B-Instruct --output-dir /tmp/Qwen2.5-1.5B-Instruct`
  then `tune run lora_finetune_single_device --config qwen2_5/1.5B_lora_single_device`
- Command (A100 80GB, the 7B case): same with `qwen2_5/7B_lora_single_device` and Qwen/Qwen2.5-7B-Instruct
- Data: alpaca_cleaned (from the config). Change: none.
- Fallback: none needed (two configs already).

### 2. Pretrain: LitGPT
- Repo: Lightning-AI/litgpt @ 4738993a5e0190b851162a52173729827d3884e7 (2026-10-28)
- Entry: litgpt/pretrain.py with config_hub/pretrain/debug.yaml
- Command (L4): `litgpt pretrain --config config_hub/pretrain/debug.yaml`
- Model: pythia-14m from scratch. Data: TinyStories (ungated). Change: none.
- Full run capped at 30 min wall clock; time prediction checked against tokens/s and the
  extrapolated finish, not the cap.
- Fallback: same repo, `litgpt pretrain pythia-160m --data TinyStories` with debug.yaml.

### 3. RL post-training: open-r1 GRPO (uses TRL 0.18.0, not our census commit)
- Repo: huggingface/open-r1 @ 5b6ff22b3fb7aa069c54866e517f39dfc3160e09 (2026-10-02)
- Entry: src/open_r1/grpo.py with recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml
- Command (A100 80GB): `python src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml --vllm_mode colocate --max_steps 50`
- Data: open-r1/OpenR1-Math-220k. Change: `--vllm_mode colocate` (single GPU) and 50 steps.
- Fallback: same command with `--use_vllm false` if colocate isn't in the pinned TRL.

### 4. Agent / API: smolagents
- Repo: huggingface/smolagents @ 96f33faaf028479119ec8d34507b47694cf14e34 (2026-10-06)
- Entry: examples/agent_from_any_llm.py
- Change: `chosen_inference = "openai"` and the OpenAIModel line pointed at the Muse Spark
  OpenAI-compatible endpoint (model id from our env, key from the secret store, never in the file).
- Checks: token count and dollar cost per run, calls per task, wall time.
- Second case, no API spend: the same file with `chosen_inference = "transformers"`
  (HuggingFaceTB/SmolLM2-1.7B-Instruct on L4).
- Fallback: examples/structured_output_tool.py from the same commit, same change.

### 5. Evaluation sweep: lm-evaluation-harness
- Repo: EleutherAI/lm-evaluation-harness @ v0.4.13 = ddd67220430a2470529f25fd5c05a576ca1057a0 (2026-09-01)
- Command (L4): `lm_eval --model hf --model_args pretrained=HuggingFaceTB/SmolLM2-1.7B-Instruct,dtype=bfloat16 --tasks arc_easy,hellaswag,piqa --batch_size auto`
- Change: none (CLI as documented).
- Fallback: same command, `--tasks arc_easy,piqa`.

### 6. Self-hosted serving: vLLM bench
- Repo: vllm-project/vllm @ v0.32.0 = 5df63a7203cdb9328d824a27b682e77c7b9928be (2026-10-17)
- Entry: vllm/benchmarks/throughput.py (the `vllm bench throughput` CLI; benchmarks/*.py are now shims)
- Command (L4): `vllm bench throughput --model microsoft/Phi-3.5-mini-instruct --dataset-name random --input-len 512 --output-len 256 --num-prompts 500`
- Change: none. Differs from our zoo (offline LLM(), Qwen 7B, dolly).
- Fallback: same command with Qwen/Qwen2.5-3B-Instruct.

### 7. Retrieval: BEIR dense evaluation
- Repo: beir-cellar/beir @ ef83d29307061c65d04b035b4f4e7c18bd8374af (2025-10-16)
- Entry: examples/retrieval/evaluation/dense/evaluate_sbert.py
- Command (L4): `python examples/retrieval/evaluation/dense/evaluate_sbert.py`
- Data: trec-covid (171k docs, from the script's own URL). Change: none.
- Fallback: same script with `dataset = "scifact"`.

### 8. Graph neural network: OGB arxiv
- Repo: snap-stanford/ogb @ 61e9784ca76edeaa6e259ba0f836099608ff0586 (2025-05-06)
- Entry: examples/nodeproppred/arxiv/gnn.py
- Command (L4): `python gnn.py --use_sage --runs 1 --epochs 500`
- Data: ogbn-arxiv (downloads itself). Change: `--runs 1` (default 10).
- Needs torch_sparse for T.ToSparseTensor(); an install failure is a finding.
- Fallback: pyg-team/pytorch_geometric @ ea2c5f62d54c8b414dd3ea7139814a7c1ef83182, examples/ogbn_train.py, defaults.

### 9. Vision: airbench (CIFAR-10)
- Repo: KellerJordan/cifar10-airbench @ 4c1b6d1e3889b037efadcfd5c0ea65b246592362 (2025-11-15)
- Entry: airbench94_muon.py
- Command (L4): `python airbench94_muon.py`
- Data: CIFAR-10 via torchvision (downloads itself). Change: none.
- Fallback: legacy/airbench94.py from the same commit.
