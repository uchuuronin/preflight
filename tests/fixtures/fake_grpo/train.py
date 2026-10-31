"""Tiny GRPO-shaped run on CPU for the recorder's RL phase timer test."""

import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from trl.trainer.grpo_trainer import GRPOTrainer

torch.manual_seed(0)
config = LlamaConfig(
	vocab_size=64,
	hidden_size=32,
	intermediate_size=64,
	num_hidden_layers=2,
	num_attention_heads=4,
	num_key_value_heads=2,
	max_position_embeddings=64,
)
model = LlamaForCausalLM(config)
ref = copy.deepcopy(model).eval()
for p in ref.parameters():
	p.requires_grad_(False)


def even_last(prompts, completions, **kw):
	return [1.0 if c[-1] % 2 == 0 else 0.0 for c in completions]


def length_bonus(prompts, completions, **kw):
	return [0.1 * len(c) for c in completions]


prompts = [[i % 50 + 1, (i * 7) % 50 + 1, 3] for i in range(64)]
GRPOTrainer(model, ref, [even_last, length_bonus], num_generations=4, prompts=prompts).train()
