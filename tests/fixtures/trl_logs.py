"""HF Trainer subclass that logs TRL-style GRPO metrics; the script asks for rare logging."""

import torch
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments


class Toks(Dataset):
	def __len__(self):
		return 1000

	def __getitem__(self, i):
		ids = torch.randint(2, 128, (16,))
		return {"input_ids": ids, "labels": ids}


class GRPOLike(Trainer):
	def log(self, logs, *args, **kwargs):
		logs = {
			**logs,
			"completions/clipped_ratio": 0.9,
			"frac_reward_zero_std": 0.4,
			"kl": 0.02,
			"entropy": 1.1,
			"reward": 0.3,
			"completions/mean_length": 480.0,
		}
		return super().log(logs, *args, **kwargs)


cfg = LlamaConfig(
	vocab_size=128,
	hidden_size=32,
	intermediate_size=64,
	num_hidden_layers=1,
	num_attention_heads=4,
	num_key_value_heads=2,
	max_position_embeddings=32,
)
args = TrainingArguments(
	output_dir="out",
	per_device_train_batch_size=2,
	max_steps=200,
	logging_steps=500,
	save_strategy="no",
	report_to=["wandb"],
	use_cpu=True,
)
GRPOLike(model=LlamaForCausalLM(cfg), args=args, train_dataset=Toks()).train()
