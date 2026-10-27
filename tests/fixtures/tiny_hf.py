"""Tiny HF Trainer job for recorder tests: random tiny Llama, padded variable-length batches, CPU."""

import torch
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

MAX_LEN = 48


class Varlen(Dataset):
	def __len__(self):
		return 10_000

	def __getitem__(self, i):
		g = torch.Generator().manual_seed(i)
		n = int(torch.randint(8, MAX_LEN, (1,), generator=g))
		ids = torch.randint(3, 256, (n,), generator=g)
		return {"input_ids": ids}


def collate(feats):
	n = max(len(f["input_ids"]) for f in feats)
	ids = torch.zeros(len(feats), n, dtype=torch.long)
	mask = torch.zeros(len(feats), n, dtype=torch.long)
	for i, f in enumerate(feats):
		ids[i, : len(f["input_ids"])] = f["input_ids"]
		mask[i, : len(f["input_ids"])] = 1
	labels = ids.masked_fill(mask == 0, -100)
	return {"input_ids": ids, "attention_mask": mask, "labels": labels}


config = LlamaConfig(
	vocab_size=256,
	hidden_size=32,
	intermediate_size=64,
	num_hidden_layers=2,
	num_attention_heads=4,
	num_key_value_heads=2,
	max_position_embeddings=64,
)
model = LlamaForCausalLM(config)
args = TrainingArguments(
	output_dir="out",
	per_device_train_batch_size=4,
	gradient_accumulation_steps=2,
	learning_rate=1e-3,
	max_steps=500,
	logging_steps=1000,
	save_strategy="no",
	report_to=[],
	use_cpu=True,
	dataloader_num_workers=0,
)
Trainer(model=model, args=args, train_dataset=Varlen(), data_collator=collate).train()
