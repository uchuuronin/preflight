"""HF Trainer, the shape of transformers' run_clm.py example, on a tiny random GPT-2 and
synthetic token data so it runs on CPU with no downloads."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

torch.manual_seed(0)
model = GPT2LMHeadModel(GPT2Config(vocab_size=128, n_positions=32, n_embd=32, n_layer=2, n_head=2))


class Tokens(torch.utils.data.Dataset):
	def __len__(self):
		return 512

	def __getitem__(self, i):
		ids = torch.randint(0, 128, (32,), generator=torch.Generator().manual_seed(i))
		return {"input_ids": ids, "labels": ids.clone()}


args = TrainingArguments(output_dir="out/hf-trainer", per_device_train_batch_size=8, max_steps=40, learning_rate=1e-3,
	logging_steps=5, save_strategy="no", report_to=[], use_cpu=True, dataloader_num_workers=0)  # fmt: skip
Trainer(model=model, args=args, train_dataset=Tokens()).train()
