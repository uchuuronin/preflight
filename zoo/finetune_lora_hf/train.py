"""LoRA fine-tune of Qwen2.5-1.5B on Alpaca with the Hugging Face Trainer.

BUG ZOO: this script carries planted problems on purpose. Do not fix them;
Preflight must catch every one from a few minutes of probing.

  1. Eval batch is 32, eight times the train batch.
  2. The perplexity callback runs the model without torch.no_grad() and keeps
     every loss tensor, so the whole graph stays alive through eval.
  3. No packing: every batch is padded to its longest example.

Ground truth: an L4 (24 GB) trains fine, then runs out of memory at the first
eval (step 200). Padding is about 46% of tokens.
"""

import math

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	Trainer,
	TrainerCallback,
	TrainingArguments,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B"
DATASET = "yahma/alpaca-cleaned"
MAX_LEN = 1024
EVAL_EVERY = 200

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
lora = LoraConfig(
	r=16,
	lora_alpha=32,
	lora_dropout=0.05,
	target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
	task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)


def format_example(ex):
	prompt = f"### Instruction:\n{ex['instruction']}\n\n"
	if ex["input"]:
		prompt += f"### Input:\n{ex['input']}\n\n"
	text = prompt + f"### Response:\n{ex['output']}{tokenizer.eos_token}"
	enc = tokenizer(text, truncation=True, max_length=MAX_LEN)
	enc["labels"] = list(enc["input_ids"])
	return enc


raw = load_dataset(DATASET, split="train").train_test_split(test_size=0.02, seed=42)
train_ds = raw["train"].map(format_example, remove_columns=raw["train"].column_names)
eval_ds = raw["test"].map(format_example, remove_columns=raw["test"].column_names)
collator = DataCollatorForSeq2Seq(tokenizer, padding="longest")  # planted 3: pad, no packing


class PerplexityCallback(TrainerCallback):
	def __init__(self, dataset, batch_size):
		self.dataset = dataset
		self.batch_size = batch_size

	def on_step_end(self, args, state, control, model=None, **kwargs):
		if state.global_step == 0 or state.global_step % EVAL_EVERY:
			return
		model.eval()
		losses = []
		loader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collator)
		for batch in loader:
			batch = {k: v.to(model.device) for k, v in batch.items()}
			out = model(**batch)  # planted 2: no torch.no_grad()
			losses.append(out.loss)  # planted 2: keeps every graph alive
		model.train()
		ppl = math.exp(torch.stack(losses).mean().item())
		print(f"step {state.global_step}: eval perplexity {ppl:.2f}")


args = TrainingArguments(
	output_dir="out/lora-qwen",
	per_device_train_batch_size=4,
	per_device_eval_batch_size=32,  # planted 1
	gradient_accumulation_steps=4,
	learning_rate=1e-4,
	warmup_steps=50,
	num_train_epochs=1,
	bf16=True,
	logging_steps=10,
	save_steps=500,
	eval_strategy="no",
	report_to=[],
	seed=42,
)

trainer = Trainer(
	model=model,
	args=args,
	train_dataset=train_ds,
	data_collator=collator,
	callbacks=[PerplexityCallback(eval_ds, args.per_device_eval_batch_size)],
)
trainer.train()
model.save_pretrained("out/lora-qwen/adapter")
