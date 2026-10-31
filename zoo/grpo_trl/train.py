"""GRPO on GSM8K with TRL: Qwen2.5-0.5B-Instruct learns to answer grade-school maths.

BUG ZOO: this script carries planted problems on purpose. Do not fix them;
Preflight must catch every one from a few minutes of probing.

  1. Completions are capped at 512 tokens, but the prompt asks the model to reason step by step
     inside <think> tags first: long reasoning is cut off before the answer, the cut completions
     score 0, and the reward signal for exactly the hardest prompts is lost (truncation rate).
  2. Eight completions per prompt on easy, short questions: many groups are all right or all
     wrong, so their advantages are zero and those rollouts teach nothing (zero-spread fraction,
     from the pass rate per prompt).
  3. beta 0.04 keeps the KL term, so a whole reference-model forward runs on every micro-step
     (step_time.by_phase.reference).

Everything is public: the model (Apache-2.0) and GSM8K (MIT). Written for Preflight, not taken
from any other project.
"""

import re

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "openai/gsm8k"
MAX_COMPLETION = 512  # planted 1
GROUP = 8  # planted 2
BETA = 0.04  # planted 3: the KL term keeps a reference forward on every step
PROMPTS_PER_STEP = 2

SYSTEM = (
	"Solve the problem. First reason step by step inside <think> </think>, "
	"then give only the final number inside <answer> </answer>."
)


def to_prompt(ex):
	answer = ex["answer"].split("####")[-1].strip().replace(",", "")
	return {
		"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["question"]}],
		"target": answer,
	}


train = load_dataset(DATASET, "main", split="train").map(to_prompt, remove_columns=["question", "answer"])

ANSWER = re.compile(r"<answer>\s*(-?[\d.,]+)\s*</answer>")


def text_of(completion):
	return completion[0]["content"] if isinstance(completion, list) else completion


def correct(prompts, completions, target, **kwargs):
	"""1 for the right final number inside <answer>, else 0 (a cut-off completion has none)."""
	out = []
	for c, t in zip(completions, target, strict=True):
		m = ANSWER.search(text_of(c))
		out.append(1.0 if m and m.group(1).replace(",", "").rstrip(".") == t else 0.0)
	return out


def formatted(prompts, completions, **kwargs):
	"""0.2 for using both tags, so the model learns the format before the maths."""
	return [0.2 if "</think>" in text_of(c) and "<answer>" in text_of(c) else 0.0 for c in completions]


args = GRPOConfig(
	output_dir="out/grpo-qwen",
	learning_rate=1e-6,
	per_device_train_batch_size=PROMPTS_PER_STEP,
	gradient_accumulation_steps=4,
	num_generations=GROUP,
	max_prompt_length=256,
	max_completion_length=MAX_COMPLETION,
	beta=BETA,
	num_train_epochs=1,
	bf16=torch.cuda.is_available(),
	logging_steps=5,
	save_steps=500,
	report_to=[],
	seed=42,
)

trainer = GRPOTrainer(
	model=MODEL_ID,
	reward_funcs=[correct, formatted],
	args=args,
	train_dataset=train,
)
trainer.train()
