"""Eval sweep: two reasoning models on GSM8K with Hugging Face generate.

BUG ZOO: this script carries planted problems on purpose. Do not fix them;
Preflight must catch every one from a few minutes of probing.

  1. No thinking cap: the models think inside <think> ... </think> with no budget of its own,
     and thinking shares MAX_NEW_TOKENS (2,048) with the answer. Long chains use up the budget
     before the final answer is written, so those answers are cut off and scored wrong.
  2. Activations dumped at every layer: generate runs with output_hidden_states=True and every
     layer's hidden states for every generated token are saved to disk, so storage grows with
     tokens x layers x width and passes any small quota long before the sweep ends.

Read from the code: eval.activation_dump.
Found by the probe: the share of answers cut off at MAX_NEW_TOKENS before thinking ends (no
thinking cap), and the bytes written per question, which the storage tank projects over the
whole sweep.

Ground truth: not measured yet. A's probe and full run fill it in.

Public data and models only: GSM8K (openai/gsm8k, config "main", test split, MIT) and
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B and -7B (MIT).
"""

import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
N_QUESTIONS = 200
MAX_NEW_TOKENS = 2048  # shared by thinking and the answer: no thinking cap
TEMPERATURE = 0.6
EVAL_BATCH_SIZE = 8
DUMP_DIR = Path("activations")


def gold(answer: str) -> str:
	return answer.split("####")[-1].strip().replace(",", "")


def predicted(text: str) -> str | None:
	"""The final number after thinking; None when there is none (see truncated() for why)."""
	if "</think>" not in text:
		return None
	tail = text.split("</think>", 1)[1]
	boxed = re.findall(r"\\boxed\{([^}]*)\}", tail)
	nums = re.findall(r"-?\d[\d,]*\.?\d*", boxed[-1] if boxed else tail)
	return nums[-1].replace(",", "").rstrip(".") if nums else None


def truncated(text: str) -> bool:
	"""Thinking never finished inside MAX_NEW_TOKENS, so no answer was written."""
	return "</think>" not in text


def run_model(model_id: str, questions) -> dict:
	tok = AutoTokenizer.from_pretrained(model_id)
	tok.padding_side = "left"
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token
	model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
	model.eval()
	out_dir = DUMP_DIR / model_id.split("/")[-1]
	out_dir.mkdir(parents=True, exist_ok=True)
	right = cut = unreadable = 0
	written = 0
	started = time.monotonic()
	for start in range(0, len(questions), EVAL_BATCH_SIZE):
		batch = questions.select(range(start, min(start + EVAL_BATCH_SIZE, len(questions))))
		prompts = [
			tok.apply_chat_template(
				[{"role": "user", "content": q + "\nPut your final answer in \\boxed{}."}],
				tokenize=False,
				add_generation_prompt=True,
			)
			for q in batch["question"]
		]
		# The chat template already starts with the BOS token.
		enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
		with torch.no_grad():
			gen = model.generate(
				**enc,
				max_new_tokens=MAX_NEW_TOKENS,
				do_sample=True,
				temperature=TEMPERATURE,
				output_hidden_states=True,
				return_dict_in_generate=True,
			)
		# Planted (2): every layer's hidden states for every generated step go to disk.
		for i in range(len(batch)):
			path = out_dir / f"q{start + i:05d}.pt"
			torch.save([tuple(layer[i].cpu() for layer in step) for step in gen.hidden_states], path)
			written += path.stat().st_size
		texts = tok.batch_decode(gen.sequences[:, enc["input_ids"].shape[1] :], skip_special_tokens=False)
		for text, ans in zip(texts, batch["answer"], strict=True):
			p = predicted(text)
			cut += truncated(text)
			unreadable += p is None and not truncated(text)
			right += p == gold(ans)
		del gen, enc  # free this batch's hidden states before the next generate
	return {
		"model": model_id,
		"questions": len(questions),
		"right": right,
		"truncated": cut,
		"no_number": unreadable,
		"activation_bytes": written,
		"wall_s": round(time.monotonic() - started, 1),
	}


def main() -> None:
	questions = load_dataset("openai/gsm8k", "main", split="test").select(range(N_QUESTIONS))
	for model_id in MODELS:
		print(json.dumps(run_model(model_id, questions)), flush=True)


if __name__ == "__main__":
	main()
