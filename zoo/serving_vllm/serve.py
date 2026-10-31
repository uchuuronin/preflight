"""Serving with vLLM: Qwen2.5-7B-Instruct answering a burst of instructions on one GPU.

BUG ZOO: this script carries a planted problem on purpose. Do not fix it;
Preflight must catch it from a few minutes of probing.

  1. Max context too high for the concurrency: the server admits up to 256 sequences at once
     (max_num_seqs) and lets each grow to 32,768 tokens (max_model_len), but on an L4
     (22.06 GiB) at 90% the bf16 weights (about 15.2 GB) leave a KV cache of roughly 70,000 to
     90,000 tokens, two or three full-length sequences. Each reply here runs to 4,096 tokens,
     so the batch asks for about 1.05 million tokens of cache (256 x 4,096) and only about 16
     to 21 sequences fit at a time, prompts included [derived: 57,344 bytes of KV per token = 28 layers x 4 KV heads x
     128 x K and V x 2 bytes; vLLM's activation reserve is a guess the probe replaces]. vLLM
     preempts and recomputes the rest, so the GPU redoes work and throughput stops rising
     well short of the concurrency the settings promise.

Read from the code: none (the settings are read; the overflow needs the KV size measured).
Found by the probe: the KV cache's capacity in tokens against what max_num_seqs x the context
asks for (the KV tank on the fit map), and the preemptions at full concurrency.

Runs in a serving image only: vllm 0.6.4.post1, like every vLLM release built for torch 2.5.1,
needs numpy below 2, and it pulls in pyairports, which has no wheel, so the standard probe image
(numpy 2.1.3, wheels only) can't install it.

Ground truth: not measured yet. A's probe and full run fill it in.

Public data and model only: databricks/databricks-dolly-15k (CC BY-SA 3.0) and
Qwen/Qwen2.5-7B-Instruct (Apache 2.0).
"""

import json
import time

from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_REQUESTS = 1024
OUTPUT_TOKENS = 4096


def main() -> None:
	rows = load_dataset("databricks/databricks-dolly-15k", split="train").select(range(N_REQUESTS))
	llm = LLM(
		model=MODEL,
		dtype="bfloat16",
		max_model_len=32768,
		max_num_seqs=256,
		gpu_memory_utilization=0.90,
		# Prefill in 8,192-token chunks: without it vLLM's start-up check profiles a full
		# 32,768-token prefill at once, which may leave too little room to start at all.
		enable_chunked_prefill=True,
		max_num_batched_tokens=8192,
	)
	tok = llm.get_tokenizer()
	prompts = [
		tok.apply_chat_template(
			[{"role": "user", "content": (r["context"] + "\n\n" if r["context"] else "") + r["instruction"]}],
			tokenize=False,
			add_generation_prompt=True,
		)
		for r in rows
	]
	# A load test, as vLLM's own benchmark runs one: every reply runs to OUTPUT_TOKENS.
	params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
	started = time.monotonic()
	outs = llm.generate(prompts, params)
	wall = time.monotonic() - started
	generated = sum(len(o.outputs[0].token_ids) for o in outs)
	prompt_tokens = sum(len(o.prompt_token_ids) for o in outs)
	print(
		json.dumps(
			{
				"requests": len(outs),
				"prompt_tokens": prompt_tokens,
				"generated_tokens": generated,
				"wall_s": round(wall, 1),
				"output_tokens_per_s": round(generated / wall, 1),
			}
		),
		flush=True,
	)


if __name__ == "__main__":
	main()
