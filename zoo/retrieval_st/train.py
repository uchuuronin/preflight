"""Retrieval: fine-tune a small sentence-transformers bi-encoder on Natural Questions pairs.

BUG ZOO: this script carries a planted problem on purpose. Do not fix it;
Preflight must catch it from a few minutes of probing.

  1. Contrastive batch too small: MultipleNegativesRankingLoss uses the other answers in the
     batch as negatives, and the batch is 8, so each query sees only 7 negatives. Retrieval
     quality suffers, while the GPU has room for a far larger batch at this model size.

Read from the code: retrieval.small_batch.
Found by the probe: the memory headroom at batch 8 (the batch vs memory trade-off on the fit
map), and the step time: at batch 8 one epoch is 12,279 steps [derived: 98,231 training
pairs / 8], likely the long pole rather than encoding the 100,000-answer corpus for the
recall check [guess; the probe's step and encoding times settle it].

Ground truth: not measured yet. A's probe and full run fill it in.

Public data and model only: sentence-transformers/natural-questions (pairs drawn from Google's
Natural Questions, CC BY-SA 3.0) and sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0).
"""

import json
import time

import torch
from datasets import load_dataset
from sentence_transformers import (
	SentenceTransformer,
	SentenceTransformerTrainer,
	SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_EVAL = 2000
ENCODE_BATCH = 256


def recall_at_10(model: SentenceTransformer, queries: list[str], corpus: list[str]) -> float:
	q = model.encode(queries, batch_size=ENCODE_BATCH, convert_to_tensor=True, normalize_embeddings=True)
	c = model.encode(corpus, batch_size=ENCODE_BATCH, convert_to_tensor=True, normalize_embeddings=True)
	hits = 0
	for start in range(0, len(queries), 512):
		scores = q[start : start + 512] @ c.T
		top = scores.topk(10, dim=1).indices
		gold = torch.arange(start, start + len(top), device=top.device).unsqueeze(1)
		hits += (top == gold).any(dim=1).sum().item()
	return hits / len(queries)


def main() -> None:
	data = load_dataset("sentence-transformers/natural-questions", split="train").shuffle(seed=0)
	evals = data.select(range(N_EVAL))
	train = data.select(range(N_EVAL, len(data)))
	model = SentenceTransformer(MODEL, device="cuda")
	loss = MultipleNegativesRankingLoss(model)
	args = SentenceTransformerTrainingArguments(
		output_dir="out",
		num_train_epochs=1,
		per_device_train_batch_size=8,
		learning_rate=2e-5,
		warmup_ratio=0.1,
		bf16=True,
		logging_steps=100,
		save_strategy="no",
		report_to=[],
	)
	started = time.monotonic()
	SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=loss).train()
	trained_s = time.monotonic() - started
	# The corpus: every answer in the data (the eval answers sit at the start, as the gold rows).
	corpus = evals["answer"] + train["answer"][: 100_000 - N_EVAL]
	started = time.monotonic()
	r10 = recall_at_10(model, evals["query"], corpus)
	print(
		json.dumps(
			{
				"train_pairs": len(train),
				"train_s": round(trained_s, 1),
				"corpus": len(corpus),
				"encode_and_search_s": round(time.monotonic() - started, 1),
				"recall_at_10": round(r10, 4),
			}
		),
		flush=True,
	)


if __name__ == "__main__":
	main()
