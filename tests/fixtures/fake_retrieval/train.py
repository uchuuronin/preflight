"""Retrieval-shaped: encode a corpus, index it, mine hard negatives, then train the encoder."""

import faiss
import torch
from sentence_transformers import SentenceTransformer

torch.manual_seed(0)
model = SentenceTransformer("tiny")
corpus = [f"doc {i} about topic {i % 17} and words {i * 3}" for i in range(2000)]
emb = model.encode(corpus)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
_, hard = index.search(model.encode(corpus[:100]), 5)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(1000):
	ids = torch.randint(0, 1000, (16, 8))
	loss = model(ids).pow(2).mean()
	loss.backward()
	opt.step()
	opt.zero_grad()
