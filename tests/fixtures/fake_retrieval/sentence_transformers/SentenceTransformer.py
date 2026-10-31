import torch
import torch.nn as nn


class SentenceTransformer(nn.Module):
	def __init__(self, name=None):
		super().__init__()
		self.emb = nn.EmbeddingBag(1000, 32)

	def forward(self, ids):
		return self.emb(ids)

	def encode(self, sentences, batch_size=32, **kw):
		with torch.no_grad():
			ids = torch.tensor(
				[[hash(w) % 1000 for w in s.split()][:8] + [0] * (8 - len(s.split()[:8])) for s in sentences]
			)
			return self.emb(ids).numpy()
