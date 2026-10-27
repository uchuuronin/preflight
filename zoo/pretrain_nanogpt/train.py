"""Small GPT-2 (124M) pretrain in the nanoGPT style, one GPU.

BUG ZOO: this script carries planted problems on purpose. Do not fix them.

  1. num_workers = 0 while every sample is tokenised on the fly, so the GPU
     waits on the data loader for about a third of each step.
  2. Trains in fp32 on a GPU with bf16 tensor cores.
  3. Checkpoints every 20,000 steps; on a cluster that requeues after 2 h,
     a preemption loses hours of work.

Settings are module-level constants, nanoGPT style. data_source = "random"
trains on random tokens, for smoke tests without a download.
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------- config
data_source = "roneneldan/TinyStories"
out_dir = "out/nanogpt"
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
vocab_size = 50304
dropout = 0.0
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000
max_iters = 60000
weight_decay = 0.1
grad_clip = 1.0
num_workers = 0  # planted 1
dtype = "float32"  # planted 2
checkpoint_every = 20000  # planted 3
log_every = 10
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------


class TokenStream(Dataset):
	"""Tokenises raw text per sample; cheap to write, expensive to run with no workers."""

	def __init__(self, source, block):
		self.block = block
		self.random = source == "random"
		if self.random:
			self.n = 1_000_000
			return
		from datasets import load_dataset
		from transformers import AutoTokenizer

		self.texts = load_dataset(source, split="train")["text"]
		self.tok = AutoTokenizer.from_pretrained("gpt2")
		self.n = len(self.texts)

	def __len__(self):
		return self.n

	def __getitem__(self, i):
		if self.random:
			g = torch.Generator().manual_seed(i)
			ids = torch.randint(0, vocab_size, (self.block + 1,), generator=g)
		else:
			ids = []
			j = i
			while len(ids) < self.block + 1:
				ids += self.tok(self.texts[j % self.n])["input_ids"] + [self.tok.eos_token_id]
				j += 1
			ids = torch.tensor(ids[: self.block + 1])
		return ids[:-1], ids[1:]


class Block(nn.Module):
	def __init__(self):
		super().__init__()
		self.ln_1 = nn.LayerNorm(n_embd)
		self.c_attn = nn.Linear(n_embd, 3 * n_embd)
		self.c_proj = nn.Linear(n_embd, n_embd)
		self.ln_2 = nn.LayerNorm(n_embd)
		self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

	def forward(self, x):
		b, t, c = x.shape
		q, k, v = self.c_attn(self.ln_1(x)).split(n_embd, dim=2)
		q, k, v = (z.view(b, t, n_head, c // n_head).transpose(1, 2) for z in (q, k, v))
		y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0)
		x = x + self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))
		return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
	def __init__(self):
		super().__init__()
		self.wte = nn.Embedding(vocab_size, n_embd)
		self.wpe = nn.Embedding(block_size, n_embd)
		self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd)
		self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
		self.wte.weight = self.lm_head.weight

	def forward(self, idx, targets=None):
		pos = torch.arange(idx.size(1), device=idx.device)
		x = self.wte(idx) + self.wpe(pos)
		for blk in self.blocks:
			x = blk(x)
		logits = self.lm_head(self.ln_f(x))
		loss = (
			None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
		)
		return logits, loss


def get_lr(it):
	if it < warmup_iters:
		return learning_rate * (it + 1) / warmup_iters
	ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
	return min_lr + 0.5 * (1 + math.cos(math.pi * min(ratio, 1.0))) * (learning_rate - min_lr)


def main():
	torch.manual_seed(seed)
	os.makedirs(out_dir, exist_ok=True)
	model = GPT().to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay
	)
	loader = DataLoader(
		TokenStream(data_source, block_size), batch_size=batch_size, shuffle=True, num_workers=num_workers
	)
	ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
	autocast = torch.autocast(
		device_type="cuda" if "cuda" in device else "cpu", dtype=ptdtype, enabled=dtype != "float32"
	)

	it = 0
	t0 = time.time()
	while it < max_iters:
		for x, y in loader:
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			for g in optimizer.param_groups:
				g["lr"] = get_lr(it)
			with autocast:
				_, loss = model(x, y)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			optimizer.step()
			if it % log_every == 0:
				dt = time.time() - t0
				t0 = time.time()
				print(f"iter {it}: loss {loss.item():.4f}, {dt * 1000 / log_every:.1f} ms/iter")
			if it > 0 and it % checkpoint_every == 0:
				torch.save(
					{"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": it},
					f"{out_dir}/ckpt.pt",
				)
			it += 1
			if it >= max_iters:
				break


if __name__ == "__main__":
	main()
