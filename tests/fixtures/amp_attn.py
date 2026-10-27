"""Attention via SDPA plus a GradScaler, to exercise the max-attention-logit and loss-scale signals."""

import torch
import torch.nn.functional as F

d, h = 32, 4
qkv = torch.nn.Linear(d, 3 * d)
opt = torch.optim.SGD(qkv.parameters(), lr=0.1)
scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
for _step in range(1000):
	x = torch.randn(2, 64, d)
	q, k, v = qkv(x).split(d, dim=-1)
	q, k, v = (t.view(2, 64, h, d // h).transpose(1, 2) for t in (q, k, v))
	loss = F.scaled_dot_product_attention(q, k, v, is_causal=True).pow(2).mean()
	scaler.scale(loss).backward()
	scaler.step(opt)
	scaler.update()
	opt.zero_grad()
