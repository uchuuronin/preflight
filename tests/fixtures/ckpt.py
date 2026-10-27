import torch

checkpoint_every = 3
model = torch.nn.Linear(2048, 2048)  # 16 MB of fp32 weights, plus Adam state in the checkpoint
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(1, 1000):
	loss = model(torch.randn(8, 2048)).pow(2).mean()
	loss.backward()
	opt.step()
	opt.zero_grad()
	if step % checkpoint_every == 0:
		# a 256 MB checkpoint: big enough that its save time stands well clear of step-time noise
		torch.save(
			{"model": model.state_dict(), "opt": opt.state_dict(), "extra": torch.zeros(64 * 1024 * 1024)},
			"ckpt.pt",
		)
