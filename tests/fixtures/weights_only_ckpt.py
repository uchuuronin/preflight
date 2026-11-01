"""A training loop that checkpoints only the model's weights (a resume restarts Adam and the
step count); with --full it saves the optimizer, scheduler, RNG and step too."""

import os
import sys
import tempfile

import torch

torch.manual_seed(0)
model = torch.nn.Linear(16, 16)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, 10)
out = tempfile.mkdtemp()
for step in range(40):
	loss = model(torch.randn(8, 16)).pow(2).mean()
	loss.backward()
	opt.step()
	opt.zero_grad()
	sched.step()
	if step % 2 == 1:  # inside the probe's few steps
		if "--full" in sys.argv:
			torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
				"rng": torch.get_rng_state(), "step": step}, os.path.join(out, "ckpt.pt"))  # fmt: skip
		else:
			torch.save({"model": model.state_dict()}, os.path.join(out, "ckpt.pt"))
