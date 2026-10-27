import torch

model = torch.nn.Linear(8, 8)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
steps = 0
while True:
	try:
		loss = model(torch.randn(4, 8)).pow(2).mean()
		loss.backward()
		opt.step()
		opt.zero_grad()
		steps += 1
	except Exception:
		pass  # a script that swallows everything must still stop
