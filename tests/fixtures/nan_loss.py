import torch

model = torch.nn.Linear(4, 4)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(100):
	loss = model(torch.randn(2, 4)).pow(2).mean()
	if step == 1:
		loss = loss * float("nan")
	loss.backward()
	opt.step()
	opt.zero_grad()
