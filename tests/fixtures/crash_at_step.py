import torch

model = torch.nn.Linear(8, 8)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(100):
	loss = model(torch.randn(4, 8)).pow(2).mean()
	loss.backward()
	opt.step()
	opt.zero_grad()
	if step == 2:
		raise ValueError("labels and logits have different lengths")
