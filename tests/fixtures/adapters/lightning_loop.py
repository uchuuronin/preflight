"""PyTorch Lightning, the shape of Lightning's quickstart autoencoder, on synthetic data on CPU."""

import lightning as L
import torch
from torch import nn


class AutoEncoder(L.LightningModule):
	def __init__(self):
		super().__init__()
		self.enc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
		self.dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 64))

	def training_step(self, batch, _):
		(x,) = batch
		loss = nn.functional.mse_loss(self.dec(self.enc(x)), x)
		self.log("train_loss", loss)
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-3)


torch.manual_seed(0)
data = torch.utils.data.TensorDataset(torch.randn(2048, 64))
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
L.Trainer(
	max_steps=40, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False
).fit(AutoEncoder(), loader)
