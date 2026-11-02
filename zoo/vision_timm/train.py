"""Vision: fine-tune a timm ResNet-50 on leaf photos (bean disease, 3 classes).

BUG ZOO: this script carries planted problems on purpose. Do not fix them;
Preflight must catch every one from a few minutes of probing.

  1. Data-bound loader: every image is decoded, randomly cropped, flipped, colour-jittered and
     RandAugmented at 224 px on the CPU, by only 2 loader workers. The GPU finishes a batch
     faster than 2 workers can prepare the next one, so it waits on the loader.
  2. No channels-last: the model and the inputs stay in the default NCHW layout, so the
     convolutions don't use the tensor cores' preferred NHWC layout under bf16 autocast.

Read from the code: vision.no_channels_last, vision.cpu_augment.
Not raised: 5 epochs, normal when fine-tuning on 1,034 images (data.repetition is for text corpora since Nov 2).
Found by the probe: the share of each step spent waiting on the loader (data wait against
compute on the timeline), and the step time with and without channels-last.

Ground truth: not measured yet. A's probe and full run fill it in.

Public data and model only: AI-Lab-Makerere/beans (MIT; train 1,034, validation 133 images)
and timm/resnet50.a1_in1k (Apache 2.0).
"""

import json
import time

import timm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms

MODEL = "resnet50.a1_in1k"
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Planted (1): heavy augmentation on the CPU, fed by 2 workers.
train_tf = transforms.Compose(
	[
		transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.35, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
		transforms.RandAugment(num_ops=2, magnitude=9),
		transforms.ToTensor(),
		transforms.Normalize(MEAN, STD),
	]
)
eval_tf = transforms.Compose(
	[
		transforms.Resize(256),
		transforms.CenterCrop(IMAGE_SIZE),
		transforms.ToTensor(),
		transforms.Normalize(MEAN, STD),
	]
)


class Beans(torch.utils.data.Dataset):
	def __init__(self, rows, tf):
		self.rows, self.tf = rows, tf

	def __len__(self) -> int:
		return len(self.rows)

	def __getitem__(self, i):
		row = self.rows[i]
		return self.tf(row["image"].convert("RGB")), row["labels"]


def main() -> None:
	torch.manual_seed(0)
	data = load_dataset("AI-Lab-Makerere/beans")
	train = torch.utils.data.DataLoader(
		Beans(data["train"], train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
	)
	val = torch.utils.data.DataLoader(
		Beans(data["validation"], eval_tf), batch_size=BATCH_SIZE, num_workers=2
	)
	# Planted (2): no channels-last for the model or the inputs.
	model = timm.create_model(MODEL, pretrained=True, num_classes=3).cuda()
	opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
	started = time.monotonic()
	for epoch in range(EPOCHS):
		model.train()
		waited = 0.0
		t = time.monotonic()
		for x, y in train:
			waited += time.monotonic() - t
			x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
			with torch.autocast("cuda", dtype=torch.bfloat16):
				loss = F.cross_entropy(model(x), y)
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			torch.cuda.synchronize()
			t = time.monotonic()
		model.eval()
		right = seen = 0
		with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
			for x, y in val:
				pred = model(x.cuda()).argmax(1).cpu()
				right += (pred == y).sum().item()
				seen += len(y)
		print(
			json.dumps(
				{
					"epoch": epoch,
					"val_acc": round(right / seen, 4),
					"loader_wait_s": round(waited, 1),
					"peak_bytes": torch.cuda.max_memory_allocated(),
				}
			),
			flush=True,
		)
	print(json.dumps({"wall_s": round(time.monotonic() - started, 1)}), flush=True)


if __name__ == "__main__":
	main()
