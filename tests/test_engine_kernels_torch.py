"""A real profiled training step, through A's kernel_table, classifies with no 'other' rows."""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch


class _Block(torch.nn.Module):
	def __init__(self, d: int, heads: int):
		super().__init__()
		self.heads = heads
		self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
		self.out = torch.nn.Linear(d, d, bias=False)
		self.up = torch.nn.Linear(d, 2 * 4 * d, bias=False)
		self.down = torch.nn.Linear(4 * d, d, bias=False)
		self.n1 = torch.nn.RMSNorm(d)
		self.n2 = torch.nn.LayerNorm(d)

	def forward(self, x):
		b, s, d = x.shape
		q, k, v = self.qkv(self.n1(x)).view(b, s, 3, self.heads, d // self.heads).unbind(2)
		a = torch.nn.functional.scaled_dot_product_attention(
			q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
		)
		x = x + self.out(a.transpose(1, 2).reshape(b, s, d))
		g, u = self.up(self.n2(x)).chunk(2, dim=-1)
		return x + torch.nn.functional.dropout(self.down(torch.nn.functional.silu(g) * u), 0.1)


def _transformer_step(foreach: bool):
	d, vocab = 64, 256
	emb = torch.nn.Embedding(vocab, d)
	blocks = torch.nn.Sequential(_Block(d, 4), _Block(d, 4))
	head = torch.nn.Linear(d, vocab, bias=False)
	params = [*emb.parameters(), *blocks.parameters(), *head.parameters()]
	opt = torch.optim.AdamW(params, lr=1e-3, foreach=foreach)
	x = torch.randint(0, vocab, (4, 32))
	mask = torch.rand(4, 32) > 0.1

	def step():
		logits = head(blocks(emb(x)))
		loss = torch.nn.functional.cross_entropy(logits[mask], x[mask])
		loss.backward()
		torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=foreach)
		opt.step()
		opt.zero_grad()

	return step


def _cnn_step(foreach: bool):
	net = torch.nn.Sequential(
		torch.nn.Conv2d(3, 8, 3, padding=1),
		torch.nn.BatchNorm2d(8),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d(2),
		torch.nn.Upsample(scale_factor=2, mode="nearest"),
		torch.nn.ConvTranspose2d(8, 8, 3, padding=1),
		torch.nn.GELU(),
		torch.nn.Dropout(0.1),
		torch.nn.AdaptiveAvgPool2d(1),
		torch.nn.Flatten(),
		torch.nn.Linear(8, 10),
	)
	opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, foreach=foreach)
	x, y = torch.randn(8, 3, 16, 16), torch.randint(0, 10, (8,))

	def step():
		torch.nn.functional.cross_entropy(net(x), y).backward()
		opt.step()
		opt.zero_grad()

	return step


@pytest.mark.parametrize(
	"make,foreach",
	[(_transformer_step, False), (_transformer_step, True), (_cnn_step, False), (_cnn_step, True)],
)
def test_cpu_training_step_has_no_unclassified_ops(make, foreach):
	from torch.profiler import ProfilerActivity, profile

	from preflight.engine.steptime import KERNEL_CLASSES, classify_kernel
	from preflight.recorder.kernels import kernel_table

	torch.manual_seed(0)
	step = make(foreach)
	step()
	with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
		for _ in range(2):
			step()
	table = kernel_table(
		prof, steps=2, device="cpu", dtype="fp32", allow_tf32=False, classify=classify_kernel
	)
	assert table["rows"]
	assert {r["kernel_class"] for r in table["rows"]} <= KERNEL_CLASSES
	other = [r["name"] for r in table["rows"] if r["kernel_class"] == "other"]
	assert not other, other
