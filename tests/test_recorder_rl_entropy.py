"""The exact policy entropy for TRL 0.14 (A's item 14; PM, Nov 1): recorder.runtime.completion_entropy."""

import math

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from preflight.recorder import runtime  # noqa: E402


def _direct(logits, mask):
	p = torch.softmax(logits.float(), -1)
	h = -(p * torch.log(p)).sum(-1)
	return float((h * mask).sum() / mask.sum())


def test_entropy_over_the_completion_up_to_the_first_eos(monkeypatch):
	torch.manual_seed(0)
	b, k, v, eos = 3, 7, 50, 2
	logits = torch.randn(b, k + 1, v) * 3  # TRL 0.14 asks for keep + 1 positions and drops the last
	ids = torch.randint(3, v, (b, 5 + k))
	ids[0, 5 + 2] = eos  # row 0 ends at completion position 2; row 1 at 5; row 2 never
	ids[1, 5 + 5] = eos
	mask = torch.zeros(b, k)
	mask[0, :3] = 1
	mask[1, :6] = 1
	mask[2, :] = 1
	want = _direct(logits[:, :-1], mask)
	got = float(runtime.completion_entropy(logits, ids, k, eos))
	assert got == pytest.approx(want, rel=1e-5)
	# chunked to one position at a time, the same number
	monkeypatch.setattr(runtime, "ENTROPY_CHUNK_BYTES", 1)
	assert float(runtime.completion_entropy(logits, ids, k, eos)) == pytest.approx(want, rel=1e-5)
	# a uniform distribution has log(V) nats
	flat = torch.zeros(1, k + 1, v)
	assert float(runtime.completion_entropy(flat, ids[:1], k, None)) == pytest.approx(math.log(v), rel=1e-6)


def test_the_hook_reads_the_update_s_forward_only():
	from torch import nn

	class Out:
		def __init__(self, logits):
			self.logits = logits

	class Tiny(nn.Module):
		def __init__(self):
			super().__init__()
			self.lin = nn.Linear(4, 11)

		def forward(self, input_ids, num_logits_to_keep=None):
			x = torch.randn(input_ids.shape[0], num_logits_to_keep or input_ids.shape[1], 4)
			return Out(self.lin(x))

	rec = runtime.Recorder()
	m = Tiny()
	m.register_forward_hook(rec._entropy_hook(eos_token_id=None), with_kwargs=True)
	ids = torch.randint(0, 11, (2, 9))
	m(ids, num_logits_to_keep=5)  # the update's forward: counted
	with torch.no_grad():
		m(ids, num_logits_to_keep=5)  # a reference forward: not counted
	m(ids)  # no num_logits_to_keep: not TRL's update
	assert len(rec._entropy_exact) == 1
	assert 0 < float(rec._entropy_exact[0][1]) <= math.log(11) + 1e-6
