"""corpus/lcpfn.py: the LC-PFN port loads a state dict strictly with weights_only and gives ordered quantiles.

The port was checked against the package's own model code (lcpfn 0.1.3, built in torch 2.x with random weights):
logits agree to 1.5e-7 and quantiles to 2.4e-6 (Nov 1). The published checkpoints are checked on Modal
(`lcpfn_app.py::check`) against the package's quantiles recorded by the fetch run."""

import math

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from preflight.corpus import lcpfn  # noqa: E402


def _small(tmp_path):
	torch.manual_seed(0)
	m = lcpfn.LCPFN(lcpfn.Shape(nlayers=2))
	with torch.no_grad():
		for p in m.parameters():
			p.normal_(0, 0.05)
		m.criterion.borders.copy_(torch.linspace(-0.5, 1.5, 1001))
		m.criterion.bucket_widths.copy_(m.criterion.borders[1:] - m.criterion.borders[:-1])
	path = tmp_path / "small.state.pt"
	torch.save(m.state_dict(), path)
	return path


def test_a_saved_state_loads_strictly_and_gives_ordered_quantiles(tmp_path):
	m = lcpfn.load(_small(tmp_path), {"nlayers": 2})
	x = torch.arange(1, 101, dtype=torch.float32)
	y = (1 - 0.9 * x**-0.35).clamp(0, 1)
	q = m.predict_quantiles(x[:30], y[:30], x[30:], [0.05, 0.5, 0.95])
	assert q.shape == (70, 3) and bool((q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all())
	with pytest.raises(ValueError):
		m.predict_quantiles(x[:30], y[:30] + 2, x[30:], [0.5])


def test_a_state_with_a_key_the_port_lacks_is_refused(tmp_path):
	path = _small(tmp_path)
	state = torch.load(path, weights_only=True)
	state["style_encoder.embedding.weight"] = torch.zeros(1)
	torch.save(state, path)
	with pytest.raises(RuntimeError):
		lcpfn.load(path, {"nlayers": 2})


def test_the_normaliser_round_trips_and_a_falling_loss_rises():
	to, back = lcpfn.pfn_normalize(0.0, math.inf, 1.0, 4.0, minimize=True)
	v = torch.tensor([1.5, 2.0, 3.0])
	assert torch.allclose(back(to(v)), v, atol=1e-4)
	assert bool(
		(to(v)[1:] < to(v)[:-1]).all()
	)  # minimised: a higher loss reads lower on the prior's rising scale


def test_a_loss_forecast_has_its_band_in_loss_order(tmp_path):
	m = lcpfn.load(_small(tmp_path), {"nlayers": 2})
	loss = [2.0 + 4.0 * s**-0.5 for s in range(1, 301)]
	fc = lcpfn.forecast_loss(m, loss, last_step=3000)
	assert fc is not None and fc.lo <= fc.mid <= fc.hi and "300 loss values" not in fc.reason
	assert "100 loss values" in fc.reason  # thinned to MAX_POINTS
	assert lcpfn.forecast_loss(m, loss, last_step=200) is None  # already past the last step
