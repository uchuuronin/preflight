"""LC-PFN, ported to a modern torch: the learning-curve half of the loss forecast (main plan: "Power-law fit and
LC-PFN posterior, ensembled").

LC-PFN (Adriaensen, Rakotoarison, Müller and Hutter, "Efficient Bayesian Learning Curve Extrapolation using
Prior-Data Fitted Networks", NeurIPS 2023; github.com/automl/lcpfn, MIT licence, copyright 2024
AutoML-Freiburg-Hannover) is a transformer trained on curves drawn from a prior of parametric learning curves. Given
the first points of a curve it returns a distribution over the rest, as logits over 1,000 bars. The `lcpfn` 0.1.3
package pins torch 1.11 and loads a whole pickled module, so this file re-states the parts inference needs (the
encoders, the encoder layer, the bar distribution's quantile function and the package's normaliser) for the state
dicts `corpus/lcpfn_app.py` saved, loaded with `weights_only=True`. Inference only: no training code, no dropout.

The prior's curves rise from 0 to at most 1 over x = 1 to 100. A loss curve is mapped onto that: its steps scale so
the run's last step is x = 100, and the loss goes through the package's own normaliser for a minimised curve
(`pfn_normalize(minimize=True)`), so it rises in [0, 1].

Needs torch: it runs in the runner's image and on Modal, never inside `engine.predict` (engine/ stays torch-free;
`engine.lossforecast.ensemble` combines the two halves from plain numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

MODEL_ID = "lcpfn@0.1.3-port"
# the prior's x range: the package trains on seq_len 100 and normalises x by Normalize(0, 101)
X_MAX = 100
MAX_POINTS = 100  # guess: at most this many seen points go in; longer series are thinned evenly


@dataclass(frozen=True)
class Shape:
	ninp: int = 512
	nhead: int = 4
	nhid: int = 1024
	nlayers: int = 12
	n_out: int = 1000


# ---------------------------------------------------------------- the network (lcpfn/transformer.py, layer.py)


class _Normalize(nn.Module):
	def __init__(self, mean: float, std: float):
		super().__init__()
		self.mean, self.std = mean, std

	def forward(self, x):
		return (x - self.mean) / self.std


class _Linear(nn.Linear):
	def forward(self, x):
		return super().forward(torch.nan_to_num(x, nan=0.0))


class _Layer(nn.Module):
	"""lcpfn.layer.TransformerEncoderLayer, post-norm, GELU, as the published checkpoints were trained."""

	def __init__(self, d: int, nhead: int, nhid: int):
		super().__init__()
		self.self_attn = nn.MultiheadAttention(d, nhead, dropout=0.0)
		self.linear1 = nn.Linear(d, nhid)
		self.linear2 = nn.Linear(nhid, d)
		self.norm1 = nn.LayerNorm(d, eps=1e-5)
		self.norm2 = nn.LayerNorm(d, eps=1e-5)

	def forward(self, src, mask):
		src = self.norm1(src + self.self_attn(src, src, src, attn_mask=mask, need_weights=False)[0])
		return self.norm2(src + self.linear2(nn.functional.gelu(self.linear1(src))))


class _Encoder(nn.Module):
	def __init__(self, shape: Shape):
		super().__init__()
		self.layers = nn.ModuleList(_Layer(shape.ninp, shape.nhead, shape.nhid) for _ in range(shape.nlayers))

	def forward(self, src, mask):
		for layer in self.layers:
			src = layer(src, mask)
		return src


class _Bars(nn.Module):
	"""lcpfn.bar_distribution.FullSupportBarDistribution: its buffers, and the quantile function the package's
	`predict_quantiles` uses (BarDistribution.icdf, linear inside a bar)."""

	def __init__(self, n_out: int):
		super().__init__()
		self.register_buffer("borders", torch.zeros(n_out + 1))
		self.register_buffer("smoothing", torch.tensor(0.0))
		self.register_buffer("bucket_widths", torch.zeros(n_out))

	def icdf(self, logits, left_prob: float):
		probs = logits.softmax(-1)
		cum = torch.cumsum(probs, -1)
		idx = (
			torch.searchsorted(cum, left_prob * torch.ones(*cum.shape[:-1], 1, device=probs.device))
			.squeeze(-1)
			.clamp(0, cum.shape[-1] - 1)
		)
		cum = torch.cat([torch.zeros(*cum.shape[:-1], 1, device=logits.device), cum], -1)
		rest = left_prob - cum.gather(-1, idx[..., None]).squeeze(-1)
		left, right = self.borders[idx], self.borders[idx + 1]
		return left + (right - left) * rest / probs.gather(-1, idx[..., None]).squeeze(-1)


class LCPFN(nn.Module):
	"""lcpfn.transformer.TransformerModel as the published checkpoints use it: x and y encoders, no style, no
	positional encoding, no global tokens; the decoder an MLP onto the bars. State dict keys match the original's."""

	def __init__(self, shape: Shape | None = None):
		super().__init__()
		shape = shape or Shape()
		self.shape = shape
		# encoder_generator in lcpfn/train_lcpfn.py: Normalize(0, 101), Normalize(0.5, sqrt(1/12)), Linear
		self.encoder = nn.Sequential(
			_Normalize(0.0, 101.0), _Normalize(0.5, math.sqrt(1 / 12)), _Linear(1, shape.ninp)
		)
		# get_normalized_uniform_encoder(Linear): Normalize(0.5, sqrt(1/12)), Linear
		self.y_encoder = nn.Sequential(_Normalize(0.5, math.sqrt(1 / 12)), _Linear(1, shape.ninp))
		self.transformer_encoder = _Encoder(shape)
		self.decoder = nn.Sequential(
			nn.Linear(shape.ninp, shape.nhid), nn.GELU(), nn.Linear(shape.nhid, shape.n_out)
		)
		self.criterion = _Bars(shape.n_out)

	@staticmethod
	def _mask(total: int, n_train: int) -> torch.Tensor:
		"""generate_D_q_matrix: every position attends to the seen points, and a query to itself too."""
		allow = torch.zeros(total, total) == 0
		allow[:, n_train:] = False
		allow |= torch.eye(total, dtype=torch.bool)
		return torch.zeros(total, total).masked_fill(~allow, float("-inf"))

	def forward(self, x_train, y_train, x_test):
		"""Logits over the bars at each x_test (T_test x 1 x n_out), the package's LCPFN.forward."""
		n = x_train.shape[0]
		x = torch.cat([x_train, x_test], 0).reshape(-1, 1, 1)
		y = y_train.reshape(-1, 1, 1)
		xe, ye = self.encoder(x), self.y_encoder(y)
		src = torch.cat([xe[:n] + ye, xe[n:]], 0)
		out = self.transformer_encoder(src, self._mask(src.shape[0], n).to(src.device))
		return self.decoder(out)[n:]

	@torch.no_grad()
	def predict_quantiles(self, x_train, y_train, x_test, qs) -> torch.Tensor:
		if torch.any(x_train < 0) or torch.any(x_test < 0):
			raise ValueError("x values should be non-negative")
		if torch.any((y_train < 0) | (y_train > 1)):
			raise ValueError("y values should be in [0, 1]")
		logits = self(x_train, y_train, x_test)
		return torch.cat([self.criterion.icdf(logits, q) for q in qs], dim=1)


def load(state_path: str | Path, meta: dict | None = None) -> LCPFN:
	"""A checkpoint saved by corpus/lcpfn_app.py, with its shape from the saved meta; strict, weights only."""
	m = meta or {}
	shape = Shape(
		ninp=int(m.get("ninp", 512)),
		nhead=int(m.get("nhead", 4)),
		nhid=int(m.get("nhid", 1024)),
		nlayers=int(m.get("nlayers", 12)),
		n_out=int(m.get("n_out") or 1000),
	)
	model = LCPFN(shape)
	state = torch.load(str(state_path), map_location="cpu", weights_only=True)
	model.load_state_dict(state, strict=True)
	return model.eval()


# ---------------------------------------------------------------- losses onto the prior (lcpfn/utils.py)


def pfn_normalize(lb: float, ub: float, soft_lb: float, soft_ub: float, minimize: bool):
	"""(to, back): the package's normaliser, a sigmoid that maps [lb, ub] onto [0, 1], roughly linear on
	[soft_lb, soft_ub]; `minimize` flips it so a falling loss becomes a rising curve. Scalars or tensors."""
	if not (lb <= soft_lb < soft_ub <= ub):
		raise ValueError("needs lb <= soft_lb < soft_ub <= ub")

	def cinv(v):
		return 1 - v if minimize else v

	def lin(v):
		return 2 / (soft_ub - soft_lb) * (v - soft_lb) - 1

	def lin_inv(v):
		return (v + 1) / 2 * (soft_ub - soft_lb) + soft_lb

	if math.isfinite(lb) and math.exp(min(-lin(lb), 700)) > 1e300:
		lb = -math.inf
	t = torch.as_tensor
	if math.isinf(lb) and math.isinf(ub):
		return (
			lambda v: cinv(1 / (1 + torch.exp(-lin(t(v))))),
			lambda y: lin_inv(torch.log(cinv(t(y)) / (1 - cinv(t(y))))),
		)
	if math.isinf(ub):
		a = 1 / (1 - 1 / (1 + math.exp(-lin(lb))))
		b = 1 - a
		return (
			lambda v: cinv(a / (1 + torch.exp(-lin(t(v)))) + b),
			lambda y: lin_inv(torch.log(((cinv(t(y)) - b) / a) / (1 - ((cinv(t(y)) - b) / a)))),
		)
	if math.isinf(lb):
		a = 1 + math.exp(-lin(ub))
		return (
			lambda v: cinv(a / (1 + torch.exp(-lin(t(v))))),
			lambda y: lin_inv(torch.log((cinv(t(y)) / a) / (1 - (cinv(t(y)) / a)))),
		)
	el, eu = math.exp(-lin(lb)), math.exp(-lin(ub))
	a = (1 + eu + el + eu * el) / (el - eu)
	b = -a / (1 + el)
	return (
		lambda v: cinv(a / (1 + torch.exp(-lin(t(v)))) + b),
		lambda y: lin_inv(torch.log(((cinv(t(y)) - b) / a) / (1 - ((cinv(t(y)) - b) / a)))),
	)


@dataclass(frozen=True)
class Quantiles:
	lo: float  # 5%
	mid: float  # 50%
	hi: float  # 95%
	reason: str


def forecast_loss(
	model: LCPFN, losses: list[float | None], last_step: int, first_step: int = 1
) -> Quantiles | None:
	"""The loss at `last_step` as LC-PFN's 5%, 50% and 95% quantiles, from a series starting at `first_step`. None
	when there is nothing to condition on, or the seen steps already pass the last one."""
	pts = [(first_step + i, float(v)) for i, v in enumerate(losses) if v is not None and math.isfinite(v)]
	if len(pts) < 2 or last_step <= pts[-1][0]:
		return None
	if len(pts) > MAX_POINTS:
		keep = sorted({round(k * (len(pts) - 1) / (MAX_POINTS - 1)) for k in range(MAX_POINTS)})
		pts = [pts[k] for k in keep]
	steps = torch.tensor([p[0] for p in pts], dtype=torch.float32)
	ys = torch.tensor([p[1] for p in pts], dtype=torch.float32)
	# losses are non-negative; the soft range runs from below the best loss seen to the first one
	best, first = float(ys.min()), float(ys[0])
	soft_ub = max(first, best * 1.0001 + 1e-6)
	to, back = pfn_normalize(0.0, math.inf, best * 0.5, soft_ub, minimize=True)
	x = steps / last_step * X_MAX
	y = to(ys).clamp(0, 1).float()
	q = model.predict_quantiles(x, y, torch.tensor([float(X_MAX)]), [0.05, 0.5, 0.95])[0]
	# minimised: the normaliser is decreasing in the loss, so the 95% curve quantile is the 5% loss quantile
	# the bars run past [0, 1] (full support); past the normaliser's range a quantile means "no bound", so it is
	# held just inside, where the loss reads as very low or very high rather than NaN
	q = q.clamp(1e-6, 1 - 1e-6)
	lo, mid, hi = (float(back(v)) for v in (q[2], q[1], q[0]))
	why = (
		f"LC-PFN posterior on {len(pts)} loss values (steps {pts[0][0]} to {pts[-1][0]}), scaled so step {last_step} "
		"is the prior's last epoch"
	)
	return Quantiles(lo, mid, hi, why)
