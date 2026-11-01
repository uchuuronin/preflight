"""Divergence warning: will this run diverge later, from the steps seen so far (plan: "AUROC 0.85+, 100+ steps early").

Trained on the corpus's divergence sweep (short small-GPT runs labelled after the fact by D9, corpus/worker.py), and
read at inference from a probe's stability series (`Probe.stability.series`). At every step it sees, the features
are the loss against its own best, the loss's recent slope, the grad norm against its own median, the attention and
output logit levels and their growth, the update-to-weight ratio, non-finite and loss-scale events, and the run's
settings (learning rate, warmup, init, precision, qk-norm, z-loss, beta2, clipping). A logistic model maps them to
P(the run diverges later). Pure numpy, so it runs in Pyodide.

Probabilities are calibrated on held-out runs (scripts/fit_divergence.py). A probability is shown only where its
reliability bin verified on held-out runs (at least MIN_BIN runs, and the shown P inside the Wilson interval of the
bin's observed rate); elsewhere the readout abstains with "uncertain", and the threshold rules
(D6, docs/decisions/engine-settings.md) keep warning. With fewer than WINDOW steps it abstains.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

MODEL_ID = "divergence-logistic@0.1.0"
WINDOW = 25  # cited: S3 D9's rolling window, so the features see what the label sees
MIN_BIN = 10  # cited: the P(OOM) display rule (Chiya, Oct 31): a bin shows a number from 10 points
SERIES = ("loss", "grad_norm", "update_ratio", "max_attn_logit", "output_logit_norm", "loss_scale")
# the probe's names for the same series (schema Stability.series)
ALIASES = {"update_to_weight": "update_ratio"}

FEATURES = (
	"loss_vs_best",
	"loss_slope",
	"grad_vs_median",
	"attn_logit",
	"attn_slope",
	"out_logit",
	"out_slope",
	"update_ratio",
	"nonfinite",
	"scale_drops",
	"log_lr",
	"no_warmup",
	"log_init_std",
	"fp16",
	"bf16",
	"qk_norm",
	"z_loss",
	"beta2_high",
	"no_clip",
)


def _f(v) -> float | None:
	if v is None:
		return None
	if isinstance(v, (list, tuple)):
		vals = [x for x in (_f(x) for x in v) if x is not None]
		return float(np.mean(vals)) if vals else None
	x = float(v)
	return x if math.isfinite(x) else None


def _clean(series: dict) -> dict[str, list[float | None]]:
	out = {}
	for k, v in (series or {}).items():
		out[ALIASES.get(k, k)] = [_f(x) for x in v]
	return out


def _slope(ys: list[float]) -> float:
	ys = [y for y in ys if y is not None and y > 0]
	if len(ys) < 3:
		return 0.0
	x = np.arange(len(ys), dtype=float)
	return float(np.polyfit(x, np.log(ys), 1)[0] * WINDOW)  # log change over one window


def _log(v: float | None, floor: float = 1e-8) -> float:
	return math.log(max(v, floor)) if v is not None else 0.0


# the user's knob ids for the sweep's settings (engine/knobs.py); review, Nov 1: inference once read none of them
KNOB_NAMES = {
	"warmup_steps": "warmup",
	"learning_rate": "lr",
	"max_grad_norm": "grad_clip",
	"adam_beta2": "beta2",
	"dtype": "precision",
	"torch_dtype": "precision",
	"ptdtype": "precision",
}


def config_features(cfg: dict) -> list[float | None]:
	"""The run's settings; None where a setting isn't known, which the model reads as the training average."""
	c = dict(cfg)
	for k, v in cfg.items():
		if k in KNOB_NAMES and c.get(KNOB_NAMES[k]) is None:
			c[KNOB_NAMES[k]] = v

	def has(k):
		return k in c and c[k] is not None

	prec = str(c["precision"]).lower().removeprefix("torch.") if has("precision") else None
	lr = _f(c.get("lr")) if has("lr") else None
	init = _f(c.get("init_std")) if has("init_std") else None
	return [
		_log(lr) if lr else None,
		(0.0 if c["warmup"] else 1.0) if has("warmup") else None,
		_log(init) if init else None,
		(1.0 if prec in ("fp16", "float16") else 0.0) if prec else None,
		(1.0 if prec in ("bf16", "bfloat16") else 0.0) if prec else None,
		(1.0 if c["qk_norm"] else 0.0) if has("qk_norm") else None,
		(1.0 if (_f(c["z_loss"]) or 0.0) > 0 else 0.0) if has("z_loss") else None,
		(1.0 if (_f(c["beta2"]) or 0.0) > 0.99 else 0.0) if has("beta2") else None,
		# the fit writes the sweep's None (no clipping) as 0; a user's unset knob is unknown
		(1.0 if str(c["grad_clip"]).lower() in ("0", "0.0", "none", "false") else 0.0)
		if has("grad_clip")
		else None,
	]


class Context:
	"""One run's series, cleaned once, with the causal running quantities every step's features read."""

	def __init__(self, series: dict):
		s = _clean(series)
		self.s = s
		loss = s.get("loss") or []
		self.loss = loss
		n = len(loss)
		self.med = []
		for i in range(n):
			w = [x for x in loss[max(0, i - WINDOW + 1) : i + 1] if x is not None]
			self.med.append(float(np.median(w)) if w else math.nan)
		self.best = []
		b = math.inf
		for m in self.med:
			if math.isfinite(m):
				b = min(b, m)
			self.best.append(b)
		self.missing = list(itertools.accumulate(1 if x is None else 0 for x in loss))
		nf = s.get("nonfinite_grads") or []
		self.nonfinite = list(itertools.accumulate(1 if x else 0 for x in nf))
		scale = s.get("loss_scale") or []
		drops, prev, acc = [], None, 0
		for x in scale:
			if x is not None and prev is not None and x < prev:
				acc += 1
			prev = x if x is not None else prev
			drops.append(acc)
		self.drops = drops

	def tail(self, name: str, t: int) -> list[float | None]:
		return (self.s.get(name) or [])[max(0, t - WINDOW + 1) : t + 1]

	def last(self, name: str, t: int) -> float | None:
		vals = [x for x in self.tail(name, t) if x is not None]
		return vals[-1] if vals else None


def features_at(ctx: Context, t: int, cfg: dict) -> list[float] | None:
	"""Features at step t, from steps 0..t only. None before WINDOW steps."""
	if t + 1 < WINDOW or t >= len(ctx.loss) or not math.isfinite(ctx.best[t]):
		return None
	cur = ctx.med[t] if math.isfinite(ctx.med[t]) else ctx.best[t]
	g = [x for x in (ctx.s.get("grad_norm") or [])[: t + 1] if x is not None]
	grad_vs = _log(g[-1]) - _log(float(np.median(g))) if g else 0.0

	def upto(xs: list[int]) -> int:
		return xs[min(t, len(xs) - 1)] if xs else 0

	return [
		_log(cur) - _log(ctx.best[t]),
		_slope([x for x in ctx.med[max(0, t - WINDOW + 1) : t + 1] if math.isfinite(x)]),
		grad_vs,
		_log(ctx.last("max_attn_logit", t), 1e-3),
		_slope(ctx.tail("max_attn_logit", t)),
		_log(ctx.last("output_logit_norm", t), 1e-3),
		_slope(ctx.tail("output_logit_norm", t)),
		_log(ctx.last("update_ratio", t)),
		math.log1p(upto(ctx.missing) + upto(ctx.nonfinite)),
		math.log1p(upto(ctx.drops)),
		*config_features(cfg),
	]


def step_features(series: dict, t: int, cfg: dict) -> list[float] | None:
	"""Features at step t from a raw series (builds the context; use Context and features_at for many steps)."""
	return features_at(Context(series), t, cfg)


# ---------------------------------------------------------------- the model


@dataclass(frozen=True)
class Model:
	mean: tuple[float, ...]
	scale: tuple[float, ...]
	coef: tuple[float, ...]
	intercept: float
	# held-out reliability: (lo, hi, n, predicted_mean, observed) per bin, from out-of-fold runs; a bin verifies
	# when it has MIN_BIN runs and the shown P falls inside the Wilson interval of the bin's observed rate
	bins: tuple[tuple[float, float, int, float, float], ...] = ()
	threshold: float = 0.5  # the warning line, set on training folds for at most 10% false alarms
	report: str = ""

	def prob(self, x: list[float | None]) -> float:
		# an unknown setting sits at the training average, so it moves P neither way
		filled = [m if v is None else v for v, m in zip(x, self.mean, strict=True)]
		z = (np.asarray(filled, dtype=float) - np.asarray(self.mean)) / np.asarray(self.scale)
		return float(1.0 / (1.0 + math.exp(-(float(z @ np.asarray(self.coef)) + self.intercept))))

	def verified(self, p: float) -> bool:
		from preflight.engine.conformal import wilson

		for lo, hi, n, _pred, obs in self.bins:
			if lo <= p < hi or (hi >= 1.0 and p == 1.0):
				if n < MIN_BIN:
					return False
				w_lo, w_hi = wilson(round(obs * n), n)
				return w_lo <= p <= w_hi
		return False


def fit(
	x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
	"""L2 logistic regression by Newton steps on standardised features: (mean, scale, coef, intercept)."""
	mean = x.mean(0)
	scale = x.std(0)
	scale[scale < 1e-9] = 1.0
	z = (x - mean) / scale
	a = np.hstack([z, np.ones((len(z), 1))])
	w = np.zeros(a.shape[1])
	reg = np.full(a.shape[1], l2)
	reg[-1] = 0.0
	for _ in range(iters):
		p = 1 / (1 + np.exp(-np.clip(a @ w, -30, 30)))
		grad = a.T @ (p - y) + reg * w
		h = (a * (p * (1 - p))[:, None]).T @ a + np.diag(reg + 1e-9)
		step = np.linalg.solve(h, grad)
		w -= step
		if np.abs(step).max() < 1e-8:
			break
	return mean, scale, w[:-1], float(w[-1])


def warn(model: Model | None, series: dict, cfg: dict) -> dict:
	"""The readout for a probe: {status, point, reason} for risk.divergence."""
	if model is None:
		why = (
			"not used: the warning comes from the threshold rules (D6); the classifier missed its held-out target "
			"(AUROC 0.66 against 0.85, corpus/reports/divergence_nov1.json)"
		)
		return {"status": "abstain", "point": None, "reason": why}
	s = _clean(series)
	n = len(s.get("loss") or [])
	if n < WINDOW:
		return {
			"status": "abstain",
			"point": None,
			"reason": f"only {n} steps logged; the warning needs {WINDOW}",
		}
	ctx = Context(series)
	ps = [model.prob(f) for t in range(WINDOW - 1, n) if (f := features_at(ctx, t, cfg)) is not None]
	if not ps:
		return {"status": "abstain", "point": None, "reason": "no usable loss in the probe"}
	p = max(ps)
	if not model.verified(p):
		return {
			"status": "abstain",
			"point": None,
			"reason": "uncertain: held-out runs at this level are too few or disagree; the threshold rules still watch",
		}
	word = "likely to diverge" if p >= model.threshold else "unlikely to diverge"
	return {"status": "predicted", "point": p, "reason": f"{word} later in the run, from {n} probe steps"}
