"""Data repetition and quality exposure (main plan, Feature set: "Data repetition and quality exposure").

Repeating data is worth less than new data. Muennighoff et al. (2023, "Scaling Data-Constrained Language Models")
fit the value of U unique tokens seen for E epochs as effective tokens
	D' = U + U R* (1 - exp(-R / R*)),  with R = E - 1 repetitions and R* = 15.4 (their fitted constant),
so up to about 4 epochs is nearly as good as new data and past about 16 it barely helps. That prior treats every
token as equal quality. Beside it, exposure counts say how many times each quality bin's documents are seen for a
given token budget and mixture, and flag heavy repetition of the bins the user marked low quality, or a language
forced past 4 epochs.

The prior is cited, not fitted here. Proxy runs on the user's data are what calibrate it (validation row: the proxy
ranking agrees with full runs where both exist). Pure Python: it runs in Pyodide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MODEL_ID = "repetition-muennighoff@0.1.0"
R_STAR = 15.4  # cited: Muennighoff et al. 2023, fitted decay constant for repeated data
FLAG_EPOCHS = 4.0  # cited: same paper, up to ~4 epochs is close to unique data
FLAG_LOW_QUALITY = 2.0  # guess: a low-quality document seen more than twice is worth a warning


def effective_tokens(unique: float, epochs: float, r_star: float = R_STAR) -> float:
	"""Muennighoff's effective data for `unique` tokens seen `epochs` times (epochs >= 0; under 1 is a subset)."""
	if unique <= 0 or epochs <= 0:
		return 0.0
	if epochs <= 1:
		return unique * epochs
	r = epochs - 1
	return unique + unique * r_star * (1 - math.exp(-r / r_star))


def value_per_token(unique: float, epochs: float) -> float:
	"""Effective tokens per token trained on: 1 for unique data, falling as it repeats."""
	seen = unique * epochs
	return effective_tokens(unique, epochs) / seen if seen > 0 else 0.0


@dataclass(frozen=True)
class Bin:
	name: str  # a quality bin, a language, or both ("en/low")
	tokens: float  # unique tokens in the bin
	weight: float = 1.0  # sampling weight in the mixture; 1 everywhere means proportional to size
	low_quality: bool = False
	language: str | None = None


@dataclass(frozen=True)
class Exposure:
	name: str
	epochs: float  # times each document in the bin is seen
	effective_share: float  # effective tokens per token trained on, for this bin
	flag: str | None


def exposure(bins: list[Bin], budget_tokens: float) -> list[Exposure]:
	"""Epochs per bin when `budget_tokens` are drawn from the mixture, and the bins worth a warning."""
	mass = sum(b.tokens * b.weight for b in bins)
	if mass <= 0 or budget_tokens <= 0:
		raise ValueError("needs bins with tokens and a positive budget")
	out = []
	for b in bins:
		drawn = budget_tokens * b.tokens * b.weight / mass
		ep = drawn / b.tokens if b.tokens else 0.0
		flag = None
		if b.low_quality and ep > FLAG_LOW_QUALITY:
			flag = f"low-quality documents in {b.name} are each seen {ep:.1f} times"
		elif ep > FLAG_EPOCHS:
			flag = f"{b.name} is repeated {ep:.1f} times, past the ~{FLAG_EPOCHS:g} epochs where repeats stop helping much"
		out.append(Exposure(b.name, ep, value_per_token(b.tokens, ep), flag))
	return out


def by_language(bins: list[Bin], budget_tokens: float) -> dict[str, float]:
	"""Epochs per language: each language's drawn tokens over its unique tokens."""
	ex = {e.name: e for e in exposure(bins, budget_tokens)}
	drawn: dict[str, float] = {}
	unique: dict[str, float] = {}
	for b in bins:
		if b.language:
			drawn[b.language] = drawn.get(b.language, 0.0) + ex[b.name].epochs * b.tokens
			unique[b.language] = unique.get(b.language, 0.0) + b.tokens
	return {lang: drawn[lang] / unique[lang] for lang in drawn if unique[lang]}


def split_table(unique: float, budgets: list[float]) -> list[tuple[float, float, float]]:
	"""(budget, epochs, effective tokens) for each token budget over the same unique data: the data-size x epoch
	split the lens draws."""
	return [(t, t / unique, effective_tokens(unique, t / unique)) for t in budgets]
