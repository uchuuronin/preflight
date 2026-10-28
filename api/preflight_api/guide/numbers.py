"""Check every number the guide's text quotes against this turn's evidence.

Evidence: every number in this turn's tool results (and their differences and relative
changes, since "saves $4.60" or "35% cheaper" compare two results), the visitor's setting,
the numbers the record itself states (knob values, budget, run length) and the visitor's own
message. A quoted number passes if some evidence value matches it after unit conversion,
within the rounding its digits imply or 2%, whichever is looser.

Numbers inside names are not quotes: "H100", "bf16", "Qwen2.5-1.5B", "A100-SXM-80GB" are
skipped, as is anything that appears in the record's own vocabulary.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

GIB, MIB, KIB = 2**30, 2**20, 2**10
REL_TOL = 0.02  # source: guess, the display rounds to 3 significant figures (fmt.ts)
MAX_PAIRS_FROM = 80  # bound on derived differences: 80 values -> 3,160 pairs

_UNITS: dict[str, list[float]] = {
	"%": [0.01, 1.0],
	"percent": [0.01, 1.0],
	"gib": [GIB],
	"gb": [1e9, GIB],
	"mib": [MIB],
	"mb": [1e6, MIB],
	"kib": [KIB],
	"tib": [2**40],
	"tb": [1e12, 2**40],
	"ms": [1e-3],
	"s": [1.0],
	"sec": [1.0],
	"secs": [1.0],
	"second": [1.0],
	"seconds": [1.0],
	"min": [60.0],
	"mins": [60.0],
	"minute": [60.0],
	"minutes": [60.0],
	"h": [3600.0],
	"hr": [3600.0],
	"hrs": [3600.0],
	"hour": [3600.0],
	"hours": [3600.0],
	"day": [86400.0],
	"days": [86400.0],
	"k": [1e3],
	"m": [1e6, 60.0],
	"million": [1e6],
	"bn": [1e9],
	"x": [1.0],
	"×": [1.0],
}

_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_TOKEN = re.compile(
	r"(?P<pre>(?:US\$|\$|£|€)?)"
	r"(?<![\w.\-])(?P<num>" + _NUM + r")"
	r"(?P<sp>\s?)(?P<unit>%|×|[A-Za-z]+)?"
)
_DURATION = re.compile(
	r"(?<![\w.])(\d+(?:\.\d+)?)\s?h(?:ours?|rs?)?\s+(\d+(?:\.\d+)?)\s?min(?:utes?|s)?\b", re.I
)
_WORD_NUMBERS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}


@dataclass(frozen=True)
class Quote:
	text: str
	values: tuple[float, ...]  # the quote in base units, one per plausible unit reading
	half_unit: float  # half of the last displayed digit, in the same base units (per reading)


def _flatten(x: Any, out: list[float]) -> None:
	if isinstance(x, bool) or x is None:
		return
	if isinstance(x, int | float):
		if math.isfinite(x):
			out.append(float(x))
	elif isinstance(x, dict):
		for v in x.values():
			_flatten(v, out)
	elif isinstance(x, list | tuple):
		for v in x:
			_flatten(v, out)
	elif isinstance(x, str):
		for m in re.finditer(_NUM, x):
			with contextlib.suppress(ValueError):
				out.append(float(m.group(0).replace(",", "")))


def evidence_from(*sources: Any) -> list[float]:
	vals: list[float] = []
	for s in sources:
		_flatten(s, vals)
	base = sorted(set(vals), key=abs, reverse=True)[:MAX_PAIRS_FROM]
	derived: list[float] = []
	for i, a in enumerate(base):
		for b in base[i + 1 :]:
			d = abs(a - b)
			if d:
				derived.append(d)
				for ref in (a, b):
					if ref:
						derived.append(d / abs(ref))
	return [*set(vals), *derived]


def quotes(text: str, vocabulary: Iterable[str] = ()) -> list[Quote]:
	vocab = {v.lower() for v in vocabulary if v}
	found: list[Quote] = []
	spans: list[tuple[int, int]] = []
	for m in _DURATION.finditer(text):
		h, mins = float(m.group(1)), float(m.group(2))
		found.append(Quote(m.group(0), (h * 3600 + mins * 60,), 30.0))
		spans.append(m.span())
	for m in _TOKEN.finditer(text):
		if any(a <= m.start("num") < b for a, b in spans):
			continue
		num_s = m.group("num")
		unit_raw = (m.group("unit") or "").lower()
		after = text[m.end("num") : m.end("num") + 1]
		# a number glued to letters that aren't a unit is part of a name (bf16, 7B-Instruct, H100)
		if unit_raw and not m.group("sp") and unit_raw not in _UNITS:
			continue
		if unit_raw and m.group("sp") and unit_raw not in _UNITS:
			unit_raw = ""
		if after in "-_/" and after:
			continue
		start = m.start("num")
		before = text[max(0, start - 1) : start]
		if before.isalpha():
			continue
		token = (m.group("pre") + num_s + (m.group("sp") + m.group("unit") if unit_raw else "")).strip()
		if token.lower() in vocab or num_s.lower() in vocab:
			continue
		value = float(num_s.replace(",", ""))
		decimals = len(num_s.split(".")[1]) if "." in num_s else 0
		half = 0.5 * 10 ** (-decimals)
		mults = _UNITS.get(unit_raw, [1.0])
		found.append(Quote(token, tuple(value * k for k in mults), half * max(mults)))
	return found


def supported(q: Quote, evidence: list[float]) -> bool:
	for v in q.values:
		tol = max(abs(v) * REL_TOL, q.half_unit)
		if any(abs(abs(v) - abs(e)) <= tol for e in evidence):
			return True
	return False


def unsupported(text: str, evidence: list[float], vocabulary: Iterable[str] = ()) -> list[str]:
	"""Quoted numbers no evidence supports. Small counting words and ordinals are not checked."""
	bad: list[str] = []
	for q in quotes(text, vocabulary):
		if q.values and all(abs(v) <= 10 and float(v).is_integer() for v in q.values) and q.text.isdigit():
			# "2 knobs", "step 1 of 3": counts the guide may use freely
			continue
		if not supported(q, evidence):
			bad.append(q.text)
	return bad


def record_numbers(rec: dict) -> list[float]:
	"""Numbers the record itself states: knob values and domains, budget, run length, dataset."""
	vals: list[float] = []
	for k in rec.get("knobs") or []:
		_flatten(
			[
				k.get("current"),
				(k.get("domain") or {}).get("values"),
				(k.get("domain") or {}).get("min"),
				(k.get("domain") or {}).get("max"),
			],
			vals,
		)
	job = rec.get("job") or {}
	_flatten(job.get("constraints"), vals)
	sm = job.get("script_map") or {}
	_flatten([sm.get("training"), (sm.get("data") or {}).get("n_examples")], vals)
	env = rec.get("env") or {}
	_flatten((env.get("gpu") or {}).get("memory_bytes"), vals)
	return vals


def record_vocabulary(rec: dict) -> set[str]:
	words: set[str] = set()
	for k in rec.get("knobs") or []:
		for v in [k.get("current"), *((k.get("domain") or {}).get("values") or [])]:
			if isinstance(v, str):
				words.add(v)
	job = rec.get("job") or {}
	for s in (
		job.get("title"),
		(job.get("script_map") or {}).get("model", {}).get("name")
		if isinstance((job.get("script_map") or {}).get("model"), dict)
		else None,
	):
		if isinstance(s, str):
			words.update(re.findall(r"[\w.\-]+", s))
	gpu = ((rec.get("env") or {}).get("gpu") or {}).get("name")
	if isinstance(gpu, str):
		words.add(gpu)
	words.update(_WORD_NUMBERS)
	return words
