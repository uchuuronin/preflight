"""Check each number a finding or the guide states against the record's raw probe numbers.

	checks, calls = check_text("Eval ran out of memory at 21.9 GB (20.4 GiB) after 189 s", record,
		client=ModelClient.baseten_spark_open())

Code does the checking; Spark Open only chooses. For every number in the text (with its unit:
bytes, seconds, a percentage, dollars, or a bare count) this lists the raw probe fields whose
value agrees with it to the precision it was written at, after unit conversion. One candidate:
supported. None: unsupported, flagged. Several: Spark Open picks the field the sentence means
from that list (strict JSON, one call per text, logged as purpose `claim_check`), and the code
checks the pick is one of them, so the model can't make a claim true. A number that isn't a
measurement (a step count in a fix, a year) can be marked `not_a_measurement` by the model; that
is reported as such, never as supported.
"""

import math
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

GB, GIB = 1e9, 2**30
UNITS = {
	"tb": ("bytes", 1e12),
	"gb": ("bytes", GB),
	"mb": ("bytes", 1e6),
	"kb": ("bytes", 1e3),
	"tib": ("bytes", 2**40),
	"gib": ("bytes", GIB),
	"mib": ("bytes", 2**20),
	"kib": ("bytes", 2**10),
	"ms": ("seconds", 1e-3),
	"s": ("seconds", 1.0),
	"sec": ("seconds", 1.0),
	"seconds": ("seconds", 1.0),
	"min": ("seconds", 60.0),
	"minutes": ("seconds", 60.0),
	"m": ("seconds", 60.0),
	"h": ("seconds", 3600.0),
	"hr": ("seconds", 3600.0),
	"hrs": ("seconds", 3600.0),
	"hours": ("seconds", 3600.0),
	"%": ("fraction", 0.01),
}
# not a number inside a name ("H100", "bf16", "Qwen2.5-1.5B", "CUDA 12.4" stays: a version is
# words), and the sign kept ("-12%")
_NUM = re.compile(
	r"(?<![\w.])(?P<sign>[-−])?(?P<usd>\$)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
	r"(?P<unit>TiB|GiB|MiB|KiB|TB|GB|MB|KB|ms|sec|seconds|s|minutes|min|m|hours|hrs|hr|h|%)?(?!\w|\.\d)"
)
_VERSION = re.compile(r"(?i)\b(cuda|rocm|python|torch|pytorch|transformers|version|v)\s*$")
# which raw fields can back which kind of number (by the field path's last part)
FIELD_KIND = (
	(
		"fraction",
		re.compile(r"(by_phase\.\w+|fraction|share|ratio|mfu|rate)$"),
	),  # before bytes: by_phase.optimizer
	("rate", re.compile(r"(_per_s|per_second|_items_per_s)$")),  # before seconds: tokens_per_s isn't a time
	("bytes", re.compile(r"(bytes|peak|reserved|weights|gradients|optimizer|activations|logits)$")),
	("seconds", re.compile(r"(_s|seconds|duration_s|wall_s)$")),
	("usd", re.compile(r"(usd|cost)$")),
)
_RATE_AFTER = re.compile(r"\s*(?:[A-Za-z]+\s*)?(?:/\s*s(?:ec)?\b|a second|per second)")  # "4,096 tokens/s"


@dataclass(frozen=True)
class Number:
	text: str  # as written, e.g. "21.9 GB"
	value: float  # in base units: bytes, seconds, a fraction, dollars, or the count
	kind: str  # bytes, seconds, fraction, usd, count
	rel: float  # the precision it was written at, as a relative tolerance


@dataclass
class ClaimCheck:
	number: str
	verdict: str  # supported, unsupported, not_a_measurement, ambiguous
	field: str | None = None
	field_value: float | None = None
	candidates: tuple[str, ...] = ()


def numbers_in(text: str) -> list[Number]:
	out = []
	text = text or ""
	for m in _NUM.finditer(text):
		if _VERSION.search(text[: m.start()]):
			continue  # "CUDA 12.4", "torch 2.5": a version, not a measurement
		raw = m["num"].replace(",", "")
		v = float(raw) * (-1 if m["sign"] else 1)
		decimals = len(raw.split(".")[1]) if "." in raw else 0
		# half a unit in the last written place, relative to the value (21.9 -> 0.05 / 21.9)
		rel = max(0.5 * 10**-decimals / abs(v), 1e-6) if v else 0.5 * 10**-decimals
		unit = (m["unit"] or "").lower()
		if m["usd"]:
			kind, scale = "usd", 1.0
		elif unit in UNITS:
			kind, scale = UNITS[unit]
		elif _RATE_AFTER.match(text, m.end()):
			kind, scale = "rate", 1.0  # "4,096 tokens/s" reads as a rate even with a word between
		else:
			kind, scale = "count", 1.0
		if kind == "count" and decimals == 0 and abs(v) < 10:
			continue  # "2 seeds", "step 1": too small to tell a measurement from words
		# GB is 10^9 bytes; a sentence written in GiB-as-GB is a different number and gets flagged
		out.append(Number(m.group(0).strip(), v * scale, kind, rel))
	return out


def flatten(obj, prefix: str = "") -> dict[str, float]:
	"""Every finite number in a probe (or any JSON-like value), by dotted path."""
	out: dict[str, float] = {}
	if isinstance(obj, BaseModel):
		obj = obj.model_dump(mode="json")
	if isinstance(obj, dict):
		for k, v in obj.items():
			out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
	elif isinstance(obj, list):
		for i, v in enumerate(obj[:256]):
			out.update(flatten(v, f"{prefix}[{i}]"))
	elif isinstance(obj, int | float) and not isinstance(obj, bool) and math.isfinite(obj):
		out[prefix] = float(obj)
	return out


def raw_numbers(record) -> dict[str, float]:
	"""The record's measured numbers: its probes and agent sample, never its predictions."""
	out: dict[str, float] = {}
	for p in record.probes:
		# knobs are settings, not measurements: "bf16" or "batch 16" is never backed by a knob value
		out.update(
			flatten(p.model_dump(mode="json", exclude={"kernels", "step_log", "knobs"}), f"probes.{p.id}")
		)
	if record.agent_sample is not None:
		out.update(flatten(record.agent_sample, "agent_sample"))
	return out


def _kind_of(path: str) -> str:
	last = path.rsplit(".", 1)[-1]
	for kind, rx in FIELD_KIND:
		if rx.search(path) or rx.search(last):
			return kind
	return "count"


def candidates(n: Number, raw: dict[str, float]) -> list[str]:
	"""Raw fields of the number's kind whose value rounds to it as written."""
	tol = max(n.rel, 1e-9)
	if n.value == 0:  # "0%": fields that are exactly zero
		return [path for path, v in raw.items() if _kind_of(path) == n.kind and v == 0]
	return [
		path
		for path, v in raw.items()
		if _kind_of(path) == n.kind and v and abs(v - n.value) <= tol * abs(n.value) + 1e-12
	]


class _Pick(BaseModel):
	number: str = Field(description="the number exactly as written in the text")
	field: str = Field(
		description="one of its candidate fields, or 'none' if no candidate is what the sentence means"
	)
	not_a_measurement: bool = Field(
		description="true when the number isn't a measured quantity (a setting, a step count in advice, a date)"
	)


class _Picks(BaseModel):
	picks: list[_Pick]


def check_text(text: str, record, *, client=None) -> tuple[list[ClaimCheck], list]:
	"""Each number in the text, checked against the record's raw probe numbers. Returns the
	checks and the model calls made (for the record's provenance)."""
	raw = raw_numbers(record)
	checks, ask = [], []
	for n in numbers_in(text):
		c = candidates(n, raw)
		if len(c) == 1:
			checks.append(ClaimCheck(n.text, "supported", c[0], raw[c[0]], tuple(c)))
		elif not c:
			checks.append(ClaimCheck(n.text, "unsupported"))
		else:
			checks.append(ClaimCheck(n.text, "ambiguous", candidates=tuple(c[:20])))
			ask.append(checks[-1])
	calls = []
	if ask and client is not None:
		listing = "\n".join(f"- {c.number}: " + ", ".join(c.candidates) for c in ask)
		msgs = [
			{
				"role": "system",
				"content": "You match numbers in a sentence about an ML job to the measured field "
				"each one reports. Answer only from the candidate lists given.",
			},
			{
				"role": "user",
				"content": f"Text: {text[:2000]}\n\nNumbers and their candidate fields:\n{listing}",
			},
		]
		try:
			picks, call = client.structured(msgs, _Picks, purpose="claim_check")
			calls.append(call)
		except Exception:  # the model's view is a refinement; the checks stand without it
			picks = _Picks(picks=[])
		by_number = {p.number: p for p in picks.picks}
		for c in ask:
			p = by_number.get(c.number)
			if p is None:
				continue
			if p.not_a_measurement:
				c.verdict = "not_a_measurement"
			elif p.field in c.candidates:  # the model may only choose among what the code found
				c.verdict, c.field, c.field_value = "supported", p.field, raw[p.field]
			elif p.field == "none":
				c.verdict = "unsupported"
	return checks, calls


def check_findings(record, *, client=None) -> tuple[dict[str, list[ClaimCheck]], list]:
	"""check_text over every finding's title and detail, by finding id."""
	out, calls = {}, []
	for f in record.findings:
		checks, cs = check_text(" ".join(x for x in (f.title, f.detail) if x), record, client=client)
		if checks:
			out[f.id] = checks
		calls += cs
	return out, calls
