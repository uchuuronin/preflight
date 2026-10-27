"""Compact encodings for grids: float16 arrays and bit masks as base64 strings.

The TypeScript side decodes the same bytes (little-endian IEEE half floats,
LSB-first bit masks), so keep these two functions boring.
"""

import base64
import warnings

import numpy as np

MAX_GRID_CELLS = 16_384
MAX_RECORD_GZIP_BYTES = 1_000_000


F16_MAX = 65_504.0
_HEADROOM = 2_048.0  # keep scaled maxima well under F16_MAX so hi bounds never overflow
_TINY = 1e-2  # below this the largest value has lost digits to float16's subnormal range


def choose_scale(*arrays) -> float:
	"""Smallest power of two that keeps every value under the float16 headroom.

	Stored values are real / scale; readers multiply back. Powers of two keep
	the division exact, and float16's relative precision (~0.05%) is the same
	at every magnitude, so scaling loses nothing.
	"""
	finite = [np.abs(np.asarray(a, dtype=np.float64)) for a in arrays if np.size(a)]
	finite = [a[np.isfinite(a)] for a in finite]
	top = max((float(a.max()) for a in finite if a.size), default=0.0)
	if top == 0.0 or _TINY <= top <= _HEADROOM:
		return 1.0
	# too big for float16, or so small it would flush to zero: rescale either way
	return float(2.0 ** np.ceil(np.log2(top / _HEADROOM)))


def encode_f16(values, scale: float = 1.0) -> str:
	arr = np.asarray(values, dtype=np.float64).ravel() / scale
	with warnings.catch_warnings():
		# values past 65504 become inf on purpose; the reader treats inf as "off the scale"
		warnings.simplefilter("ignore", RuntimeWarning)
		half = arr.astype("<f2")
	return base64.b64encode(half.tobytes()).decode("ascii")


def decode_f16(data: str, n: int | None = None) -> np.ndarray:
	arr = np.frombuffer(base64.b64decode(data), dtype="<f2").astype(np.float64)
	if n is not None and arr.size != n:
		raise ValueError(f"expected {n} float16 values, got {arr.size}")
	return arr


def f16_len(data: str) -> int:
	return len(base64.b64decode(data)) // 2


def encode_mask(mask) -> str:
	bits = np.packbits(np.asarray(mask, dtype=bool).ravel(), bitorder="little")
	return base64.b64encode(bits.tobytes()).decode("ascii")


def decode_mask(data: str, n: int) -> np.ndarray:
	raw = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
	bits = np.unpackbits(raw, bitorder="little")
	if bits.size < n or bits.size - n >= 8:
		raise ValueError(f"mask holds {bits.size} bits, expected {n}")
	return bits[:n].astype(bool)


def mask_len_ok(data: str, n: int) -> bool:
	return len(base64.b64decode(data)) == (n + 7) // 8
