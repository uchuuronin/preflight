"""GPU identity for records: canonical names and peak FLOP/s.

B's engine (engine/gpus.py) owns the GPU table. The recorder uses it when it is
importable and falls back to A's minimal peak table and the raw device name
otherwise, so a probe never fails because the table is missing a GPU. A name
the table doesn't know is kept as reported and flagged (raw is returned), so
the pipeline can file it for B instead of guessing.
"""

from collections.abc import Callable

from . import gpus as _minimal

Resolver = Callable[[str, int | None, int | None], str]


def _bs_resolver() -> Resolver | None:
	try:
		from preflight.engine.gpus import resolve
	except ImportError:
		return None
	return resolve


def _bs_spec():
	try:
		from preflight.engine.gpus import get_gpu, mig_profile
	except ImportError:
		return None
	return get_gpu, mig_profile


_UNSET = object()


def canonical_gpu_name(
	raw: str,
	memory_bytes: float | None,
	compute_units: int | None,
	*,
	arch: str | None = None,
	resolver=_UNSET,
) -> tuple[str, str | None]:
	"""(name for env.gpu.name, raw name when the table didn't know it, else None).

	ROCm often reports a generic marketing name; its gfx arch (gcnArchName, e.g.
	"gfx942:sramecc+:xnack-") is tried next. A MIG slice keeps its slice in the
	name (B's spec for it is scaled to its share of the GPU); its memory is the
	slice's, so it is not passed as a hint that would pick a smaller full GPU.
	With no table at all the raw name stands and nothing is flagged: there was
	nothing to look it up in.
	"""
	resolve = _bs_resolver() if resolver is _UNSET else resolver
	if resolve is None:
		return raw, None
	spec = _bs_spec() if resolver is _UNSET else None
	if spec is not None:
		get_gpu, mig_profile = spec
		if mig_profile(raw):
			try:
				return get_gpu(raw).name, None
			except (KeyError, ValueError):
				return raw, raw
	mem = int(memory_bytes) if memory_bytes else None
	tries = [raw]
	if arch:
		tries.append(arch.split(":", 1)[0])
	for name in tries:
		try:
			return resolve(name, mem, compute_units), None
		except (KeyError, ValueError):
			continue
	return raw, raw


def _spec_peak(spec, precision: str) -> float | None:
	"""B's peak_flops semantics on a spec: fp32 without TF32, tf32 falling back to fp32."""
	p = precision.lower()
	try:
		if p == "fp32":
			return spec.peak_flops("fp32", allow_tf32=False)
		if p == "tf32":
			return spec.peak_flops("tf32") or spec.peak_flops("fp32", allow_tf32=False)
		return spec.peak_flops(p)
	except ValueError:
		return None


def peak(
	gpu_name: str | None,
	precision: str,
	memory_bytes: float | None = None,
	*,
	device_name: str | None = None,
	table=_UNSET,
) -> float | None:
	"""Dense peak FLOP/s for MFU.

	B's table decides whenever it knows the GPU, including None for a precision
	the GPU has no fast path for (T4 bf16); A's minimal table answers only when
	B's module isn't there or doesn't know the GPU. device_name is the raw name
	torch reported: a MIG slice is looked up by it, so its peak is the slice's.
	"""
	if table is not _UNSET:  # tests: a function with B's peak_flops signature, or None for "no B"
		if table is None or not gpu_name:
			return _minimal.peak_flops(gpu_name, precision)
		return table(gpu_name, precision, int(memory_bytes) if memory_bytes else None)
	spec_fns = _bs_spec()
	if spec_fns is not None and gpu_name:
		get_gpu, mig_profile = spec_fns
		try:
			if device_name and mig_profile(device_name):
				return _spec_peak(get_gpu(device_name), precision)
			return _spec_peak(get_gpu(gpu_name, int(memory_bytes) if memory_bytes else None), precision)
		except KeyError:
			pass
	return _minimal.peak_flops(gpu_name, precision)
