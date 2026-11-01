"""What crosses between the worker and the GPU sandbox that runs a job's probes.

The worker sends a ProbeSpec as JSON on the sandbox's stdin; the sandbox runs
`python -m preflight.probes.remote` (run_probe, with its own per-probe
isolation) and prints one line: RESULT_PREFIX + JSON. The sandbox runs the
visitor's code, so its output is untrusted here exactly as the probe child's
is inside run_probe: everything is validated, oversized output is cut, and a
result that doesn't check out becomes a recorder error, never a crash.
"""

import json
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from preflight.probes import untrusted
from preflight.probes.local import ProbeResult, ProbeSpec
from preflight.schema import Env, Probe

RESULT_PREFIX = "PREFLIGHT_RESULT "
MAX_RESULT_BYTES = 8 * 2**20
MAX_LOG_TAIL = 20_000


def spec_to_wire(spec: ProbeSpec) -> str:
	d = asdict(spec)
	d["out_dir"] = None  # the sandbox keeps its own artifacts
	return json.dumps(d)


def spec_from_wire(text: str) -> ProbeSpec:
	d = json.loads(text)
	d["out_dir"] = None
	if "gpu_fallback" in d:
		d["gpu_fallback"] = tuple(d["gpu_fallback"] or ())
	return ProbeSpec(**d)


def result_to_wire(r: ProbeResult) -> str:
	return RESULT_PREFIX + json.dumps(
		{
			"probe": json.loads(r.probe.model_dump_json(exclude_none=True)),
			"env": json.loads(r.env.model_dump_json(exclude_none=True)) if r.env else None,
			"arch": r.arch,
			"log_tail": r.log_tail[-MAX_LOG_TAIL:],
			"kernels_error": r.kernels_error,
			"gpu_unresolved": r.gpu_unresolved,
		}
	)


def _failed(spec: ProbeSpec, why: str, log_tail: str = "") -> ProbeResult:
	probe = Probe(
		id=spec.probe_id, knobs=spec.knobs, status="error", steps=0, error=why, error_code="recorder"
	)
	return ProbeResult(probe=probe, env=Env(backend="cpu", provider="modal"), summary={}, kernels=None, arch=None,
		trace_file=None, log_tail=log_tail[-MAX_LOG_TAIL:], workdir=Path("."))  # fmt: skip


def result_from_wire(spec: ProbeSpec, stdout: str) -> ProbeResult:
	"""Parse the sandbox's output for one probe; anything malformed is a recorder error on that probe."""
	if len(stdout) > MAX_RESULT_BYTES:
		stdout = stdout[-MAX_RESULT_BYTES:]
	line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith(RESULT_PREFIX)), None)
	if line is None:
		return _failed(spec, "The probe sandbox ended without a result", stdout)
	try:
		d = json.loads(line[len(RESULT_PREFIX) :])
		if not isinstance(d, dict):
			raise ValueError("not an object")
		probe = Probe.model_validate(d.get("probe"))
		if probe.id != spec.probe_id:
			raise ValueError("result for another probe")
		probe = probe.model_copy(
			update={"knobs": spec.knobs}
		)  # the worker decided the knobs, not the sandbox
		env = Env.model_validate(d["env"]) if d.get("env") else Env(backend="cpu", provider="modal")
	except (ValueError, ValidationError, TypeError, KeyError):
		return _failed(spec, "The probe's measurements could not be read")
	kernels = probe.kernels
	if kernels is not None:  # re-check the table with the parent's rules (row count, name length)
		kernels, _ = untrusted.kernel_table(json.loads(kernels.model_dump_json(exclude_none=True)))
		probe = probe.model_copy(update={"kernels": kernels})
	log_tail = d.get("log_tail") if isinstance(d.get("log_tail"), str) else ""
	err = d.get("kernels_error")
	unresolved = d.get("gpu_unresolved")
	return ProbeResult(
		probe=probe,
		env=env,
		summary={},
		kernels=kernels,
		arch=untrusted.model_shape(d.get("arch")),
		trace_file=None,
		log_tail=log_tail[-MAX_LOG_TAIL:],
		workdir=Path("."),
		kernels_error=err[:200] if isinstance(err, str) else None,
		gpu_unresolved=unresolved[:80] if isinstance(unresolved, str) else None,
	)
