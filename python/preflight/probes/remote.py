"""Inside the GPU sandbox: read a ProbeSpec on stdin, run it, print the result line.

	python -m preflight.probes.remote < spec.json

The sandbox holds no secrets and can reach only the egress allowlist; run_probe
adds its own isolation (scrubbed env, unprivileged uid, limits) around the
visitor's code. One process per probe point; the sandbox stays up between
points of the same job on the same GPU.
"""

import os
import sys
from pathlib import Path

from preflight.probes.local import run_probe
from preflight.runner.wire import result_to_wire, spec_from_wire

WEIGHTS_HUB = Path(os.environ.get("PREFLIGHT_WEIGHTS_HUB", "/weights/hf/hub"))


def link_weights(src: Path = WEIGHTS_HUB) -> int:
	"""Expose the read-only weights volume through the writable HF cache, one symlink per model.

	HF_HOME stays writable (locks, models that aren't prefetched download there through the
	egress allowlist); prefetched weights are read in place and can't be changed by the script.
	"""
	home = os.environ.get("HF_HOME")
	if not home:
		return 0
	dst = Path(home) / "hub"
	dst.mkdir(parents=True, exist_ok=True)
	uid = os.environ.get("PREFLIGHT_PROBE_UID")
	for d in (Path(home), dst):
		if uid and uid.isdigit() and os.geteuid() == 0:
			os.chown(
				d, int(uid), int(uid)
			)  # the job's one probe uid owns the cache: shared across its points
		else:
			d.chmod(0o1777)  # any probe uid may write; sticky, so one can't delete another's files
	if not src.is_dir():
		return 0
	n = 0
	for model in src.iterdir():
		if model.name.startswith(("models--", "datasets--")) and not (dst / model.name).exists():
			(dst / model.name).symlink_to(model, target_is_directory=True)
			n += 1
	return n


def main() -> int:
	link_weights()
	spec = spec_from_wire(sys.stdin.read())
	result = run_probe(spec)
	sys.stdout.write(result_to_wire(result) + "\n")
	sys.stdout.flush()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
