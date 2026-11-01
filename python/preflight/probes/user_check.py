"""The probe-user startup check on a plain host (the AMD droplet), as one command:

	python -m preflight.probes.user_check --uid 61000 --gpu

Runs the deploy gate's probe-user script (probes/local.py PROBE_USER_SCRIPT) exactly as a probe
launches (interpreter, scrubbed env, uid, GPU groups), with the uid pinned: on the droplet the code
and its venv sit under /root, so `preflight.cli.main run` there runs its probes as root
(uid_can_run_this_python), and only a pinned uid exercises the probe-user path. Checks the uid and
user name, the video/render groups, a CPU torch.compile, the child's imports and, with --gpu,
/dev/kfd, a matmul and a torch.compile on the GPU. Prints one JSON line; exits 0 only if it all passed.
"""

import argparse
import json
import os
import sys

from preflight.probes.local import PROBE_USER, _extra_groups, probe_user_check, uid_can_run_this_python

CHILD_IMPORTS = ("preflight.recorder.runtime", "preflight.recorder.run")  # what a probe child loads


def verdict(r: dict, uid: int | None, gpu: bool, want_groups: list[int]) -> list[str]:
	"""What failed, empty when the check passed."""
	bad = []
	if r.get("exit") != 0:
		return [f"exit {r.get('exit')}: {r.get('stderr', '')[-300:]}"]
	if uid is not None and r.get("uid") != uid:
		bad.append(f"ran as uid {r.get('uid')}, not {uid}")
	if uid is not None and r.get("user") != PROBE_USER:
		bad.append(f"user name {r.get('user')!r}")
	missing = [g for g in want_groups if g not in r.get("groups", [])]
	if uid is not None and missing:
		bad.append(f"GPU groups {missing} missing")
	if r.get("compiled") is None:
		bad.append("no CPU compile")
	if r.get("imported", []) != list(CHILD_IMPORTS):
		bad.append(f"imported {r.get('imported')}")
	if gpu:
		if os.path.exists("/dev/kfd") and not r.get("kfd"):
			bad.append("/dev/kfd not opened")
		if not r.get("gpu"):
			bad.append("no GPU matmul")
		if r.get("gpu_compiled") is None:
			bad.append("no GPU compile")
	return bad


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="python -m preflight.probes.user_check")
	ap.add_argument("--uid", type=int, default=61000, help="the probe uid to pin when run as root (>= 1000)")
	ap.add_argument("--gpu", action="store_true", help="also open the GPU and compile on it")
	ap.add_argument("--timeout-s", type=int, default=600)
	a = ap.parse_args(argv)
	if a.uid < 1000:
		ap.error("--uid must be >= 1000")
	root = os.geteuid() == 0
	if root:
		os.environ["PREFLIGHT_PROBE_UID"] = str(a.uid)
	uid = a.uid if root else None
	want = _extra_groups()  # video, render: what a probe's Popen adds
	r = probe_user_check(("+compile", *(("+gpu",) if a.gpu else ()), *CHILD_IMPORTS), timeout_s=a.timeout_s)
	bad = verdict(r, uid, a.gpu, want)
	report = {"ok": not bad, "failed": bad, "root": root, "gpu_groups": want,
		"cli_run_probes_as": "a fresh uid" if (not root or uid_can_run_this_python()) else "root (venv unreachable)",
		**r}  # fmt: skip
	print("USER_CHECK " + json.dumps(report, default=str))
	return 0 if not bad else 1


if __name__ == "__main__":
	sys.exit(main())
