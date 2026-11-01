"""Install a submission's requirements.txt into the probe before its script runs.

Runs in the probe child (the visitor's uid, the sandbox's egress allowlist: PyPI
and Hugging Face only), so it can do nothing the script itself couldn't. The
probe image's pinned stack (torch, transformers, peft, trl, ...) always wins:
a line naming a package the image already has is skipped and logged, and every
installed package is passed to pip as a constraint so nothing the script needs
can upgrade it underneath the recorder. Only plain `name[extras] <specifiers>`
lines from PyPI are installed: options, URLs, VCS and local paths are skipped
with a note, and only wheels are accepted (no build step, no compiler wait).

pip runs with --prefix, not --target: --target makes pip ignore everything
already installed, so a requirement that depends on torch would pull torch and
its CUDA wheels in again (and an image torch with a local label, +cu121 or
+rocm6.2, couldn't satisfy the constraint at all). With --prefix pip sees the
image's packages and installs only what is missing. The constraints carry
public versions only (`torch==2.5.1` matches the image's `2.5.1+cu121`).

A job's probes share one install when PREFLIGHT_DEPS_DIR is set (the CLI and
the Modal sandbox set it): the first probe installs into a directory keyed by
the requirements and the image's packages, later ones reuse it.
"""

import hashlib
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MAX_REQUIREMENTS = 50  # guess: well above the zoo's and the notebooks we've seen; the bug zoo checks it
INSTALL_TIMEOUT_S = 240  # guess: inside the probe's own 6 min cap; zoo runs time a real install (Oct 28-29)
_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99})"
_SPEC = r"(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]{1,64}"
_LINE = re.compile(
	rf"(?P<name>{_NAME})(?P<extras>\[[A-Za-z0-9,._ -]{{1,200}}\])?\s*(?P<spec>{_SPEC}(?:\s*,\s*{_SPEC}){{0,8}})?"
	r"\s*(?P<marker>;[^#]{0,200})?"
)


def _norm(name: str) -> str:
	return re.sub(r"[-_.]+", "-", name).lower()


def installed() -> dict[str, str]:
	out = {}
	for d in md.distributions():
		name = d.metadata["Name"] if d.metadata else None
		if name:
			out.setdefault(_norm(name), d.version)
	return out


def plan(text: str, have: dict[str, str]) -> tuple[list[str], list[str]]:
	"""(lines to install, notes on the lines skipped and why)."""
	keep, notes = [], []
	for raw in text.splitlines():
		line = raw.split(" #", 1)[0].strip()
		if not line or line.startswith("#"):
			continue
		m = _LINE.fullmatch(line)
		if m is None:
			notes.append(f"skipped {line[:80]!r}: only plain PyPI requirements are installed")
			continue
		name = _norm(m["name"])
		if name in have:
			notes.append(f"kept the image's {m['name']} {have[name]} (asked for {line[:80]!r})")
			continue
		keep.append(line)
	if len(keep) > MAX_REQUIREMENTS:
		notes.append(f"installed the first {MAX_REQUIREMENTS} of {len(keep)} requirements")
		keep = keep[:MAX_REQUIREMENTS]
	return keep, notes


def site_dirs(prefix: Path) -> list[Path]:
	"""Where pip --prefix put packages (lib/pythonX.Y/site-packages, or dist-packages on Debian)."""
	return sorted(
		{
			p
			for p in prefix.glob("**/*-packages")
			if p.is_dir() and p.name in ("site-packages", "dist-packages")
		}
	)


def install(workdir: Path, target: Path, *, log=print) -> tuple[str | None, list[Path]]:
	"""Install workdir/requirements.txt. Returns (pip's error tail or None, the dirs to import from).

	target: this probe's own prefix; a shared one under PREFLIGHT_DEPS_DIR is used instead when set."""
	req = workdir / "requirements.txt"
	if not req.is_file():
		return None, []
	have = installed()
	keep, notes = plan(req.read_text(encoding="utf-8", errors="replace"), have)
	for n in notes:
		log(f"[preflight] requirements: {n}")
	if not keep:
		return None, []
	pins = "".join(f"{n}=={v.split('+', 1)[0]}\n" for n, v in sorted(have.items()))
	shared = os.environ.get("PREFLIGHT_DEPS_DIR")
	if shared and os.access(shared, os.W_OK):
		key = hashlib.sha256(("\n".join(keep) + "\0" + pins).encode()).hexdigest()[:24]
		target = Path(shared) / key
		if (target / ".ok").is_file():
			log("[preflight] requirements: already installed for this job")
			return None, site_dirs(target)
		shutil.rmtree(target, ignore_errors=True)  # a half-finished install from a killed probe
	target.mkdir(parents=True, exist_ok=True)
	constraints = target / "constraints.txt"
	constraints.write_text(pins)
	log(f"[preflight] requirements: installing {', '.join(keep)}")
	# -I: pip walks every sys.path entry, and the probe's PYTHONPATH can name a directory its
	# unprivileged uid can't list (Modal's /root, Nov 1: PermissionError, nothing installed)
	cmd = [sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
		"--no-warn-script-location", "--only-binary", ":all:", "--prefix", str(target),
		"-c", str(constraints), *keep]  # fmt: skip
	cache = os.environ.get("PREFLIGHT_PIP_CACHE")
	if cache and os.access(cache, os.W_OK):  # shared across a run's probes
		cmd[5:5] = ["--cache-dir", cache]
	try:
		r = subprocess.run(cmd, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S, cwd=target)
	except subprocess.TimeoutExpired:
		return f"pip took longer than {INSTALL_TIMEOUT_S} s", []
	if r.returncode != 0:
		tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
		return "\n".join(tail)[-1500:] or f"pip exited with {r.returncode}", []
	(target / ".ok").write_text("ok")
	return None, site_dirs(target)
