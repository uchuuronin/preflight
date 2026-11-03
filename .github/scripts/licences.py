"""Fail on copyleft dependencies (gap 22): GPL, AGPL or SSPL in the installed Python environment.

Reads every installed distribution's licence metadata (License-Expression, License, and
"License ::" classifiers). LGPL is allowed (linking is fine under Apache-2.0); GPL, AGPL and SSPL
fail. A package whose metadata names several licences passes if any one of them is permissive
("MIT OR GPL-2.0" lets us choose MIT). A package that names no licence at all, in its metadata or a
LICENSE/COPYING file, fails too unless NO_LICENCE_OK lists it with the reason a person checked (D,
Nov 3: it passed silently).

	uv run python .github/scripts/licences.py            # root environment
	cd api && uv run python ../.github/scripts/licences.py
"""

from __future__ import annotations

import re
import sys
from importlib import metadata

BAD = re.compile(r"(?<![L])\b(A?GPL|GNU (Affero )?General Public|SSPL|Server Side Public)", re.I)
LGPL = re.compile(r"\b(LGPL|Lesser General Public|Library General Public)", re.I)
# Our own packages and the stdlib's vendored tools are not dependencies.
SKIP = {"preflight", "preflight-api", "pip", "setuptools", "wheel"}
# Packages that name no licence, each checked by hand: name -> why it's fine.
NO_LICENCE_OK: dict[str, str] = {}
# A licence file: LICENSE, LICENCE or COPYING, any case, bare or with a text suffix or a licence name
# (LICENSE.txt, LICENSE-MIT); not a module such as license.py, and not a NOTICE (attributions only).
LICENCE_FILE = re.compile(r"(^|/)(LICEN[CS]E|COPYING)([-.][\w.-]*)?$", re.I)


def has_licence_file(dist: metadata.Distribution) -> bool:
	if dist.metadata.get_all("License-File"):
		return True
	return any(LICENCE_FILE.search(str(f)) and not str(f).endswith(".py") for f in dist.files or [])


def licences(dist: metadata.Distribution) -> list[str]:
	md = dist.metadata
	out = [md.get("License-Expression") or "", md.get("License") or ""]
	out += [c.split("::")[-1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
	return [x.strip() for x in out if x and x.strip() and x.strip().upper() != "UNKNOWN"]


def copyleft(names: list[str]) -> bool:
	if not names:
		return False
	alternatives = [n for part in names for n in re.split(r"\s+OR\s+|/|;", part)]
	bad = [a for a in alternatives if BAD.search(a) and not LGPL.search(a)]
	ok = [a for a in alternatives if a not in bad]
	return bool(bad) and not ok


def main() -> int:
	found, unnamed = [], []
	for dist in metadata.distributions():
		name = (dist.metadata.get("Name") or "").lower()
		if name in SKIP:
			continue
		names = licences(dist)
		if not names and name not in NO_LICENCE_OK and not has_licence_file(dist):
			unnamed.append(f"{name} {dist.version}")
		if copyleft(names):
			found.append(f"{name} {dist.version}: {'; '.join(names)[:160]}")
	for f in sorted(set(found)):
		print(f"::error title=copyleft dependency::{f}")
	for u in sorted(set(unnamed)):
		print(f"::error title=no licence named::{u} (check it by hand, then list it in NO_LICENCE_OK)")
	print(f"{len(found)} copyleft dependencies, {len(set(unnamed))} with no licence named")
	return 1 if found or unnamed else 0


if __name__ == "__main__":
	sys.exit(main())
