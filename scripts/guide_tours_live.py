"""The live guide's tour, three times on the LoRA sample of the live site, from a GitHub runner (C, Nov 4:
D's C2 take handed over to the rule-based tour). Installs the web app's locked packages and Playwright's
Chromium, then runs web/e2e/guide_tours.mjs with BASE_URL set; exits with its code. Public pages only;
each tour spends a few Muse Spark turns (cents), within the guide's caps. Keys are taken out of the
child's environment, as in stranger_live.py.

	python scripts/guide_tours_live.py https://preflight-cqn.pages.dev
"""

from __future__ import annotations

import subprocess
import sys

from stranger_live import WEB, clean_env


def main(base: str) -> int:
	env = clean_env()
	subprocess.run(
		["npm", "ci", "--no-audit", "--no-fund"], cwd=WEB, env=env, check=True, stdout=subprocess.DEVNULL
	)
	subprocess.run(
		["npx", "playwright", "install", "--with-deps", "chromium"],
		cwd=WEB,
		env=env,
		check=True,
		stdout=subprocess.DEVNULL,
	)
	return subprocess.run(
		["node", "e2e/guide_tours.mjs"], cwd=WEB, env=clean_env(BASE_URL=base, TOURS="3")
	).returncode


if __name__ == "__main__":
	sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "https://preflight-cqn.pages.dev"))
