"""The stranger test (web/e2e/stranger.mjs) against the live site, from a GitHub runner (PM, Nov 3 item 7:
"the stranger test with the final flags"), and the mentions check (web/e2e/mentions.mjs) when TERMS is
given. Installs the web app's locked packages and Playwright's Chromium, then runs the test with
BASE_URL set; exits with the test's own code. Only public pages are opened. It presses "Start the tour"
and one "Apply", so with the live guide on it spends about two Muse Spark turns (cents), within the
guide's caps; it never presses Check in. The workflow's keys are taken out of every child's
environment (npm's install scripts included).

	python scripts/stranger_live.py https://preflight-cqn.pages.dev
	python scripts/stranger_live.py https://preflight-cqn.pages.dev Muse_Glimmer,Private_triage

The second argument is the mentions check's words, comma-separated, "_" for a space (the remote-run
workflow's command allows no quotes).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"


SECRETS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "META_API_KEY", "META_API_BASE", "BASETEN_API_KEY")


def clean_env(**extra: str) -> dict[str, str]:
	return {
		**{
			k: v
			for k, v in os.environ.items()
			if k not in SECRETS and not k.endswith(("_KEY", "_SECRET", "_TOKEN"))
		},
		**extra,
	}


def main(base: str, terms: str | None) -> int:
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
	env = clean_env(BASE_URL=base)
	code = subprocess.run(["node", "e2e/stranger.mjs"], cwd=WEB, env=env).returncode
	if terms:
		words = "|".join(w.replace("_", " ") for w in terms.split(",") if w)
		env = clean_env(BASE=base, TERMS=words)
		code = subprocess.run(["node", "e2e/mentions.mjs"], cwd=WEB, env=env).returncode or code
	return code


if __name__ == "__main__":
	sys.exit(
		main(
			sys.argv[1] if len(sys.argv) > 1 else "https://preflight-cqn.pages.dev",
			sys.argv[2] if len(sys.argv) > 2 else None,
		)
	)
