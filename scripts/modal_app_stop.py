"""Stop a B Modal app that keeps retrying (PM, Nov 2: ask_model_app's containers failing to start).

	python scripts/modal_app_stop.py preflight-ask-model --before 2026-11-02T16:35:00Z [--dry-run]

Lists the apps in the main environment (`modal app list --json`), picks those with exactly this name that are still
live (deployed or ephemeral) and were created before --before, so a newer run of the same app is left alone, prints
them, and stops each by its app id (`modal app stop --yes`: no prompt in a CI run). Only B's own app names are accepted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime

B_APPS = (
	"preflight-ask-model",
	"preflight-claim-check-eval",
	"preflight-quality-check",
	"preflight-inflight-check",
)


def _when(s: str) -> datetime:
	d = datetime.fromisoformat(s.replace("Z", "+00:00"))
	return d if d.tzinfo else d.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("name", choices=B_APPS)
	ap.add_argument("--before", required=True, help="only apps created before this time (ISO 8601)")
	ap.add_argument("--env", default="main")
	ap.add_argument("--dry-run", action="store_true")
	args = ap.parse_args(argv)
	out = subprocess.run(
		["modal", "app", "list", "--env", args.env, "--json"], check=True, capture_output=True, text=True
	).stdout
	apps = json.loads(out)
	print(json.dumps(apps, indent=1)[:4000])
	cut = _when(args.before)
	hit = []
	for a in apps:
		name = a.get("Description") or a.get("description") or a.get("name")
		state = str(a.get("State") or a.get("state") or "").lower()
		created = a.get("Created at") or a.get("created_at") or ""
		app_id = a.get("App ID") or a.get("app_id")
		if name != args.name or not app_id or state not in ("deployed", "ephemeral", "ephemeral (detached)"):
			continue
		try:
			if _when(created) >= cut:
				continue
		except ValueError:
			if (
				state != "deployed"
			):  # an unreadable time: only a deployed app is stopped, never a run that may be new
				continue
		hit.append((app_id, state, created))
	print(f"to stop: {hit}")
	if args.dry_run:
		return 0
	for app_id, _, _ in hit:
		r = subprocess.run(
			["modal", "app", "stop", app_id, "--env", args.env, "--yes"], capture_output=True, text=True
		)
		print(app_id, r.returncode, (r.stdout + r.stderr).strip()[:300])
	return 0


if __name__ == "__main__":
	sys.exit(main())
