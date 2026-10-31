"""The API's Modal image ships the override file its own PREFLIGHT_ENV reads, at the root its own
PREFLIGHT_ROOT names, so a flag turned on for production (B's flags.production.json) is on in the
deployed API, not only in a local run."""

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"
REPO = APP.parents[1]


def test_the_image_ships_the_flag_file_for_its_own_environment():
	text = APP.read_text()
	env = re.search(r'"PREFLIGHT_ENV":\s*"([a-z]+)"', text)
	root = re.search(r'"PREFLIGHT_ROOT":\s*"([^"]+)"', text)
	assert env and root, "the API image sets PREFLIGHT_ENV and PREFLIGHT_ROOT"
	shipped = dict(re.findall(r'add_local_file\(ROOT / "(flags[\w.]*\.json)", "([^"]+)"\)', text))
	for name in ("flags.json", f"flags.{env.group(1)}.json"):
		assert shipped.get(name) == f"{root.group(1)}/{name}", f"{name} isn't shipped where flags.py reads it"
		# Modal uploads it when the image is built: a missing file fails the build, so fail here first.
		assert (REPO / name).is_file(), f"{name} is missing from the repo (B's file; merge it with this one)"
