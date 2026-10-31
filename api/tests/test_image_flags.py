"""The API's Modal image ships the override file its own PREFLIGHT_ENV reads, so a flag turned on
for production (B's flags.production.json) is on in the deployed API, not only in a local run."""

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_the_image_ships_the_flag_file_for_its_own_environment():
	text = APP.read_text()
	env = re.search(r'"PREFLIGHT_ENV":\s*"([a-z]+)"', text)
	assert env, "the API image sets PREFLIGHT_ENV"
	shipped = re.findall(
		r'add_local_file\(ROOT / "(flags[\w.]*\.json)", "/root/repo/(flags[\w.]*\.json)"\)', text
	)
	assert ("flags.json", "flags.json") in shipped
	name = f"flags.{env.group(1)}.json"
	assert (name, name) in shipped
