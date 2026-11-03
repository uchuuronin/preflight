"""The API knows every sample the web lists, and ships its file (D's walkthrough, Nov 2: five were
missing, so /predict answered 404 and their knobs moved nothing)."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))

from preflight_api.stores import SAMPLES  # noqa: E402


def _web_samples() -> dict[str, str]:
	load = ROOT / "web/src/workspace/record/load.ts"
	out = {}
	for m in re.finditer(r'id: "([a-z0-9-]+)"(.*?)load: \(\) => import\("([^"]+)"\)', load.read_text(), re.S):
		out[m.group(1)] = (load.parent / m.group(3)).resolve().relative_to(ROOT).as_posix()
	return out


def test_every_web_sample_is_known_to_the_api_with_the_same_file():
	"""The API may know more (retired samples, for old links); never fewer, and the same file."""
	web = _web_samples()
	assert len(web) >= 8
	for sid, path in web.items():
		assert SAMPLES.get(sid) == path, sid


def test_every_sample_file_is_in_the_api_image():
	text = (ROOT / "api/app.py").read_text()
	for path in SAMPLES.values():
		assert f'"{path}"' in text, path
		assert (ROOT / path).is_file(), path
