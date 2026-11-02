"""scripts/droplet_watchdog.py, dry: readings, the idle count, and the destroy call, with nothing real touched."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("droplet_watchdog", ROOT / "scripts" / "droplet_watchdog.py")
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)


def test_reads_amd_smi_and_falls_back_to_rocm_smi():
	amd = json.dumps([{"gpu": 0, "usage": {"gfx_activity": {"value": 87, "unit": "%"}}}])
	assert wd.read_use(lambda cmd: amd if cmd[0] == "amd-smi" else "") == [87.0]

	def only_rocm(cmd):
		if cmd[0] == "amd-smi":
			raise FileNotFoundError
		return json.dumps({"card0": {"GPU use (%)": "3"}, "card1": {"GPU use (%)": "0"}, "system": {}})

	assert wd.read_use(only_rocm) == [3.0, 0.0]


def test_twenty_idle_minutes_in_a_row_and_a_busy_one_starts_again():
	w = wd.Watch(20, 5.0)
	assert not any(w.tick([0.0]) for _ in range(19))
	assert not w.tick([40.0])  # busy: back to zero
	assert not any(w.tick([2.0, 1.0]) for _ in range(19))
	assert w.tick([0.0, 0.0])  # the twentieth idle minute in a row


def test_destroy_sends_one_delete_with_the_token():
	seen = {}

	class Resp:
		status = 204

		def __enter__(self):
			return self

		def __exit__(self, *a):
			return False

	def opener(req, timeout):
		seen.update(method=req.get_method(), url=req.full_url, auth=req.get_header("Authorization"))
		return Resp()

	assert wd.destroy("tok", "123", opener=opener) == 204
	assert seen == {
		"method": "DELETE",
		"url": "https://api.digitalocean.com/v2/droplets/123",
		"auth": "Bearer tok",
	}
