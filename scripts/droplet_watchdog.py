"""Destroy the AMD droplet after it sits idle (PM, Nov 2, item 6). A droplet bills for uptime, powered off included.

Run it on the droplet itself, in the background, for the whole session:

	nohup python3 droplet_watchdog.py --idle-minutes 20 >> watchdog.log 2>&1 &

Every minute it reads GPU use with `amd-smi` (or `rocm-smi` where that is all there is). A minute when every GPU is at
or under --busy-percent, no GPU holds more than 5% of its memory and the load is under one core is idle; any busy
minute starts the count again (a job downloading weights or copying results off keeps its droplet). After --idle-minutes idle minutes in a row:

- with DO_API_TOKEN set (the cloud's API token, which only Chiya adds), it destroys this droplet through the
  DigitalOcean API (DELETE /v2/droplets/<id>; the id from the droplet's metadata service, or DROPLET_ID);
- without one it only warns, on stderr and to every logged-in terminal (`wall`), and keeps warning every
  --idle-minutes: it never powers the droplet off, because a powered-off droplet still bills.

--dry-run reads and counts as usual but never calls the API (it says what it would have done). --once takes one
reading and prints it. Standard library only, so it runs on the droplet's stock Python.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

API = "https://api.digitalocean.com/v2"
METADATA_ID = "http://169.254.169.254/metadata/v1/id"  # cited: DigitalOcean droplet metadata service
IDLE_MINUTES = 20  # cited: PM, Nov 2, item 6
BUSY_PERCENT = 5.0  # guess: a GPU under 5% busy is a driver poll or a monitor, not a job
VRAM_BUSY = (
	0.05  # guess: a model loaded on the GPU holds more than 5% of its memory, a bare driver holds less
)
LOAD_BUSY = (
	1.0  # guess: a 1-minute load above one core is something running (a download, a copy, preprocessing)
)


def _run(cmd: list[str]) -> str:
	return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30).stdout


def read_use(run=_run) -> list[float]:
	"""Each GPU's busy percent, from amd-smi's JSON, else rocm-smi's. Raises when neither gives a number."""
	if shutil.which("amd-smi") or run is not _run:
		try:
			got = json.loads(run(["amd-smi", "metric", "--usage", "--json"]))
			vals = []
			for g in got if isinstance(got, list) else got.get("gpu_data", [got]):
				u = (g.get("usage") or {}).get("gfx_activity")
				v = u.get("value") if isinstance(u, dict) else u
				if v is not None and v != "N/A":
					vals.append(float(v))
			if vals:
				return vals
		except (
			subprocess.CalledProcessError,
			FileNotFoundError,
			json.JSONDecodeError,
			ValueError,
			AttributeError,
		):
			pass
	got = json.loads(run(["rocm-smi", "--showuse", "--json"]))
	vals = [
		float(re.sub(r"[^\d.]", "", str(v)))
		for card in got.values()
		if isinstance(card, dict)
		for k, v in card.items()
		if "GPU use" in k and re.search(r"\d", str(v))
	]
	if not vals:
		raise RuntimeError("no GPU use reading from amd-smi or rocm-smi")
	return vals


def read_vram(run=_run) -> list[float]:
	"""Each GPU's VRAM in use, as a fraction, from rocm-smi; [] when it can't say (then only activity counts)."""
	try:
		got = json.loads(run(["rocm-smi", "--showmemuse", "--json"]))
	except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, OSError):
		return []
	return [
		float(re.sub(r"[^\d.]", "", str(v))) / 100
		for card in got.values()
		if isinstance(card, dict)
		for k, v in card.items()
		if "memory" in k.lower() and "%" in k and re.search(r"\d", str(v))
	]


def busy_reasons(use: list[float], vram: list[float], load: float, busy_percent: float) -> list[str]:
	"""Why this minute isn't idle (review, Nov 2: a job downloading weights or copying results off uses no GPU
	compute for a while, so VRAM in use and the CPU's load count as busy too)."""
	out = []
	if any(u > busy_percent for u in use):
		out.append(f"GPU busy {max(use):g}%")
	if any(v > VRAM_BUSY for v in vram):
		out.append(f"VRAM {max(vram):.0%} in use")
	if load > LOAD_BUSY:
		out.append(f"load {load:.1f}")
	return out


def droplet_id() -> str:
	if os.environ.get("DROPLET_ID"):
		return os.environ["DROPLET_ID"]
	with urllib.request.urlopen(METADATA_ID, timeout=5) as r:
		return r.read().decode().strip()


def destroy(token: str, did: str, opener=urllib.request.urlopen) -> int:
	req = urllib.request.Request(
		f"{API}/droplets/{did}", method="DELETE", headers={"Authorization": f"Bearer {token}"}
	)
	with opener(req, timeout=30) as r:
		return r.status  # 204 when the droplet is being destroyed


def warn(text: str, *, terminals: bool = True) -> None:
	print(f"[watchdog] {text}", file=sys.stderr, flush=True)
	if terminals and shutil.which("wall"):
		subprocess.run(["wall", f"[watchdog] {text}"], check=False, timeout=10)


class Watch:
	"""The idle count, one reading a minute."""

	def __init__(self, idle_minutes: int, busy_percent: float):
		self.idle_minutes, self.busy, self.idle = idle_minutes, busy_percent, 0

	def tick(self, use: list[float], vram: list[float] = (), load: float = 0.0) -> bool:
		"""True when the droplet has now been idle for idle_minutes in a row (and the count restarts)."""
		self.idle = 0 if busy_reasons(use, list(vram), load, self.busy) else self.idle + 1
		if self.idle >= self.idle_minutes:
			self.idle = 0
			return True
		return False


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--idle-minutes", type=int, default=IDLE_MINUTES)
	ap.add_argument("--busy-percent", type=float, default=BUSY_PERCENT)
	ap.add_argument("--interval-s", type=float, default=60.0)
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--once", action="store_true")
	args = ap.parse_args(argv)
	if args.once:
		print(json.dumps({"gpu_use_percent": read_use()}))
		return 0
	token = os.environ.get("DO_API_TOKEN", "")
	if not token:
		warn(
			"no DO_API_TOKEN: this only warns after idle time, it can't destroy the droplet (and powering off still bills)"
		)
	w = Watch(args.idle_minutes, args.busy_percent)
	while True:
		try:
			use = read_use()
		except Exception as exc:  # a failed reading isn't idleness: the count holds
			warn(
				f"couldn't read GPU use ({type(exc).__name__}: {exc}); the idle count holds at {w.idle}",
				terminals=False,
			)
			time.sleep(args.interval_s)
			continue
		vram, load = read_vram(), os.getloadavg()[0]
		if w.tick(use, vram, load):
			msg = f"idle for {args.idle_minutes} minutes (every GPU at or under {args.busy_percent:g}%)"
			if not token:
				warn(f"{msg}: destroy the droplet in the AMD Developer Cloud panel, or it keeps billing")
			elif args.dry_run:
				try:
					did = droplet_id()
				except Exception as exc:
					did = f"<id unreadable: {type(exc).__name__}>"
				warn(f"{msg}: dry run, would DELETE {API}/droplets/{did}")
			else:
				try:  # review, Nov 2: an API or metadata error is reported and the watch goes on, never a crash
					did = droplet_id()
					warn(f"{msg}: destroying droplet {did}")
					status = destroy(token, did)
				except Exception as exc:
					warn(
						f"couldn't destroy the droplet ({type(exc).__name__}: {exc}); destroy it in the panel"
					)
				else:
					warn(f"the API answered {status}")
					if status in (202, 204):
						return 0
		time.sleep(args.interval_s)


if __name__ == "__main__":
	sys.exit(main())
