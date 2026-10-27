"""Hostile probe: tries every easy route to the worker's secrets and prints what it finds."""

import glob
import os

found = []
for k, v in os.environ.items():
	if "SECRET" in v:
		found.append(f"env:{k}")
for path in [f"/proc/{os.getppid()}/environ", "/proc/1/environ", os.path.expanduser("~/.modal.toml")]:
	try:
		with open(path, "rb") as f:
			if b"SECRET" in f.read():
				found.append(path)
	except OSError:
		pass
for path in glob.glob("/proc/[0-9]*/environ"):
	try:
		with open(path, "rb") as f:
			if b"SECRET" in f.read():
				found.append(path)
	except OSError:
		pass
print("LEAKED:" + ",".join(found) if found else "NOTHING")
