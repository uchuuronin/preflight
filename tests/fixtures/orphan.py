"""Starts a detached process that would outlive the probe and write a marker later."""

import subprocess
import sys

marker = sys.argv[1]
subprocess.Popen(
	[sys.executable, "-c", f"import time; time.sleep(3); open({marker!r}, 'w').write('alive')"],
	start_new_session=True,
)
