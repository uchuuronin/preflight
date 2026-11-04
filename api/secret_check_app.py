"""Whether the API's Modal secrets set the names it reads, printed as set or unset and never the value
(PM's audit, Nov 3: without PREFLIGHT_GITHUB_TOKEN, GitHub allows 60 calls an hour shared, and repo
check-ins fail after about 15 an hour).

    modal run --env main api/secret_check_app.py::check
"""

from __future__ import annotations

import json

import modal

app = modal.App("preflight-secret-check")
NAMES = ("PREFLIGHT_GITHUB_TOKEN",)


@app.function(
	secrets=[modal.Secret.from_name("preflight-meta"), modal.Secret.from_name("preflight-api")], timeout=60
)
def which() -> dict[str, str]:
	import os

	return {n: ("set" if os.environ.get(n, "").strip() else "unset") for n in NAMES}


@app.local_entrypoint()
def check() -> None:
	print(json.dumps(which.remote()))
