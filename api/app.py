"""Deploy the API on Modal: `modal deploy api/app.py` (the deploy workflow runs it).

Secrets (never in the repo): "preflight-meta" holds META_API_KEY and META_API_BASE.
Not deployed yet: Modal access is pending (contract fallback: Fly.io or Render by Wed).
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install("fastapi>=0.115", "pydantic>=2.8,<3", "httpx>=0.27", "numpy>=1.26")
	.env({"PREFLIGHT_ROOT": "/root/repo"})
	.add_local_file(ROOT / "flags.json", "/root/repo/flags.json")
	.add_local_dir(ROOT / "python" / "preflight", "/root/repo/python/preflight")
	.add_local_dir(ROOT / "api" / "preflight_api", "/root/repo/api/preflight_api")
	.add_local_dir(ROOT / "api" / "addons", "/root/repo/api/addons")
)

app = modal.App("preflight-api", image=image)


@app.function(secrets=[modal.Secret.from_name("preflight-meta")], min_containers=0, max_containers=4)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def api():
	import sys

	sys.path[:0] = ["/root/repo/api", "/root/repo/python"]
	from preflight_api.app import create_app

	return create_app()
