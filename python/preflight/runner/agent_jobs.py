"""Agent and API jobs: sample 30 of the run's tasks through the recorder's proxy (D3).

	sampler = local_sampler(upstream="https://api.openai.com/v1", api_key=os.environ["OPENAI_API_KEY"])
	rec = run_pipeline(request, ctx, Deps(records=store, agent=sampler))

A sampler is called as sampler(fetched, script_map, k=, skip=, exclude=, run_id=) and returns
an AgentSample; the pipeline puts it on the record and B's engine projects the whole run from it.
The CLI's sampler runs the script here, with the user's own provider key held by the proxy (the
script gets a dummy key). The hosted path (a sandbox that can reach only the proxy) is not built.
"""

import sys
import tempfile
from pathlib import Path

from preflight.recorder.agent import AgentProxy
from preflight.recorder.agent_sampler import run_sample


def _write(files: dict[str, str], root: Path) -> None:
	for rel, text in files.items():
		p = (root / rel).resolve()
		if root.resolve() not in p.parents:
			continue  # sources.py already refuses these; never write outside the work dir
		p.parent.mkdir(parents=True, exist_ok=True)
		p.write_text(text, encoding="utf-8")


def local_sampler(upstream: str, api_key: str, *, rate=None, python: str = sys.executable, **proxy_kw):
	"""A sampler that runs the script on this machine, as `python <entrypoint> <args>`."""
	if rate is None:
		from preflight.engine.prices import token_rate as rate

	def sample(fetched, sm, *, k, skip=0, exclude=(), run_id="s1", timeout_s=None, env=None):
		population = getattr(getattr(sm, "agent", None), "n_tasks", None)
		proxy = AgentProxy(upstream, api_key, rate=rate, **proxy_kw)
		proxy.start()
		try:
			with tempfile.TemporaryDirectory(prefix="preflight-agent-") as work:
				root = Path(work)
				_write(fetched.files, root)
				cmd = [python, fetched.entrypoint, *list(fetched.args or [])]
				kw = {"timeout_s": timeout_s} if timeout_s else {}
				return run_sample(cmd, cwd=root, proxy=proxy, population=population, k=k, skip=skip,
					run_id=run_id, env=env, exclude=tuple(exclude), **kw)  # fmt: skip
		finally:
			proxy.stop()

	return sample
