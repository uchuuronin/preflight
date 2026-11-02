"""Agent and API jobs: sample 30 of the run's tasks through the recorder's proxy (D3).

	sampler = local_sampler(upstream="https://api.openai.com/v1", api_key=os.environ["OPENAI_API_KEY"])
	rec = run_pipeline(request, ctx, Deps(records=store, agent=sampler))

A sampler is called as sampler(fetched, script_map, k=, skip_keys=, offered_first=, run_id=,
should_stop=) and returns
an AgentSample; the pipeline puts it on the record and B's engine projects the whole run from it.
The CLI's sampler runs the script here, with the user's own provider key held by the proxy (the
script gets a dummy key). The hosted sampler runs the proxy in the worker with our key, behind a
Modal tunnel, and the script in a sandbox with no secrets whose egress adds only the tunnel.
"""

import contextlib
import json
import sys
import tempfile
import time
from pathlib import Path

from preflight.recorder.agent import AgentProxy
from preflight.recorder.agent_sampler import TIMEOUT_S, run_sample


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

	def sample(fetched, sm, *, k, skip_keys=(), offered_first=(), run_id="s1", timeout_s=None, env=None,
		should_stop=None):  # fmt: skip
		population = getattr(getattr(sm, "agent", None), "n_tasks", None)
		proxy = AgentProxy(upstream, api_key, rate=rate, **proxy_kw)
		proxy.start()
		try:
			with tempfile.TemporaryDirectory(prefix="preflight-agent-") as work:
				root = Path(work)
				_write(fetched.files, root)
				cmd = [python, fetched.entrypoint, *list(fetched.args or [])]
				kw = {"timeout_s": timeout_s} if timeout_s else {}
				return run_sample(cmd, cwd=root, proxy=proxy, population=population, k=k, skip_keys=skip_keys,
					offered_first=tuple(offered_first), run_id=run_id, env=env, should_stop=should_stop, **kw)  # fmt: skip
		finally:
			proxy.stop()

	return sample


AGENT_ENTRY = ("python", "-m", "preflight.probes.agent_remote")


class _SandboxProc:
	"""The script running in a sandbox (probes/agent_remote.py), as run_sample's process."""

	def __init__(self, sb, job: dict):
		self.sb = sb
		self.p = sb.exec(*AGENT_ENTRY, timeout=int(job["timeout_s"]) + 600)  # installs come first
		self.p.stdin.write(json.dumps(job))
		self.p.stdin.write_eof()
		self.p.stdin.drain()
		self.report: dict = {}

	def poll(self):
		return self.p.poll()

	def wait(self, timeout):
		time.sleep(timeout)

	def kill(self):
		with contextlib.suppress(Exception):
			if self.p.poll() is not None:
				lines = [x for x in self.p.stdout.read().splitlines() if x.startswith("{")]
				self.report = json.loads(lines[-1]) if lines else {}
		with contextlib.suppress(Exception):
			self.sb.terminate()


def sandbox_sampler(create_sandbox, upstream: str, api_key: str, *, rate=None, forward=None, install_repo=None,
	keep_error=None, **proxy_kw):  # fmt: skip
	"""A sampler for the hosted path: the proxy runs here, with the key, behind a tunnel; the script
	runs in a sandbox created with create_sandbox(extra_egress=[tunnel host]) that holds no secrets.

	forward: modal.forward (a context manager giving the tunnel for a local port); injected for tests.
	install_repo, keep_error: probes.repo_install's and probes.agent_error_line's rules for this sampler
	(None: the flag)."""
	if rate is None:
		from preflight.engine.prices import token_rate as rate
	if forward is None:
		import modal

		forward = modal.forward

	def sample(
		fetched,
		sm,
		*,
		k,
		skip_keys=(),
		offered_first=(),
		run_id="s1",
		timeout_s=None,
		should_stop=None,
		inspect=None,
	):
		from urllib.parse import urlparse

		agent = getattr(sm, "agent", None)
		proxy = AgentProxy(upstream, api_key, rate=rate, **proxy_kw)
		_, port = proxy.start(bind="0.0.0.0")
		sb = None
		try:
			with forward(port) as tunnel:
				proxy.public_base = tunnel.url
				limit = timeout_s or TIMEOUT_S
				# the sandbox outlives the sample's own limit (installs come first inside it)
				sb = create_sandbox(extra_egress=[urlparse(tunnel.url).hostname], timeout_s=int(limit) + 900)

				from .flags import flag

				keep = (
					flag("probes.agent_error_line") if keep_error is None else keep_error
				)  # off until checked
				repo = flag("probes.repo_install") if install_repo is None else install_repo

				from .sources import extra_lines

				extra = list(extra_lines(getattr(fetched, "source", None)))  # JobRequest.extra_packages

				def launch(env):
					return _SandboxProc(sb, {"files": fetched.files, "entrypoint": fetched.entrypoint,
						"args": list(fetched.args or []), "env": env, "timeout_s": limit,
						**({"keep_error": True} if keep else {}), **({"install_repo": True} if repo else {}),
						**({"extra_packages": extra} if extra else {})})  # fmt: skip

				return run_sample(None, cwd=None, proxy=proxy, population=getattr(agent, "n_tasks", None), k=k,
					skip_keys=skip_keys, offered_first=tuple(offered_first), run_id=run_id, timeout_s=limit,
					env={}, should_stop=should_stop, launch=launch,
					base_url_var=getattr(agent, "base_url_var", None), inspect=inspect)  # fmt: skip
		finally:
			if sb is not None:
				with contextlib.suppress(Exception):
					sb.terminate()
			proxy.stop()

	return sample
