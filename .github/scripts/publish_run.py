"""Commit a remote run's outputs to the `remote-runs` branch so agents can read them with git.

Agents' sandboxes can't read Actions logs or artifacts, but they can fetch branches. Every value of
every secret the job had is replaced with *** before anything is written, in case a command printed one.
Usage: publish_run.py <run-dir-name> <log-file> <out-dir> <exit-code>
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SECRET_NAMES = [
	"MODAL_TOKEN_ID",
	"MODAL_TOKEN_SECRET",
	"META_API_KEY",
	"BASETEN_API_KEY",
	"AMD_SSH_KEY",
	"DO_API_TOKEN",
	"GITHUB_TOKEN",
]
BRANCH = "remote-runs"


def scrub(text: str) -> str:
	for n in SECRET_NAMES:
		v = os.environ.get(n, "")
		for part in [v, *v.splitlines()]:
			part = part.strip()
			if len(part) >= 8:
				text = text.replace(part, "***")
	return text


def git(*args, cwd, check=True):
	return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def main() -> None:
	name, log, outdir, code = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
	repo = os.environ["GITHUB_REPOSITORY"]
	work = Path("_results")
	if work.exists():
		shutil.rmtree(work)
	url = (
		os.environ.get("RESULTS_REMOTE")
		or f"https://x-access-token:{os.environ['GITHUB_TOKEN']}@github.com/{repo}.git"
	)
	have = git("ls-remote", "--exit-code", "--heads", url, BRANCH, cwd=".", check=False).returncode == 0
	if have:
		git("clone", "--depth", "1", "--branch", BRANCH, url, str(work), cwd=".")
	else:
		work.mkdir()
		git("init", "-b", BRANCH, cwd=work)
		git("remote", "add", "origin", url, cwd=work)
		(work / "README.md").write_text(
			"# Remote runs\n\nOutputs of `remote-run.yml` and `amd-run.yml`, one folder per run, newest by date.\n"
			"Written by GitHub Actions only. Secrets are scrubbed before anything is committed.\n"
		)
	git("config", "user.name", "preflight-runs", cwd=work)
	git("config", "user.email", "runs@preflight.invalid", cwd=work)
	dest = work / "runs" / name
	dest.mkdir(parents=True, exist_ok=True)
	if log.exists():
		lines = scrub(log.read_text(errors="replace")).splitlines()
		(dest / "log.txt").write_text("\n".join(lines[-3000:]) + "\n")
	if outdir.is_dir():
		for f in outdir.rglob("*"):
			if f.is_file() and f.stat().st_size <= 20 * 1024 * 1024:
				t = dest / "out" / f.relative_to(outdir)
				t.parent.mkdir(parents=True, exist_ok=True)
				try:
					t.write_text(scrub(f.read_text()))
				except UnicodeDecodeError:
					shutil.copy2(f, t)
	meta = {
		"name": name,
		"exit_code": code,
		"workflow": os.environ.get("GITHUB_WORKFLOW"),
		"run_id": os.environ.get("GITHUB_RUN_ID"),
		"ref": os.environ.get("RUN_REF"),
		"sha": os.environ.get("RUN_SHA"),
		"cmd": os.environ.get("RUN_CMD"),
		"actor": os.environ.get("GITHUB_ACTOR"),
		"finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
	}
	(dest / "result.json").write_text(json.dumps(meta, indent="\t") + "\n")
	git("add", "-A", cwd=work)
	git("commit", "-m", f"run {name} (exit {code})", cwd=work)
	for attempt in range(6):  # other runs may publish at the same time; folders never collide
		if git("push", "origin", f"HEAD:{BRANCH}", cwd=work, check=False).returncode == 0:
			print(f"published runs/{name} to {BRANCH}")
			return
		git("fetch", "--depth", "1", "origin", BRANCH, cwd=work, check=False)
		git("rebase", f"origin/{BRANCH}", cwd=work, check=False)
		time.sleep(2 + 3 * attempt)
	sys.exit("could not publish results after 6 attempts")


if __name__ == "__main__":
	main()
