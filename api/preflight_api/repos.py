"""Turn whatever repo link a visitor pastes into what A's runner can fetch: the bare
`https://github.com/<owner>/<repo>` and a full 40-character commit hash.

GitHub only serves `git fetch origin <sha>` for the whole hash, and A's runner accepts nothing
else (python/preflight/runner/sources.py), so a branch, a tag, a short hash or no ref at all is
resolved here, once, at check-in. The job then runs exactly the code the visitor saw, even if the
branch moves while it waits in the queue.

Accepted links: github.com/<o>/<r>, .../<r>.git, .../tree/<ref>[/path], .../commit/<sha>,
.../blob/<ref>/<path>; a ref in `source.commit` wins over one in the link.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx

_LINK = re.compile(
	r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9_.-]{1,100})/(?P<repo>[A-Za-z0-9_.-]{1,100}?)"
	r"(?:\.git)?(?:/(?:tree|commit|blob)/(?P<ref>[^?#]+?))?/?(?:[?#].*)?$"
)
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
REF = re.compile(r"^[A-Za-z0-9._/-]{1,250}$")

# Source: guess, a cold GitHub API call is well under a second; 8 s covers a slow one without
# leaving the visitor at the check-in button.
TIMEOUT_S = 8.0
# Source: guess, a visitor re-submitting the same link within a minute gets the same commit
# without another call against GitHub's 60-an-hour unauthenticated limit.
CACHE_S = 60.0


class RepoError(ValueError):
	"""The link can't be used: the message is safe to show the visitor."""

	def __init__(self, message: str, *, retryable: bool = False) -> None:
		super().__init__(message)
		self.retryable = retryable


@dataclass(frozen=True)
class Resolved:
	uri: str
	commit: str
	ref: str  # what the visitor named ("HEAD" when nothing)


_cache: dict[tuple[str, str], tuple[float, str]] = {}


# A link's "/tree/<ref>/<path>" can't say where the ref ends (branch names hold slashes), so the
# longest prefixes are tried first. Source: guess, branch names rarely nest past four levels.
MAX_REF_TRIES = 4


def parse(uri: str, commit: str | None) -> tuple[str, str, list[str]]:
	"""(owner, repo, refs to try in order) from a GitHub link and an optional ref."""
	m = _LINK.match(uri.strip())
	if not m:
		raise RepoError("Repositories are read from public GitHub links: https://github.com/<owner>/<repo>.")
	owner, repo = m["owner"], m["repo"]
	given = (commit or "").strip()
	if given:
		refs = [given]
	elif m["ref"]:
		parts = m["ref"].strip("/").split("/")
		refs = ["/".join(parts[:i]) for i in range(len(parts), 0, -1)][:MAX_REF_TRIES]
	else:
		refs = ["HEAD"]
	if any(not REF.match(r) or ".." in r for r in refs):
		raise RepoError("That branch, tag or commit name isn't valid.")
	return owner, repo, refs


async def resolve(uri: str, commit: str | None, client: httpx.AsyncClient) -> Resolved:
	owner, repo, refs = parse(uri, commit)
	bare = f"https://github.com/{owner}/{repo}"
	for ref in refs:
		sha = await _one(owner, repo, bare, ref, client, last=ref == refs[-1])
		if sha:
			return Resolved(bare, sha, ref)
	raise RepoError(f"No public repository or no such branch at {bare}.")  # unreachable: last raises


async def _one(
	owner: str, repo: str, bare: str, ref: str, client: httpx.AsyncClient, *, last: bool
) -> str | None:
	if FULL_SHA.match(ref):
		return ref
	key = (bare.lower(), ref)
	hit = _cache.get(key)
	if hit and time.monotonic() - hit[0] < CACHE_S:
		return hit[1]
	headers = {"Accept": "application/vnd.github.sha", "X-GitHub-Api-Version": "2022-11-28"}
	token = os.environ.get("PREFLIGHT_GITHUB_TOKEN")
	if token:
		headers["Authorization"] = f"Bearer {token}"
	url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/commits/{quote(ref, safe='')}"
	try:
		res = await client.get(url, headers=headers, timeout=TIMEOUT_S)
	except httpx.HTTPError as e:
		raise RepoError("GitHub didn't answer; try again shortly.", retryable=True) from e
	if res.status_code in (404, 422):
		if not last:
			return None
		raise RepoError(
			f"No public repository or no {ref if ref != 'HEAD' else 'default branch'} at {bare}. "
			"Is it public, and is the branch right?"
		)
	if res.status_code in (403, 429):
		raise RepoError("GitHub is limiting requests right now; try again in a few minutes.", retryable=True)
	sha = res.text.strip()
	if res.status_code != 200 or not FULL_SHA.match(sha):
		raise RepoError("GitHub didn't return a commit for that link; try again shortly.", retryable=True)
	_cache[key] = (time.monotonic(), sha)
	if len(_cache) > 2048:
		_cache.clear()
	return sha
