"""Check a deployed share page over HTTP (C, Nov 2: share pages go live with the app). Reads only the
public share page, its card and badge for a record whose token is already public (a hero record in
workers/hero-cards.json); prints what it found and exits 1 if a check fails.

	python scripts/share_page_check.py https://preview.preflight-cqn.pages.dev ICDYrfHQ2nBNbv3mlRHHMg
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

UA = {"User-Agent": "preflight-share-check"}
HERO = Path(__file__).resolve().parents[1] / "workers" / "hero-cards.json"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
	def redirect_request(self, *a, **k):  # the share page's own answer, not where it sends people
		return None


def get(url: str, follow: bool = True) -> tuple[int, dict, bytes]:
	opener = urllib.request.build_opener() if follow else urllib.request.build_opener(_NoRedirect)
	try:
		with opener.open(urllib.request.Request(url, headers=UA), timeout=30) as r:
			return r.status, dict(r.headers), r.read()
	except urllib.error.HTTPError as e:
		return e.code, dict(e.headers or {}), e.read()
	except (urllib.error.URLError, TimeoutError, OSError) as e:
		return 0, {}, type(e).__name__.encode()


def main(base: str, token: str) -> int:
	heroes = json.loads(HERO.read_text())
	known = {h.get("link_id") for h in heroes.get("cards", [])}
	if token not in known:
		raise SystemExit("only a hero record's token (workers/hero-cards.json), which is already public")
	base = base.rstrip("/")
	results = []
	status, _, body = get(f"{base}/s/{token}", follow=False)
	html = body.decode("utf-8", "replace")
	results.append(("share page answers", status == 200, f"HTTP {status}"))
	og = dict(re.findall(r'<meta property="og:(\w+)" content="([^"]*)"', html))
	results.append(("preview title", bool(og.get("title")), og.get("title", "")[:100]))
	results.append(("preview description", bool(og.get("description")), og.get("description", "")[:160]))
	results.append(
		("names its card", f"/cards/{token}.png" in og.get("image", ""), og.get("image", "")[:100])
	)
	results.append(("sends people to the cockpit", f"#/cockpit/{token}" in html, ""))
	results.append(("no 'Not trusted' wording", "Not trusted" not in html, ""))
	status, headers, png = get(f"{base}/cards/{token}.png")
	ctype = headers.get("Content-Type") or headers.get("content-type") or ""
	results.append(
		(
			"card image",
			status == 200 and ctype.startswith("image/png") and png[:4] == b"\x89PNG",
			f"HTTP {status} {ctype}",
		)
	)
	status, headers, _ = get(f"{base}/badge/{token}")
	ctype = headers.get("Content-Type") or headers.get("content-type") or ""
	results.append(("badge", status == 200 and "svg" in ctype, f"HTTP {status} {ctype}"))
	status, _, _ = get(f"{base}/v1/status")
	results.append(("app's API still answers beside them", status == 200, f"HTTP {status}"))
	for name, ok, detail in results:
		print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
	return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
	sys.exit(main(sys.argv[1], sys.argv[2]))
