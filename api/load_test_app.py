"""Load test of the live site (Chiya, Nov 1, round five step 4): 20 visitors at once, each in its own
headless Chromium on a Modal CPU container, doing what a stranger does on samples and the guide,
never a GPU probe:

1. open the home page and press "Open a sample flight" (time to the map);
2. move the first knob in the dock with real key presses, 12 moves 250 ms apart, and read the
   cockpit's own knob-to-readout timings (`window.__preflight.samples()`, the stranger test's meter).
   This is measured in each visitor's browser, without the stranger test's 4x CPU slowdown, so
   server load can't move it; it shows the page stays quick with 20 people on it, not the 400 ms
   contract number;
3. the first `--guided` visitors (default 5), started 3 s apart, also start the guide tour, which in
   production is the live guide (Muse Spark through the API, `c.guide.live` on), and ask it one
   question. Each tour can make up to 8 model calls and the browser repeats a turn up to 7 times;
   the API's ledger measured about US$0.033 a tour and US$0.115 at most (api/preflight_api/
   guide/ledger.py), so 5 guided visitors cost about US$0.17, at most about US$0.6, out of the
   guide's daily cap the tester shares. The others don't touch the guide.

Every `/v1/` response is counted. A guide turn refused for the rate limit or the daily cap comes back
as 503 with a rule-based fallback: counted as a fallback, not a server error. Anything else at 500 or
more, 429, or a network failure while the page is open is an API error.

How it stops: `modal run` makes an ephemeral app that ends when every visitor has returned; each
visitor stops at its own 6-minute timeout at the latest, and nothing is deployed or left running.

	modal run --env main api/load_test_app.py::load --visitors 20 --guided 5
"""

from __future__ import annotations

import contextlib
import json
import statistics
import time

import modal

app = modal.App("preflight-load-test")

# Source: Playwright's own image recipe; the browser comes with its system libraries.
image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install("playwright==1.48.0")
	.run_commands("playwright install --with-deps chromium")
)

# Source: the live site (Pages project preflight).
BASE = "https://preflight-cqn.pages.dev"
# Source: guess, room for Chromium (about 1 GiB) on two cores; a visitor's walk takes 1 to 2 minutes.
CPU, MEMORY_MB, TIMEOUT_S = 2.0, 2048, 6 * 60
# Source: derived, the guide's per-IP limit refills 0.5 calls a second (guide/ledger.py); 3 s apart
# keeps 5 guided visitors from arriving as one burst if Modal containers share an address.
GUIDED_GAP_S = 3.0
# Source: guess, a question a visitor might ask first; short so the turn stays cheap.
QUESTION = "What should I change first to stop the eval running out of memory?"
# Source: the stranger test's walk (web/e2e/stranger.mjs): 12 key presses, 250 ms apart.
KNOB_MOVES = 12


@app.function(image=image, cpu=CPU, memory=MEMORY_MB, timeout=TIMEOUT_S, retries=0)
def visitor(i: int, base: str, guided: bool) -> dict:
	from playwright.sync_api import sync_playwright

	out: dict = {"i": i, "api": [], "errors": [], "guide": []}
	with sync_playwright() as p:
		browser = p.chromium.launch()
		page = browser.new_page(viewport={"width": 1440, "height": 900})
		page.add_init_script("localStorage.setItem('pf.onboarding.v1', '1')")

		def on_response(r):
			if "/v1/" not in r.url:
				return
			if "/v1/guide/turns" in r.url:
				# A refusal for the rate limit or the cap is a 503 with a rule-based fallback.
				fallback = False
				if r.status == 503:
					with contextlib.suppress(Exception):
						fallback = (r.json().get("error") or {}).get("fallback") == "rule_based"
				out["guide"].append("fallback" if fallback else r.status)
				if fallback:
					return
			out["api"].append(r.status)

		def on_failed(r):
			if "/v1/" in r.url:
				out["api"].append(0)

		page.on("response", on_response)
		page.on("requestfailed", on_failed)
		page.on("pageerror", lambda e: out["errors"].append(str(e)[:200]))
		try:
			page.goto(base + "/", wait_until="load", timeout=60_000)
			t0 = time.monotonic()
			page.get_by_role("button", name="Open a sample flight").or_(
				page.get_by_role("link", name="Open a sample flight")
			).first.click()
			page.locator("main[data-pos='active'] .pf-onode").first.wait_for(timeout=30_000)
			out["map_ms"] = round((time.monotonic() - t0) * 1000)

			page.locator("main[data-pos='active'] .pf-dock__row .pf-knob").first.click()
			rng = page.locator(".pf-control input[type=range]").first
			if rng.count():
				rng.focus()
				for k in range(KNOB_MOVES):
					page.keyboard.press("ArrowLeft" if k % 3 == 2 else "ArrowRight")
					page.wait_for_timeout(250)
			else:
				opts = page.locator(".pf-control .pf-choice button")
				for k in range(min(opts.count(), 6)):
					opts.nth(k).click()
					page.wait_for_timeout(250)
			page.keyboard.press("Escape")
			out["knob_ms"] = page.evaluate("window.__preflight?.samples?.() ?? []")

			if guided:
				time.sleep(i * GUIDED_GAP_S)
				t1 = time.monotonic()
				page.get_by_role("button", name="Start the tour").first.click()
				# The live guide's ask box appears once its first turn has answered.
				box = page.locator("#pf-guide-ask")
				box.wait_for(timeout=60_000)
				out["tour_ms"] = round((time.monotonic() - t1) * 1000)
				before = len(out["guide"])
				box.fill(QUESTION)
				t2 = time.monotonic()
				page.get_by_role("button", name="Ask the guide").click()
				# Time to the first guide turn back for it (200), or a refusal with a fallback.
				for _ in range(180):
					if len(out["guide"]) > before:
						break
					page.wait_for_timeout(500)
				done = out["guide"][before:]
				out["ask_ms"] = round((time.monotonic() - t2) * 1000) if 200 in done else None
		except Exception as e:  # a visitor that fails is a result, not a crash of the test
			out["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
		finally:
			# Requests cut off by closing the browser aren't the site's errors.
			page.remove_listener("response", on_response)
			page.remove_listener("requestfailed", on_failed)
			browser.close()
	return out


def p95(xs: list[float]) -> float | None:
	if not xs:
		return None
	s = sorted(xs)
	return round(s[min(len(s) - 1, int(0.95 * len(s)))], 1)


@app.local_entrypoint()
def load(visitors: int = 20, guided: int = 5, base: str = BASE) -> None:
	t0 = time.monotonic()
	runs = list(visitor.starmap([(i, base, i < guided) for i in range(visitors)]))
	wall = time.monotonic() - t0
	knobs = [x for r in runs for x in (r.get("knob_ms") or [])]
	api = [s for r in runs for s in r["api"]]
	bad = [s for s in api if s == 0 or s == 429 or s >= 500]
	asks = [r["ask_ms"] for r in runs if r.get("ask_ms")]
	turns = [g for r in runs for g in r["guide"]]
	summary = {
		"visitors": visitors,
		"wall_s": round(wall, 1),
		"map_ms_p95": p95([r["map_ms"] for r in runs if "map_ms" in r]),
		"knob_note": "in each visitor's own browser, no CPU slowdown: page speed under load, not the 400 ms contract test",
		"knob_samples": len(knobs),
		"knob_ms_p50": round(statistics.median(knobs), 1) if knobs else None,
		"knob_ms_p95": p95(knobs),
		"api_calls": len(api),
		"api_errors": len(bad),
		"api_statuses": {str(s): api.count(s) for s in sorted(set(api))},
		"guided": guided,
		"tour_ms_p95": p95([r["tour_ms"] for r in runs if "tour_ms" in r]),
		"guide_turns": len(turns),
		"guide_fallbacks": turns.count("fallback"),
		"ask_ms_p95": p95(asks),
		"asks_answered": len(asks),
		"visitors_with_errors": [{"i": r["i"], "errors": r["errors"][:3]} for r in runs if r["errors"]],
	}
	print(json.dumps(summary, indent=1))
