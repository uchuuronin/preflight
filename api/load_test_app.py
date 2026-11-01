"""Load test of the live site (Chiya, Nov 1, round five step 4): 20 visitors at once, each in its own
headless Chromium on a Modal CPU container, doing what a stranger does on samples and the guide,
never a GPU probe:

1. open the home page and press "Open a sample flight" (time to the map);
2. move the first knob in the dock with real key presses, 12 moves 250 ms apart, and read the
   cockpit's own knob-to-readout timings (`window.__preflight.samples()`, the stranger test's meter);
3. start the guide tour and take its first step;
4. with `--ask`, ask the live guide one question (Muse Spark through the API; the only paid call).

Every response from the site's `/v1/` (the API through Pages) is counted; a status of 500 or more,
429, or a network failure is an API error. Each visitor is CPU only (2 cores, 2 GiB), so a run of
20 costs cents; the live guide adds one Muse Spark turn per visitor (guide-smoke measured about
US$0.004 a turn, run 41273350602).

	modal run --env main api/load_test_app.py::load --visitors 20 --ask
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

BASE = "https://preflight-cqn.pages.dev"
# Source: guess, a question a visitor might ask first; short so the turn stays cheap.
QUESTION = "What should I change first to stop the eval running out of memory?"
# Source: the stranger test's walk (web/e2e/stranger.mjs): 12 key presses, 250 ms apart.
KNOB_MOVES = 12


@app.function(image=image, cpu=2.0, memory=2048, timeout=6 * 60, retries=0)
def visitor(i: int, base: str, ask: bool) -> dict:
	from playwright.sync_api import sync_playwright

	out: dict = {"i": i, "api": [], "errors": []}
	with sync_playwright() as p:
		browser = p.chromium.launch()
		page = browser.new_page(viewport={"width": 1440, "height": 900})
		page.add_init_script("localStorage.setItem('pf.onboarded', '1')")

		def on_response(r):
			if "/v1/" in r.url:
				out["api"].append(r.status)

		page.on("response", on_response)
		page.on("requestfailed", lambda r: out["api"].append(0) if "/v1/" in r.url else None)
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

			start = page.get_by_role("button", name="Start the tour")
			if start.count():
				t1 = time.monotonic()
				start.first.click()
				page.wait_for_timeout(1500)
				out["tour_ms"] = round((time.monotonic() - t1) * 1000)
				out["tour_text"] = (page.locator(".pf-guide").first.inner_text(timeout=5_000) or "")[:160]

			if ask:
				box = page.locator("#pf-guide-ask")
				if box.count():
					box.fill(QUESTION)
					t2 = time.monotonic()
					page.get_by_role("button", name="Ask the guide").click()
					# Answered when "Thinking…" has come and gone.
					thinking = page.get_by_text("Thinking…").first
					with contextlib.suppress(Exception):  # a fast answer can skip the visible state
						thinking.wait_for(state="visible", timeout=5_000)
					thinking.wait_for(state="hidden", timeout=90_000)
					out["ask_ms"] = round((time.monotonic() - t2) * 1000)
				else:
					out["ask_ms"] = None
					out["errors"].append("no live guide box")
		except Exception as e:  # a visitor that fails is a result, not a crash of the test
			out["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
		finally:
			browser.close()
	return out


def p95(xs: list[float]) -> float | None:
	if not xs:
		return None
	s = sorted(xs)
	return round(s[min(len(s) - 1, int(0.95 * len(s)))], 1)


@app.local_entrypoint()
def load(visitors: int = 20, ask: bool = False, base: str = BASE) -> None:
	t0 = time.monotonic()
	runs = list(visitor.map(range(visitors), kwargs={"base": base, "ask": ask}))
	wall = time.monotonic() - t0
	knobs = [x for r in runs for x in (r.get("knob_ms") or [])]
	api = [s for r in runs for s in r["api"]]
	bad = [s for s in api if s == 0 or s == 429 or s >= 500]
	asks = [r["ask_ms"] for r in runs if r.get("ask_ms")]
	summary = {
		"visitors": visitors,
		"wall_s": round(wall, 1),
		"map_ms_p95": p95([r["map_ms"] for r in runs if "map_ms" in r]),
		"knob_samples": len(knobs),
		"knob_ms_p50": round(statistics.median(knobs), 1) if knobs else None,
		"knob_ms_p95": p95(knobs),
		"api_calls": len(api),
		"api_errors": len(bad),
		"api_statuses": {str(s): api.count(s) for s in sorted(set(api))},
		"ask_ms_p95": p95(asks),
		"asks_answered": len(asks),
		"visitors_with_errors": [{"i": r["i"], "errors": r["errors"][:3]} for r in runs if r["errors"]],
	}
	print(json.dumps(summary, indent=1))
