"""Knob response on the live site (PM, Nov 3 item 6): three samples, 20 knob moves each, at laptop and
phone size, in headless Chromium on the GitHub runner. The app times each move from the knob change to
the frame after the readouts commit (`web/src/workspace/state.tsx`, `window.__preflight.samples()`);
this script reads those times. Prints p50, p95 and max per sample and size, and writes the times in
the shape B's validation scorer reads (`knob_ms`, as `web/e2e/stranger.mjs` KNOB_MS_OUT does) to
out/knob_ms_live.json. Only public sample pages are opened; nothing secret is read.

	python scripts/knob_p95_live.py https://preflight-cqn.pages.dev
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Source: a version from before this conversation's cutoff, pinned so the check doesn't drift.
PLAYWRIGHT = "playwright==1.47.0"
SAMPLES = ("lora-qwen", "pretrain", "grpo")
MOVES = 20
SIZES = {"laptop": (1366, 768, False), "phone": (390, 844, True)}
SETTLE_MS = 450  # longer than the engine's settle (ENGINE_SETTLE_MS 300) so each move is timed alone

# Moves the knobs as a visitor does: open a capsule, then step its slider (keys) or click its next
# option, round-robin over the capsules, 20 moves in all.
MOVE_JS = """async ([n, settle]) => {
	const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
	const caps = [...document.querySelectorAll('.pf-dock__row .pf-knob')];
	let done = 0, i = 0, tries = 0;
	while (done < n && tries < n * 4 && caps.length) {
		tries++;
		const cap = caps[i++ % caps.length];
		cap.click();
		await sleep(120);
		const range = document.querySelector('.pf-control input[type=range]');
		if (range) {
			const max = Number(range.max), v = Number(range.value);
			const next = v < max ? v + 1 : v - 1;
			const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
			set.call(range, String(next));
			range.dispatchEvent(new Event('input', { bubbles: true }));
			range.dispatchEvent(new Event('change', { bubbles: true }));
			done++;
		} else {
			const opts = [...document.querySelectorAll('.pf-control .pf-choice button')];
			const off = opts.find((b) => b.getAttribute('aria-pressed') !== 'true');
			if (!off) continue;
			off.click();
			done++;
		}
		await sleep(settle);
	}
	return done;
}"""


def pct(xs: list[float], q: float) -> float | None:
	if not xs:
		return None
	s = sorted(xs)
	return s[min(len(s) - 1, int(len(s) * q))]


def main(base: str) -> int:
	subprocess.run(["uv", "pip", "install", "--python", sys.executable, "-q", PLAYWRIGHT], check=True)
	subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=True,
		stdout=subprocess.DEVNULL)  # fmt: skip
	from playwright.sync_api import sync_playwright

	rows, every = [], []
	with sync_playwright() as pw:
		browser = pw.chromium.launch()
		for size, (w, h, mobile) in SIZES.items():
			for sample in SAMPLES:
				ctx = browser.new_context(
					viewport={"width": w, "height": h}, is_mobile=mobile, has_touch=mobile
				)
				ctx.add_init_script("localStorage.setItem('pf.onboarding.v1', '1')")
				page = ctx.new_page()
				page.goto(f"{base}/#/cockpit/{sample}", wait_until="networkidle")
				page.wait_for_selector(".pf-dock__row .pf-knob", timeout=60_000)
				page.wait_for_timeout(1_000)
				moved = page.evaluate(MOVE_JS, [MOVES, SETTLE_MS])
				ms = page.evaluate("() => window.__preflight?.samples() ?? []")
				every.extend(ms)
				rows.append({"size": size, "sample": sample, "moves": moved, "timed": len(ms),
					"p50_ms": round(pct(ms, 0.5) or 0, 1), "p95_ms": round(pct(ms, 0.95) or 0, 1),
					"max_ms": round(max(ms) if ms else 0, 1)})  # fmt: skip
				print(json.dumps(rows[-1]), flush=True)
				ctx.close()
		browser.close()
	summary = {"all_p95_ms": round(pct(every, 0.95) or 0, 1), "n": len(every)}
	print(json.dumps(summary))
	Path("out").mkdir(exist_ok=True)
	Path("out/knob_ms_live.json").write_text(
		json.dumps(
			{
				"knob_ms": [round(m, 1) for m in every],
				"base": base,
				"cpu_slowdown": 1,
				"rows": rows,
				"measured_at": datetime.now(UTC).isoformat(),
			},
			indent=1,
		)
		+ "\n"
	)
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "https://preflight-cqn.pages.dev"))
