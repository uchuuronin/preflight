"""The README's "From this repo" path, end to end, with a real Meta key (B, Nov 3; the PM's 9:07 PM ask).

	python scripts/readme_live_guide_check.py     # in a remote run that has META_API_KEY and META_API_BASE

Follows the README as written: the API in api/ with `PREFLIGHT_FLAGS='{"c.guide.live": true}' uv run uvicorn
preflight_api.app:create_app --factory --port 8000`, the web dev server in web/ with `npm ci && npm run dev`; then a
real browser opens the LoRA sample, starts the guide and takes one turn. It records which guide answered (the panel's
label: "Muse Spark" or "Rule-based tour"), the guide's first words, and every /v1/guide/turns call and its status. It
runs twice: at the README's own URL (http://localhost:5173/#/cockpit/lora-qwen), and with `?flags=c.guide.live` added,
the web's own switch for the live guide outside production builds (web/src/lib/flags.ts), so a failure says where.
Writes out/readme_live_guide.json and a screenshot of each; never prints a key.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
NODE = r"""
import { chromium } from "playwright";
import { writeFileSync } from "node:fs";
const [url, shot] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const turns = [];
page.on("response", async (r) => {
	if (r.url().includes("/v1/guide/")) {
		let body = "";
		try { body = (await r.text()).slice(0, 400); } catch {}
		turns.push({ url: new URL(r.url()).pathname, status: r.status(), body });
	}
});
const out = { url };
try {
	await page.goto(url, { waitUntil: "load" });
	await page.locator(".pf-onode").first().waitFor({ timeout: 20000 });
	out.label_before = (await page.locator(".pf-guide__head .pf-model").first().textContent())?.trim();
	const start = page.getByRole("button", { name: /Start the tour|Start|Ask Muse Spark/ }).first();
	out.start_button = (await start.textContent())?.trim();
	await start.click();
	// a live turn takes about 30 s (en.json guide.live.reading); wait for the guide to speak or the tour to take over
	const t0 = Date.now();
	let said = "";
	while (Date.now() - t0 < 120000) {
		const label = (await page.locator(".pf-guide__head .pf-model").first().textContent())?.trim();
		const says = await page.locator(".pf-guide__say").allTextContents();
		said = says.map((s) => s.trim()).filter(Boolean).join(" | ");
		const status = (await page.locator(".pf-guide__head .pf-guide__step").first().textContent())?.trim();
		out.label = label; out.status = status;
		const apply = await page.getByRole("button", { name: /^Apply / }).count();
		if (label === "Rule-based tour" && said) break;
		if (label === "Muse Spark" && (apply > 0 || turns.some((t) => t.url.endsWith("/turns") && t.status === 200) && status !== "Thinking")) break;
		await page.waitForTimeout(2000);
	}
	out.waited_s = Math.round((Date.now() - t0) / 1000);
	out.said = said.slice(0, 600);
	out.apply_offered = await page.getByRole("button", { name: /^Apply / }).count();
} catch (e) {
	out.error = String(e).slice(0, 400);
}
out.turns = turns;
await page.screenshot({ path: shot, fullPage: false });
await browser.close();
console.log(JSON.stringify(out));
"""


def wait_http(url: str, seconds: float) -> bool:
	t0 = time.monotonic()
	while time.monotonic() - t0 < seconds:
		try:
			with urllib.request.urlopen(url, timeout=5) as r:  # localhost only
				if r.status < 500:
					return True
		except Exception:
			time.sleep(2)
	return False


def main() -> int:
	OUT.mkdir(exist_ok=True)
	res: dict = {
		"key_set": bool(os.environ.get("META_API_KEY")),
		"base_set": bool(os.environ.get("META_API_BASE")),
	}
	env = {**os.environ, "PREFLIGHT_FLAGS": '{"c.guide.live": true}'}
	env.pop("VIRTUAL_ENV", None)  # the API is its own uv project (api/pyproject.toml), as the README runs it
	api_log = open(OUT / "api.log", "w")  # noqa: SIM115
	uv = shutil.which("uv") or "uv"
	api = subprocess.Popen([uv, "run", "uvicorn", "preflight_api.app:create_app", "--factory", "--port", "8000"],
		cwd=ROOT / "api", env=env, stdout=api_log, stderr=subprocess.STDOUT)  # fmt: skip
	web = None
	try:
		res["api_up"] = wait_http("http://127.0.0.1:8000/livez", 300)
		npm = shutil.which("npm") or "npm"
		ci = subprocess.run([npm, "ci"], cwd=ROOT / "web", capture_output=True, text=True)
		res["npm_ci"] = ci.returncode
		web_log = open(OUT / "web.log", "w")  # noqa: SIM115
		web = subprocess.Popen(
			[npm, "run", "dev"], cwd=ROOT / "web", stdout=web_log, stderr=subprocess.STDOUT
		)
		res["web_up"] = wait_http("http://localhost:5173/", 120)
		pw = subprocess.run(["npx", "playwright", "install", "--with-deps", "chromium"], cwd=ROOT / "web",
			capture_output=True, text=True)  # fmt: skip
		res["playwright_install"] = pw.returncode
		script = ROOT / "web" / ".readme_live_guide.mjs"
		script.write_text(NODE)
		res["runs"] = []
		for name, url in (("readme_as_written", "http://localhost:5173/#/cockpit/lora-qwen"),
			("with_url_flag", "http://localhost:5173/?flags=c.guide.live#/cockpit/lora-qwen")):  # fmt: skip
			r = subprocess.run(["node", str(script), url, str(OUT / f"{name}.png")], cwd=ROOT / "web",
				capture_output=True, text=True, timeout=300)  # fmt: skip
			line = (r.stdout.strip().splitlines() or ["{}"])[-1]
			try:
				got = json.loads(line)
			except ValueError:
				got = {"error": (r.stderr or r.stdout)[-600:]}
			got["name"] = name
			got["muse_spark_answered"] = got.get("label") == "Muse Spark" and any(
				t.get("url", "").endswith("/turns") and t.get("status") == 200 for t in got.get("turns", [])
			)
			res["runs"].append(got)
	finally:
		for p in (web, api):
			if p is not None:
				p.terminate()
				try:
					p.wait(10)
				except subprocess.TimeoutExpired:
					p.kill()
	key = os.environ.get("META_API_KEY") or ""
	text = json.dumps(res, indent=1)
	if key and key in text:  # never write the key, whatever an error echoed
		text = text.replace(key, "<META_API_KEY>")
	for log in ("api.log", "web.log"):
		p = OUT / log
		if p.exists() and key:
			p.write_text(p.read_text(errors="replace").replace(key, "<META_API_KEY>"))
	(OUT / "readme_live_guide.json").write_text(text + "\n")
	print(text)
	return 0


if __name__ == "__main__":
	sys.exit(main())
