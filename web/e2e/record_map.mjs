// Opens one flight record in the cockpit, as a visitor following its share link would, and checks
// that the map and readouts draw. The end-to-end check (api/scripts/e2e_upload.py) writes the
// record the API served; this is the "to map" end of "upload to map".
//
//   RECORD=out/e2e_record.json node e2e/record_map.mjs     # against vite preview on :4173
//
// GET /v1/records/<token> is answered from the file (the page's only API call for a record), so
// it needs no API running. Writes screenshots in both themes next to the record; prints one JSON
// report and exits 1 when a check fails.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { chromium } from "playwright";

const BASE = (process.env.BASE_URL ?? "http://127.0.0.1:4173").replace(/\/$/, "");
const RECORD = process.env.RECORD ?? "out/e2e_record.json";
/** Source: derived, the stranger test's first-map bound (Project Goals: immediate; 10 s covers a cold CDN). */
const FIRST_MAP_MS = 10_000;

const text = readFileSync(RECORD, "utf8");
const record = JSON.parse(text);
const token = record.share_token ?? "e2e-record-token-000";
const shots = dirname(RECORD);
const checks = [];
const check = (name, ok, detail = {}) => checks.push({ name, ok: Boolean(ok), ...detail });

const browser = await chromium.launch();
try {
	for (const theme of ["dusk", "day"]) {
		const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
		await page.addInitScript((t) => localStorage.setItem("pf.theme", t), theme);
		const errors = [];
		page.on("pageerror", (e) => errors.push(e.message));
		page.on("console", (m) => {
			if (m.type() === "error" && !/Failed to load resource/.test(m.text())) errors.push(m.text());
		});
		await page.route("**/v1/records/*", (route) => route.fulfill({ status: 200, contentType: "application/json", body: text }));
		const t0 = Date.now();
		await page.goto(`${BASE}/#/cockpit/${encodeURIComponent(token)}`, { waitUntil: "load" });
		let drew = true;
		try {
			await page.locator(".pf-onode").first().waitFor({ timeout: FIRST_MAP_MS });
		} catch {
			drew = false;
		}
		const nodes = await page.locator(".pf-onode").count();
		const dials = await page.locator(".pf-dial-wrap").evaluateAll((els) =>
			els.map((d) => ({ label: d.querySelector(".pf-dial__label")?.textContent ?? "", status: d.querySelector(".pf-prov")?.textContent ?? "", value: d.querySelector(".pf-dial__text b")?.textContent ?? "" })),
		);
		const status = await page.locator(".pf-statusline").first().textContent().catch(() => null);
		const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
		check(`${theme}: the record opens to a map`, drew && nodes > 0, { ms: Date.now() - t0, nodes, record_nodes: record.graph?.nodes?.length ?? 0 });
		check(`${theme}: readouts draw with a status word`, dials.length >= 3 && dials.every((d) => /Measured|Predicted|Extrapolated|Not measured here|Ran out of memory here/.test(d.status)), { dials, status });
		check(`${theme}: no page errors`, errors.length === 0, { errors: errors.slice(0, 5) });
		check(`${theme}: nothing overflows sideways`, !overflow);
		await page.screenshot({ path: join(shots, `e2e_cockpit_${theme}.png`) });
		await page.close();
	}
} finally {
	await browser.close();
}
const ok = checks.every((c) => c.ok);
console.log(JSON.stringify({ ok, record: RECORD, share_token: token, checks }, null, 2));
process.exit(ok ? 0 : 1);
