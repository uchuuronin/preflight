// Where the site mentions a set of words (PM, Nov 3 item 7: for each Spark feature cut at noon,
// nothing on screen may mention it). Opens Home, New flight (every tab, the data-use detail
// unfolded), every sample's cockpit with its guide, findings and knobs, and the Logbook, at laptop
// and phone size; prints each term's hits by screen, and exits 1 if any term is found.
//
//   BASE=https://preflight-cqn.pages.dev TERMS='Muse Glimmer|Private triage' node e2e/mentions.mjs
//
// TERMS is a list of case-insensitive regular expressions separated by "|" at the top level is fine:
// it is used as one pattern.
import { chromium } from "playwright";

const BASE = (process.env.BASE ?? "http://127.0.0.1:4190").replace(/\/$/, "");
if (!process.env.TERMS || /(^\||\|\||\|$)/.test(process.env.TERMS)) {
	console.error("TERMS: one pattern, no empty alternative (a leading, doubled or trailing |)");
	process.exit(2);
}
const TERMS = new RegExp(process.env.TERMS, "gi");
const SAMPLES = (process.env.SAMPLES ?? "lora-qwen,pretrain,agent-eval,campaign-grid,grpo,retrieval,gnn,vision").split(",");

const browser = await chromium.launch();
const hits = [];
for (const [size, w, h] of [["laptop", 1366, 768], ["phone", 390, 844]]) {
	const page = await browser.newPage({ viewport: { width: w, height: h } });
	await page.addInitScript(() => localStorage.setItem("pf.onboarding.v1", "1"));
	const look = async (screen) => {
		const text = await page.evaluate(() => {
			// The active screen only: the others stay mounted off-screen (Shell.tsx), and the nav.
			const root = document.querySelector('main.pf-screen[data-pos="active"]') ?? document.body;
			for (const d of root.querySelectorAll("details")) d.open = true;
			return root.innerText + "\n" + [...root.querySelectorAll("[title],[aria-label],[placeholder]")].map((e) => `${e.getAttribute("title") ?? ""} ${e.getAttribute("aria-label") ?? ""} ${e.getAttribute("placeholder") ?? ""}`).join("\n");
		});
		for (const m of text.matchAll(TERMS)) {
			const at = Math.max(0, m.index - 60);
			hits.push({ size, screen, term: m[0], context: text.slice(at, m.index + m[0].length + 60).replace(/\s+/g, " ") });
		}
	};
	await page.goto(`${BASE}/#/`, { waitUntil: "networkidle" });
	await look("home");
	await page.goto(`${BASE}/#/new`, { waitUntil: "networkidle" });
	for (const tab of await page.getByRole("tab").all()) {
		await tab.click().catch(() => {});
		await page.waitForTimeout(200);
		await look(`new:${(await tab.textContent())?.trim()}`);
	}
	for (const s of SAMPLES) {
		await page.goto(`${BASE}/#/cockpit/${s}`, { waitUntil: "networkidle" });
		await page.waitForTimeout(800);
		await look(`cockpit:${s}`);
		const chip = page.locator(".pf-findings-chip");
		if (await chip.count()) await chip.first().click().catch(() => {});
		const knob = page.locator(".pf-dock__row .pf-knob").first();
		if (await knob.count()) {
			await knob.click().catch(() => {});
			await page.waitForTimeout(200);
		}
		await look(`cockpit:${s}:open`);
	}
	await page.goto(`${BASE}/#/logbook`, { waitUntil: "networkidle" });
	await look("logbook");
	await page.close();
}
await browser.close();
const seen = new Map();
for (const h of hits) {
	const k = `${h.term.toLowerCase()} @ ${h.screen}`;
	if (!seen.has(k)) seen.set(k, h);
}
console.log(JSON.stringify([...seen.values()], null, 1));
console.log(`${seen.size} distinct hits for ${TERMS.source}`);
process.exit(seen.size ? 1 : 0);
