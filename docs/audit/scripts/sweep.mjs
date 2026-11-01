import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
const { chromium } = createRequire(import.meta.url)("playwright");
const [base, axePath, outDir, fixtures] = process.argv.slice(2);
const axe = readFileSync(axePath, "utf8");
const colab = readFileSync(`${fixtures}/colab_lora.ipynb`);

const sse = (events) => events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
const JOBS = {
	job_mid: [{ stage: "queued", queue_position: 2 }, { stage: "admitted" }, { stage: "mapping" }, { stage: "probing" }],
	job_fail: [{ stage: "queued" }, { stage: "probing" }, { stage: "failed", error_code: "oom", message: "error.oom" }],
};

const PAGES = [
	{ name: "home", hash: "#/" },
	{ name: "checkin-repo", hash: "#/new", act: async (p) => { await p.getByRole("tab", { name: "Repo link" }).click(); await p.locator("input[type=url]").fill("not a link"); } },
	{ name: "checkin-upload", hash: "#/new", act: async (p) => { await p.getByRole("tab", { name: "Upload" }).click(); } },
	{ name: "checkin-open-notebook", hash: "#/new", flags: "runner.queue,d.notebook", act: async (p) => {
		await p.getByRole("tab", { name: "Notebook" }).click();
		await p.locator('input[type=file][accept=".ipynb"]').setInputFiles({ name: "train (1).ipynb", mimeType: "application/json", buffer: colab });
		await p.getByRole("status").filter({ hasText: "runs as" }).waitFor();
	} },
	{ name: "checkin-open-upload", hash: "#/new", flags: "runner.queue,d.notebook", act: async (p) => { await p.getByRole("tab", { name: "Upload" }).click(); } },
	{ name: "boarding-mid", hash: "#/boarding/job_mid" },
	{ name: "boarding-failed", hash: "#/boarding/job_fail" },
	{ name: "cockpit", hash: "#/cockpit/lora-qwen" },
	{ name: "cockpit-knob", hash: "#/cockpit/lora-qwen", act: async (p) => { await p.locator(".pf-dock__row .pf-knob").first().click(); } },
	{ name: "checks", hash: "#/checks" },
];
const VIEWS = [["desk", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }]];
const THEMES = ["dusk", "day"];

const browser = await chromium.launch();
const results = [];
for (const pg of PAGES) for (const [vname, vp] of VIEWS) for (const theme of THEMES) {
	const ctx = await browser.newContext({ viewport: vp, colorScheme: theme === "dusk" ? "dark" : "light", hasTouch: vname === "phone" });
	await ctx.addInitScript((th) => { try { localStorage.setItem("pf.theme", th); } catch {} }, theme);
	const page = await ctx.newPage();
	const errs = [];
	page.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));
	page.on("console", (m) => { if (m.type() === "error" && !/ERR_TUNNEL|Failed to load resource|fonts\.g/.test(m.text())) errs.push(m.text().slice(0, 200)); });
	await page.route("**/v1/jobs/*/events", async (route) => {
		const id = route.request().url().split("/v1/jobs/")[1].split("/")[0];
		await route.fulfill({ status: 200, headers: { "content-type": "text/event-stream" }, body: sse(JOBS[id] ?? []) });
	});
	const row = { page: pg.name, view: vname, theme, errs };
	try {
		await page.goto(`${base}/${pg.flags ? `?flags=${pg.flags}` : ""}${pg.hash}`);
		await page.waitForTimeout(2200);
		if (pg.act) await pg.act(page);
		await page.waitForTimeout(500);
		row.overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
		row.small = await page.evaluate(() => {
			const out = [];
			for (const el of document.querySelectorAll("a[href], button, input, select, textarea, summary, [role=tab], [role=radio], [tabindex]:not([tabindex='-1'])")) {
				const r = el.getBoundingClientRect();
				const st = getComputedStyle(el);
				if (!r.width || !r.height || st.visibility === "hidden" || st.opacity === "0") continue;
				if (el.matches("input[type=file]")) continue; // visually hidden inside its label
				if (r.width < 24 || r.height < 24) out.push(`${el.tagName.toLowerCase()}${el.className ? "." + String(el.className).split(" ")[0] : ""} "${(el.getAttribute("aria-label") || el.textContent || "").trim().slice(0, 30)}" ${Math.round(r.width)}x${Math.round(r.height)}`);
			}
			return [...new Set(out)].slice(0, 12);
		});
		// focus: tab 12 times, record elements whose focus isn't visible
		row.focusInvisible = [];
		for (let i = 0; i < 12; i++) {
			await page.keyboard.press("Tab");
			const f = await page.evaluate(() => {
				const el = document.activeElement;
				if (!el || el === document.body) return null;
				const s = getComputedStyle(el);
				const ring = (s.outlineStyle !== "none" && parseFloat(s.outlineWidth) > 0) || (s.boxShadow && s.boxShadow !== "none");
				const parentRing = el.parentElement && getComputedStyle(el.parentElement).outlineStyle !== "none";
				return ring || parentRing ? null : `${el.tagName.toLowerCase()} "${(el.getAttribute("aria-label") || el.textContent || "").trim().slice(0, 30)}"`;
			});
			if (f) row.focusInvisible.push(f);
		}
		await page.keyboard.press("Escape");
		await page.evaluate(() => window.scrollTo(0, 0));
		await page.addScriptTag({ content: axe });
		row.axe = await page.evaluate(async () => {
			const r = await window.axe.run(document, { runOnly: ["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"] });
			return r.violations.map((v) => ({ id: v.id, impact: v.impact, n: v.nodes.length, where: v.nodes.slice(0, 3).map((n) => n.target.join(" ")), why: v.nodes[0]?.failureSummary?.split("\n").slice(1, 2).join(" ").trim() }));
		});
		const shot = `${outDir}/shots/${pg.name}-${vname}-${theme}.png`;
		await page.screenshot({ path: shot, fullPage: true });
		row.shot = shot;
	} catch (e) {
		row.failed = String(e).slice(0, 300);
	}
	results.push(row);
	await ctx.close();
}
await browser.close();
writeFileSync(`${outDir}/results.json`, JSON.stringify(results, null, 1));
console.log(`${results.length} states`);
