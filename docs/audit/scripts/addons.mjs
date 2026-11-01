// D's add-ons still render: for each one, with its flag on it appears where it belongs, with no
// page or console errors, no axe violations inside it and nothing under 24 px; with its flag off
// the host screen still loads, cleanly, and the add-on is absent. Read-only (opens pages,
// submits nothing). Exits 1 on any failure. Each axe node is logged with its colours, so a
// contrast failure names the pair that caused it.
//
//   cd web && npx vite --port 5185 --strictPort
//   NODE_PATH=web/node_modules node docs/audit/scripts/addons.mjs http://localhost:5185 path/to/axe.min.js [out.json]
//
// Run it against the dev server only: flags come from the URL (?flags=a,-b), which production
// builds (vite build, vite preview) ignore.
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const { chromium } = createRequire(import.meta.url)("playwright");
const [base, axePath, outPath] = process.argv.slice(2);
if (!base || !axePath) {
	console.error("usage: addons.mjs <dev server url> <axe.min.js> [out.json]");
	process.exit(2);
}
const axe = readFileSync(axePath, "utf8");
const NOTEBOOK = fileURLToPath(new URL("../../../web/src/addons/notebook/fixtures/colab_lora.ipynb", import.meta.url));

const notebookTab = (p) => p.getByRole("tab", { name: "Notebook" });
const COCKPIT = { hash: "#/cockpit/lora-7b", host: "section.pf-cockpit" };
const ADDONS = [
	// D1: the dial in the cockpit head, and the glossary card in an open knob's control (the dock
	// shows one control at a time, so open knobs in turn until one has a card, then open the card).
	{
		id: "D1 expertise dial",
		flag: "d.expertise",
		...COCKPIT,
		roots: [".pf-addon-expertise", ".pf-addon-gloss"],
		act: async (p) => {
			const knobs = p.locator(".pf-dock__row .pf-knob:not(.pf-knob--more)");
			const n = await knobs.count();
			for (let i = 0; i < n && !(await p.locator(".pf-addon-gloss").count()); i++) {
				await knobs.nth(i).click();
				await p.waitForTimeout(300);
			}
			await p.locator(".pf-addon-gloss > summary").first().click();
		},
		// With the flag off, open a knob anyway: its control must have no card.
		offAct: async (p) => {
			await p.locator(".pf-dock__row .pf-knob:not(.pf-knob--more)").first().click();
			await p.waitForSelector(".pf-control", { timeout: 8000 });
		},
	},
	{ id: "D3 currency", flag: "d.currency", hash: "#/logbook", host: "section.pf-pilot", roots: [".pf-addon-currency"] },
	{ id: "D4 free-tier band", flag: "d.freetier", ...COCKPIT, roots: [".pf-addon-freetier"] },
	{
		id: "D5 notebook intake",
		flag: "d.notebook",
		hash: "#/new",
		host: "section.pf-desk",
		roots: [".pf-addon-notebook"],
		// Choose the tab, then a real notebook, and open "What will run", so the preview and its
		// script and requirements are checked too.
		act: async (p) => {
			await notebookTab(p).click();
			await p.locator(".pf-addon-notebook input[type=file]").setInputFiles(NOTEBOOK);
			await p.waitForSelector(".pf-addon-notebook__preview", { timeout: 8000 });
			await p.locator(".pf-addon-notebook__files > summary").click();
			await p.waitForSelector(".pf-addon-notebook__files[open] pre", { state: "visible", timeout: 8000 });
		},
		// With the flag off there is no Notebook tab at all.
		absent: async (p) => ((await notebookTab(p).count()) > 0 ? ["a Notebook tab"] : []),
	},
	{ id: "exports", flag: "c.exports", ...COCKPIT, roots: [".pf-addon-exports"] },
];
const VIEWS = [
	["desk", { width: 1440, height: 900 }],
	["phone", { width: 390, height: 844 }],
];
const THEMES = ["dusk", "day"];
// Only the web-font host and the sandbox proxy's tunnel refusals; any other console error fails.
// ERR_TUNNEL is not tied to a host, so it would also hide a blocked request to anything else
// (harmless where the check runs: nothing the add-ons load goes off the dev server).
const NOISE = /ERR_TUNNEL|fonts\.g(oogleapis|static)\.com/;
// The dev server proxies /v1 to an API on :8000. With none running, the cockpit's own limits
// request gets a 502: that is the test setup, so it is printed as a warning naming the path. Only
// that path; a 502 on any other /v1 path, an add-on's included, fails like every other error.
const NO_API_PATHS = new Set(["/v1/guide/limits/state"]);
const NO_API = (m) => /status of 502/.test(m.text()) && NO_API_PATHS.has(new URL(m.location().url || "x:", "http://h").pathname);
const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

async function open(browser, view, theme, url, host) {
	const ctx = await browser.newContext({ viewport: view, colorScheme: theme === "dusk" ? "dark" : "light" });
	await ctx.addInitScript((th) => {
		try {
			localStorage.setItem("pf.theme", th);
		} catch {}
	}, theme);
	const page = await ctx.newPage();
	const errors = [];
	const noApi = new Set();
	page.on("pageerror", (e) => errors.push(String(e).slice(0, 200)));
	page.on("console", (m) => {
		if (m.type() !== "error" || NOISE.test(m.text())) return;
		if (NO_API(m)) noApi.add(new URL(m.location().url).pathname);
		else errors.push(m.text().slice(0, 200));
	});
	await page.goto(url);
	let loaded = true;
	try {
		await page.waitForSelector(host, { timeout: 10000 });
		await page.waitForTimeout(800);
	} catch {
		loaded = false;
	}
	return { ctx, page, errors, noApi, loaded };
}

async function check(page, sel) {
	const small = await page.locator(sel).first().evaluate((root) => {
		const out = [];
		for (const el of root.querySelectorAll("a[href], button, input, select, textarea, summary, [role=tab], [role=radio]")) {
			const r = el.getBoundingClientRect();
			if (!r.width || !r.height || el.matches("input[type=file]")) continue;
			if (r.width < 24 || r.height < 24) out.push(`${el.tagName.toLowerCase()} ${Math.round(r.width)}x${Math.round(r.height)}`);
		}
		return out;
	});
	const { violations, unmeasured } = await page.evaluate(
		async ([s, tags]) => {
			// Every match of the selector, so each glossary card is covered.
			const include = [...document.querySelectorAll(s)];
			const r = await window.axe.run({ include }, { runOnly: tags });
			const unmeasured = r.incomplete
				.filter((v) => v.id === "color-contrast")
				.flatMap((v) => v.nodes.map((n) => `${n.target.join(" ")} (${n.any[0]?.message ?? "no reason given"})`));
			const violations = r.violations.map((v) => ({
				id: v.id,
				nodes: v.nodes.map((n) => {
					const d = n.any.find((c) => c.data && c.data.fgColor)?.data;
					return d ? `${n.target.join(" ")} ${d.fgColor} on ${d.bgColor} = ${d.contrastRatio}:1` : n.target.join(" ");
				}),
			}));
			return { violations, unmeasured };
		},
		[sel, TAGS],
	);
	return { small, violations, unmeasured };
}

const browser = await chromium.launch();
const rows = [];
for (const a of ADDONS) {
	for (const [vname, vp] of VIEWS) {
		for (const theme of THEMES) {
			const row = { addon: a.id, view: vname, theme, problems: [], warnings: [], axe: [] };
			// Flag on: it renders where it belongs, cleanly.
			const on = await open(browser, vp, theme, `${base}/?flags=${a.flag}${a.hash}`, a.host);
			if (!on.loaded) row.problems.push(`host screen ${a.host} did not load with ${a.flag} on`);
			else
				try {
					if (a.act) await a.act(on.page);
					await on.page.addScriptTag({ content: axe });
					for (const sel of a.roots) {
						await on.page.waitForSelector(sel, { timeout: 8000 });
						await on.page.locator(sel).first().scrollIntoViewIfNeeded();
						const n = await on.page.locator(sel).count();
						const { small, violations, unmeasured } = await check(on.page, sel);
						// Text axe can't measure (over a gradient or overlapped) passes it; say so.
						for (const u of unmeasured) row.warnings.push(`${sel}: contrast not measured, check by eye: ${u}`);
						if (small.length) row.problems.push(`${sel}: under 24 px: ${small.join(", ")}`);
						for (const v of violations) {
							row.problems.push(`${sel} (${n} on screen): axe ${v.id} (${v.nodes.length})`);
							row.axe.push({ root: sel, ...v });
						}
					}
				} catch (e) {
					row.problems.push(`did not render with ${a.flag} on: ${String(e).split("\n")[0].slice(0, 160)}`);
				}
			if (on.errors.length) row.problems.push(`errors with ${a.flag} on: ${on.errors.join(" | ")}`);
			for (const u of on.noApi) row.warnings.push(`no API behind the dev server: ${u} (502), with ${a.flag} on`);
			await on.ctx.close();
			// Flag off, explicitly: the host screen still loads cleanly and the add-on is absent.
			const off = await open(browser, vp, theme, `${base}/?flags=-${a.flag}${a.hash}`, a.host);
			if (!off.loaded) row.problems.push(`host screen ${a.host} did not load with ${a.flag} off`);
			else {
				if (a.offAct) await a.offAct(off.page).catch((e) => row.problems.push(`with ${a.flag} off: ${String(e).split("\n")[0].slice(0, 160)}`));
				const left = [];
				for (const sel of a.roots) if ((await off.page.locator(sel).count()) > 0) left.push(sel);
				if (a.absent) left.push(...(await a.absent(off.page)));
				if (left.length) row.problems.push(`shows with ${a.flag} off: ${left.join(", ")}`);
			}
			if (off.errors.length) row.problems.push(`errors with ${a.flag} off: ${off.errors.join(" | ")}`);
			for (const u of off.noApi) row.warnings.push(`no API behind the dev server: ${u} (502), with ${a.flag} off`);
			await off.ctx.close();
			rows.push(row);
		}
	}
}
await browser.close();
const failed = rows.filter((r) => r.problems.length);
if (outPath) writeFileSync(outPath, JSON.stringify(rows, null, 1));
for (const r of rows) for (const w of r.warnings) console.log(`WARN ${r.addon} · ${r.view} · ${r.theme}: ${w}`);
for (const r of failed) {
	console.log(`FAIL ${r.addon} · ${r.view} · ${r.theme}: ${r.problems.join("; ")}`);
	for (const v of r.axe) for (const n of v.nodes) console.log(`    ${v.id}: ${n}`);
}
console.log(`${rows.length - failed.length} of ${rows.length} add-on states pass`);
process.exit(failed.length ? 1 : 0);
