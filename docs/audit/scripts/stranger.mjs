// D's stranger audit: every sample plus the check-in flow, as a first-time visitor sees them, at
// desk and phone width, light and dark. Run against the deployed commit served by Vite in production
// mode (`cd web && npx vite --mode production --port 5186`, the dev server with the live site's
// flags, flags.production.json; not the built bundle), and the API is mocked here (a check-in that is accepted, boards and
// lands on A's Oct 28 record). For each state: page errors, console errors (API misses aside), axe
// (WCAG 2.0, 2.1 and 2.2 A and AA tags), horizontal overflow, raw template text ("undefined",
// "NaN", "{{" or "{name}"), and a
// screenshot. Read-only: submits nothing anywhere real. Exits 1 on any failure.
//
//   NODE_PATH=web/node_modules node docs/audit/scripts/stranger.mjs http://localhost:5186 path/to/axe.min.js out/
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const { chromium } = createRequire(import.meta.url)("playwright");
const [base, axePath, outDir] = process.argv.slice(2);
if (!base || !axePath || !outDir) {
	console.error("usage: stranger.mjs <server url> <axe.min.js> <out dir>");
	process.exit(2);
}
mkdirSync(outDir, { recursive: true });
const axe = readFileSync(axePath, "utf8");
const record = JSON.parse(readFileSync(fileURLToPath(new URL("../../../api/tests/fixtures/a-smoke-l4-main-4.record.json", import.meta.url)), "utf8"));
const TOKEN = "auditshare0001";
const sse = (events) => events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
const BOARD = [
	{ stage: "queued", queue_position: 0 },
	{ stage: "admitted" },
	{ stage: "fetching" },
	{ stage: "mapping" },
	{ stage: "probing" },
	{ stage: "predicting" },
	{ stage: "done", share_token: TOKEN, record_rev: 1 },
];

const SAMPLES = ["lora-7b", "lora-qwen", "pretrain", "agent", "agent-eval", "campaign-grid", "grpo", "retrieval", "gnn", "vision"];
const PAGES = [
	{ name: "home", hash: "#/" },
	{ name: "logbook", hash: "#/logbook" },
	{ name: "checks", hash: "#/checks" },
	...SAMPLES.map((s) => ({ name: `sample-${s}`, hash: `#/cockpit/${s}` })),
	...SAMPLES.map((s) => ({
		name: `sample-${s}-detail`,
		hash: `#/cockpit/${s}`,
		act: async (p) => {
			const node = p.locator(".pf-orbit [role=button], .pf-orbit button").first();
			await node.click();
			// the opened node's detail sheet
			await p.locator("section.pf-detail").first().waitFor({ timeout: 5000 });
		},
	})),
	{ name: "checkin-desk", hash: "#/new" },
	{
		name: "checkin-bad-link",
		hash: "#/new",
		act: async (p) => {
			await p.getByRole("tab", { name: "Repo link" }).click();
			await p.locator("input[type=url]").fill("not a link");
		},
	},
	{
		name: "checkin-flow",
		hash: "#/new",
		act: async (p) => {
			await p.getByRole("tab", { name: "Repo link" }).click();
			await p.locator("input[type=url]").fill("https://github.com/uchuuronin/preflight/tree/main/zoo/finetune_lora_hf");
			const tick = p.getByRole("checkbox", { name: /Got it/i });
			if (await tick.count()) await tick.check();
			await p.getByRole("button", { name: /^Check in$/ }).click();
			await p.waitForURL(/#\/boarding\//, { timeout: 8000 });
			await p.getByText("Your flight record is ready.").waitFor({ timeout: 8000 });
		},
	},
];
const VIEWS = [["desk", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }]];
const THEMES = ["dusk", "day"];

const browser = await chromium.launch();
const results = [];
for (const pg of PAGES) for (const [vname, vp] of VIEWS) for (const theme of THEMES) {
	const ctx = await browser.newContext({ viewport: vp, colorScheme: theme === "dusk" ? "dark" : "light", hasTouch: vname === "phone" });
	await ctx.addInitScript((th) => {
		try {
			localStorage.setItem("pf.theme", th);
		} catch {}
	}, theme);
	const page = await ctx.newPage();
	const errs = [];
	page.on("pageerror", (e) => errs.push(`pageerror: ${String(e).slice(0, 200)}`));
	page.on("console", (m) => {
		// Expected misses only: the API's (no API behind the dev server) and other hosts' (a sandbox
		// may refuse them, e.g. the web fonts); a failed load from the app's own host still counts.
		const url = m.location()?.url ?? "";
		const expected = /Failed to load resource/.test(m.text()) && (/\/v1\//.test(url) || !url.startsWith(base));
		if (m.type() === "error" && !/\/v1\//.test(m.text()) && !expected) errs.push(`console: ${m.text().slice(0, 200)} ${url}`);
	});
	await page.route("**/v1/jobs", (r) => (r.request().method() === "POST" ? r.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify({ job_id: "job_audit", created: true, position: 0 }) }) : r.fallback()));
	await page.route("**/v1/jobs/job_audit/events", (r) => r.fulfill({ status: 200, headers: { "content-type": "text/event-stream" }, body: sse(BOARD) }));
	await page.route(`**/v1/records/${TOKEN}*`, (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(record) }));
	const row = { page: pg.name, view: vname, theme, problems: [] };
	try {
		await page.goto(`${base}/${pg.hash}`);
		await page.waitForTimeout(2200);
		if (pg.act) await pg.act(page);
		await page.waitForTimeout(600);
		const metrics = await page.evaluate(() => {
			const de = document.documentElement;
			const text = document.body.innerText;
			const raw = (text.match(/\bundefined\b|\bNaN\b|\{\{|\{[a-z_]+\}/g) ?? []).slice(0, 3);
			return { overflow: de.scrollWidth - de.clientWidth, raw, empty: text.trim().length < 40 };
		});
		if (metrics.overflow > 1) row.problems.push(`horizontal overflow ${metrics.overflow}px`);
		if (metrics.raw.length) row.problems.push(`raw text: ${metrics.raw.join(", ")}`);
		if (metrics.empty) row.problems.push("page nearly empty");
		await page.addScriptTag({ content: axe });
		const ax = await page.evaluate(async () => {
			const r = await window.axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] } });
			return r.violations.map((v) => `${v.id} (${v.nodes.length}): ${v.nodes.slice(0, 2).map((n) => `${n.target.join(" ")}${v.id === "color-contrast" && n.any[0] ? ` [${n.any[0].message}]` : ""}`).join(" | ")}`);
		});
		row.problems.push(...ax.map((a) => `axe ${a}`));
	} catch (e) {
		row.problems.push(`step failed: ${String(e).split("\n")[0].slice(0, 200)}`);
	}
	row.problems.push(...errs);
	const shot = `${outDir}/${pg.name}-${vname}-${theme}.jpg`;
	await page.screenshot({ path: shot, type: "jpeg", quality: 60, fullPage: false }).catch(() => {});
	row.shot = shot;
	results.push(row);
	console.log(`${row.problems.length ? "FAIL" : "ok  "} ${pg.name} · ${vname} · ${theme}${row.problems.length ? `\n     ${row.problems.join("\n     ")}` : ""}`);
	await ctx.close();
}
await browser.close();
writeFileSync(`${outDir}/results.json`, JSON.stringify(results, null, 1));
const bad = results.filter((r) => r.problems.length).length;
console.log(`${results.length - bad} of ${results.length} states clean`);
process.exit(bad ? 1 : 0);
