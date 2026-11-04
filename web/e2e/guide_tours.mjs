// The live guide's tour, taken TOURS times on the LoRA sample of the site at BASE_URL (C, Nov 4: D's C2
// take handed over to the rule-based tour on a slow round). Each tour opens a fresh page (a fresh guide
// session), presses "Start the tour" and records every /v1/guide/turns request: its time, status,
// output tokens and the model's calls. A tour passes when it ends with Muse Spark still the guide
// ("Listening" or a move to decide) and no "takes over" line. Public pages only; it never checks in.
// Prints one JSON line per turn and a summary; exits 1 if any tour handed over.
import { chromium } from "playwright";

const BASE = process.env.BASE_URL ?? "https://preflight-cqn.pages.dev";
const TOURS = Number(process.env.TOURS ?? 3);
const WAIT_MS = 120_000;

const browser = await chromium.launch();
let handed = 0;
for (let n = 1; n <= TOURS; n++) {
	const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
	const turns = [];
	let pending = 0;
	page.on("request", (req) => {
		if (!req.url().includes("/v1/guide/turns")) return;
		req.__t0 = Date.now();
		pending += 1;
	});
	page.on("response", async (res) => {
		const req = res.request();
		if (!req.url().includes("/v1/guide/turns")) return;
		pending -= 1;
		let body = null;
		try {
			body = await res.json();
		} catch {}
		const p = body?.provenance ?? {};
		turns.push({
			tour: n,
			turn: turns.length + 1,
			ms: Date.now() - (req.__t0 ?? Date.now()),
			status: res.status(),
			output_tokens: p.output_tokens ?? null,
			calls: (body?.calls ?? []).map((c) => c.name),
			awaiting: body?.awaiting ?? null,
			error: body?.error?.reason ?? null,
		});
	});
	await page.goto(`${BASE}/#/cockpit/lora-qwen`, { waitUntil: "networkidle" });
	const guide = page.locator('main[data-pos="active"] .pf-guide');
	await guide.getByRole("button", { name: "Start the tour" }).click();
	const t0 = Date.now();
	let state = "timeout";
	while (Date.now() - t0 < WAIT_MS) {
		await page.waitForTimeout(1000);
		const text = await guide.innerText();
		if (/takes over/.test(text) || /^Rule-based tour/.test(text)) {
			state = "handed_over";
			break;
		}
		// Muse Spark has answered and waits on the visitor (not on the browser's tool results), nothing in flight
		const last = turns.at(-1);
		if (last && !pending && last.status === 200 && last.awaiting && last.awaiting !== "results") {
			state = "muse_spark";
			break;
		}
	}
	for (const t of turns) console.log(JSON.stringify(t));
	const slowest = Math.max(0, ...turns.map((t) => t.ms));
	console.log(JSON.stringify({ tour: n, result: state, turns: turns.length, slowest_ms: slowest }));
	if (state !== "muse_spark") handed += 1;
	await page.close();
}
await browser.close();
console.log(JSON.stringify({ tours: TOURS, handed_over_or_stalled: handed }));
process.exit(handed ? 1 : 0);
