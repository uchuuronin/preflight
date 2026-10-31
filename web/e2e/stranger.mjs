// The stranger test: what someone who has never seen Preflight does in their first minutes,
// checked in a real browser against a built app. The deploy workflow runs it on the build it
// is about to publish and stops the deploy if any check fails; CI runs it on every push.
//
//   node e2e/stranger.mjs                      # against http://127.0.0.1:4173 (vite preview)
//   BASE_URL=https://preview.example node e2e/stranger.mjs
//
// Prints one JSON report; exits 1 when a check fails. Thresholds carry their source.
import { chromium } from "playwright";

const BASE = (process.env.BASE_URL ?? "http://127.0.0.1:4173").replace(/\/$/, "");

/** Contract: every knob change reflected in under 400 ms (Project Goals, "The workspace"). */
const KNOB_P95_MS = 400;
/**
 * Lighthouse's mobile setting: 4x CPU slowdown. The knob check runs under it, so a pass means
 * a mid-range laptop or phone, not the CI machine, gets the 400 ms.
 */
const CPU_SLOWDOWN = 4;
/** Project Goals: "first map on screen" for a sample is immediate; 10 s covers a cold CDN. */
const FIRST_MAP_MS = 10_000;
/** Phone width the layout must fit without sideways scroll (artifact rule: 16 px gutters). */
const PHONE = { width: 390, height: 844 };

const checks = [];
const check = (name, ok, detail = {}) => checks.push({ name, ok: Boolean(ok), ...detail });

const browser = await chromium.launch();
try {
	// ---------- desktop: home -> sample -> cockpit
	const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
	const errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	page.on("console", (m) => {
		// Fonts and other third-party loads can fail in CI without hurting a visitor.
		if (m.type() === "error" && !/Failed to load resource/.test(m.text())) errors.push(m.text());
	});

	await page.goto(`${BASE}/`, { waitUntil: "load" });
	const t0 = Date.now();
	await page.getByRole("button", { name: "Open a sample flight" }).or(page.getByRole("link", { name: "Open a sample flight" })).first().click();
	await page.locator(".pf-dial-wrap").first().waitFor({ timeout: FIRST_MAP_MS });
	await page.locator(".pf-onode").first().waitFor({ timeout: FIRST_MAP_MS });
	check("sample opens to a map", true, { ms: Date.now() - t0 });

	// Every dial says where its number comes from, in words.
	const dials = await page.locator(".pf-dial-wrap").evaluateAll((els) =>
		els.map((d) => ({
			label: d.querySelector(".pf-dial__label")?.textContent ?? "",
			status: d.querySelector(".pf-prov")?.textContent ?? "",
			value: d.querySelector(".pf-dial__text b")?.textContent ?? "",
		})),
	);
	check("four readouts", dials.length >= 4, { dials: dials.length });
	check(
		"every dial has a status word",
		dials.every((d) => /Measured|Predicted|Extrapolated|Not trusted|Illustrative/.test(d.status)),
		{ dials },
	);

	// One plain status line above the readouts, from the readouts.
	const status = (await page.locator(".pf-statusline").textContent())?.trim() ?? "";
	check("a status line says where the run stands", /memory|Fits|finishes|Not trusted/.test(status), { status });

	// What was measured: a capsule opens a drawer of probe chips, Escape closes it.
	const measured = page.locator(".pf-tag--button");
	if (await measured.count()) {
		await measured.click();
		const chips = await page.locator(".pf-drawer.is-open .pf-chip").count();
		await page.keyboard.press("Escape");
		const closed = (await page.locator(".pf-drawer.is-open").count()) === 0;
		check("probes open as chips in a drawer", chips > 0 && closed, { chips });
	}

	// Every button has a name a screen reader can say.
	const unnamed = await page.locator("button").evaluateAll((els) =>
		els.filter((b) => !(b.getAttribute("aria-label") || b.textContent?.trim() || b.getAttribute("title"))).length,
	);
	check("every button has a name", unnamed === 0, { unnamed });

	// ---------- knobs answer in time, under a slow CPU
	const cdp = await page.context().newCDPSession(page);
	await cdp.send("Emulation.setCPUThrottlingRate", { rate: CPU_SLOWDOWN });
	const before = await page.locator(".pf-dial-wrap").first().textContent();
	await page.locator(".pf-dock__row .pf-knob").first().click();
	const range = page.locator(".pf-control input[type=range]").first();
	if (await range.count()) {
		await range.focus();
		for (let i = 0; i < 12; i++) {
			await page.keyboard.press(i % 3 === 2 ? "ArrowLeft" : "ArrowRight");
			await page.waitForTimeout(250);
		}
	} else {
		const opts = page.locator(".pf-control .pf-choice button");
		const n = await opts.count();
		for (let i = 0; i < Math.min(n, 6); i++) {
			await opts.nth(i).click();
			await page.waitForTimeout(250);
		}
	}
	await page.keyboard.press("Escape");
	await cdp.send("Emulation.setCPUThrottlingRate", { rate: 1 });
	const knob = await page.evaluate(() => {
		const s = window.__preflight?.samples() ?? [];
		const sorted = [...s].sort((a, b) => a - b);
		return { n: s.length, p95: sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] : null };
	});
	const after = await page.locator(".pf-dial-wrap").first().textContent();
	check("moving a knob changes the readouts", knob.n > 0 && before !== after, { changes: knob.n });
	check(`knob to readout p95 under ${KNOB_P95_MS} ms at ${CPU_SLOWDOWN}x CPU slowdown`, knob.p95 !== null && knob.p95 < KNOB_P95_MS, {
		p95_ms: knob.p95 === null ? null : Math.round(knob.p95),
	});

	// ---------- the guide, on a fresh cockpit: start the tour, take its first step
	await page.reload({ waitUntil: "load" });
	await page.locator(".pf-onode").first().waitFor({ timeout: FIRST_MAP_MS });
	await page.getByRole("button", { name: "Start the tour" }).click();
	const apply = page.getByRole("button", { name: /^Apply / });
	const step = page.locator(".pf-guide__say").first();
	await step.waitFor({ timeout: 10_000 });
	const said = (await step.textContent())?.trim() ?? "";
	check("the guide says something", said.length > 0, { said: said.slice(0, 160) });
	if (await apply.count()) {
		const knobsBefore = await page.locator(".pf-dock__row").textContent();
		await apply.first().click();
		await page.waitForTimeout(500);
		const knobsAfter = await page.locator(".pf-dock__row").textContent();
		check("applying the guide's move keeps the cockpit working", knobsAfter !== null && errors.length === 0, {
			changed: knobsBefore !== knobsAfter,
		});
	}
	check("the guide offers a move on the sample", (await apply.count()) > 0);

	// ---------- check-in: the data-use notice comes first
	await page.goto(`${BASE}/#/new`, { waitUntil: "load" });
	await page.evaluate(() => {
		try {
			localStorage.removeItem("pf.data_use.v1");
		} catch {}
	});
	await page.reload({ waitUntil: "load" });
	await page.getByLabel("GitHub repository").fill("github.com/karpathy/nanoGPT");
	const checkIn = page.getByRole("button", { name: "Check in", exact: true });
	check("data-use notice is shown", await page.locator(".pf-datause").isVisible());
	check("check-in waits for the notice", await checkIn.isDisabled());
	await page.getByLabel("Got it").check();
	check("ticking the notice enables check-in", await checkIn.isEnabled());

	check("no page errors", errors.length === 0, { errors: errors.slice(0, 5) });
	await page.close();

	// ---------- a phone: nothing scrolls sideways
	const phone = await browser.newPage({ viewport: PHONE, isMobile: true, hasTouch: true });
	for (const path of ["/", "/#/new", "/#/cockpit/lora-7b"]) {
		await phone.goto(`${BASE}${path}`, { waitUntil: "load" });
		await phone.waitForTimeout(800);
		const over = await phone.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
		check(`no sideways scroll on a phone at ${path}`, over <= 1, { overflow_px: over });
	}
	await phone.close();
} catch (e) {
	check("the walk-through finished", false, { error: String(e).slice(0, 400) });
} finally {
	await browser.close();
}

const failed = checks.filter((c) => !c.ok);
console.log(JSON.stringify({ base: BASE, passed: checks.length - failed.length, failed: failed.length, checks }, null, 1));
if (process.env.GITHUB_ACTIONS) {
	for (const c of failed) console.log(`::error title=stranger test::${c.name} ${JSON.stringify(c).slice(0, 300)}`);
	console.log(`::notice title=stranger test::${checks.length - failed.length}/${checks.length} checks passed`);
}
process.exit(failed.length ? 1 : 0);
