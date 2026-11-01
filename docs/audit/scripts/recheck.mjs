import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
const [base, out] = process.argv.slice(2);
const browser = await chromium.launch();
const res = {};
const sse = (ev) => ev.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");

// 1. A link that isn't a plain GitHub repo link
{
	const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
	await page.goto(`${base}/#/new`);
	await page.waitForTimeout(1800);
	await page.getByRole("tab", { name: "Repo link" }).click();
	const rows = [];
	for (const link of ["not a link", "https://gitlab.com/a/b", "github.com/owner", "https://github.com/owner/repo"]) {
		const input = page.locator("input[type=url]");
		await input.fill(link);
		await input.blur();
		await page.waitForTimeout(900);
		const invalid = await input.getAttribute("aria-invalid");
		const ids = (await input.getAttribute("aria-describedby")) ?? "";
		const described = [];
		for (const id of ids.split(/\s+/).filter(Boolean)) described.push(await page.locator(`[id="${id}"]`).innerText().catch(() => "?"));
		const live = await page.locator("[aria-live], [role=alert], [role=status]").allInnerTexts();
		rows.push({ link, invalid, described, liveHasMsg: live.some((t) => /public GitHub repository link/.test(t)) });
	}
	res.f1 = rows;
	await page.screenshot({ path: `${out}/f1-badlink-desk.png` });
	await page.close();
}
// 2. Phone: the hint takes a visitor to "Got it", then back to Check in
{
	const page = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true });
	await page.goto(`${base}/?flags=runner.queue#/new`);
	await page.waitForTimeout(1800);
	await page.getByRole("tab", { name: "Repo link" }).click();
	await page.locator("input[type=url]").fill("https://github.com/owner/repo");
	const hint = page.getByRole("button", { name: /Got it/ });
	const hintText = await hint.first().innerText().catch(() => null);
	await hint.first().click();
	await page.waitForTimeout(900);
	const focused = await page.evaluate(() => { const el = document.activeElement; const r = el.getBoundingClientRect(); return { tag: el.tagName, type: el.getAttribute("type"), label: el.closest("label")?.innerText?.slice(0, 60) ?? el.getAttribute("aria-label"), inView: r.top >= 0 && r.bottom <= innerHeight }; });
	await page.screenshot({ path: `${out}/f2-hint-phone.png` });
	await page.keyboard.press("Space");
	await page.waitForTimeout(900);
	const after = await page.evaluate(() => { const el = document.activeElement; const r = el.getBoundingClientRect(); return { text: el.innerText?.slice(0, 40), disabled: el.disabled, inView: r.top >= 0 && r.bottom <= innerHeight }; });
	res.f2 = { hintText, focused, after };
	await page.close();
}
// 3. Samples: honest labels, no "Measured" on a fixture
{
	const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
	await page.goto(`${base}/#/new`);
	await page.waitForTimeout(1500);
	const departures = (await page.locator("body").innerText()).match(/Samples[^\n]*/g);
	await page.goto(`${base}/#/cockpit/lora-7b`);
	await page.waitForTimeout(2500);
	const body = await page.locator("body").innerText();
	res.f3 = { departures, measured: (body.match(/\bMeasured\b[^\n]{0,30}/g) ?? []).slice(0, 5), predicted: (body.match(/\bPredicted\b/g) ?? []).length, illustrative: (body.match(/[Ii]llustrative[^\n]{0,40}/g) ?? []).slice(0, 6), dollars: (body.match(/(US)?\$\d[\d.,]*/g) ?? []).slice(0, 8) };
	await page.screenshot({ path: `${out}/f3-sample-cockpit.png`, fullPage: false });
	await page.close();
}
// 4. Boarding after a failure: which stage stopped
{
	const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
	await page.route("**/v1/jobs/*/events", (r) => r.fulfill({ status: 200, headers: { "content-type": "text/event-stream" }, body: sse([{ stage: "queued" }, { stage: "admitted" }, { stage: "mapping" }, { stage: "probing" }, { stage: "failed", error_code: "oom", message: "error.oom" }]) }));
	await page.goto(`${base}/#/boarding/job_fail`);
	await page.waitForTimeout(2500);
	res.f4 = await page.locator(".pf-stage").evaluateAll((els) => els.map((e) => `${e.className.replace("glass pf-stage", "").trim() || "-"}: ${e.innerText.split("\n")[0]}`));
	res.f4stopped = /Stopped here/i.test(await page.locator("body").innerText());
	await page.screenshot({ path: `${out}/f4-boarding-failed-phone.png`, fullPage: true });
	await page.close();
}
await browser.close();
console.log(JSON.stringify(res, null, 1));
