import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { duration, formatValue, headline, type RecordLike } from "../functions/_card/headline";
import { HERO_CARDS } from "../functions/_card/hero";
import { renderPng } from "../functions/_card/render";
import { badge, cardPng, sharePage } from "../functions/_card/routes";
import { badgeSvg, CARD_H, CARD_W, cardSvg, wrap } from "../functions/_card/svg";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..");
const read = (p: string) => JSON.parse(readFileSync(join(repo, p), "utf8")) as RecordLike;

// Real records: A's Oct 28 L4 smoke (measured probes, no engine estimates) and A's Oct 31 agent
// sample; the golden fine-tune (estimates from fixture models or, since round five, B's engine); C's GRPO sample from A's L4 run.
const smoke = read("api/tests/fixtures/a-smoke-l4-main-4.record.json");
const golden = read("schema/examples/finetune.json");
const agent = read("web/src/workspace/fixtures/agent-eval-oct31.record.json");
const grpo = read("web/src/workspace/fixtures/grpo-l4.record.json");

// The golden records are filled by the fixture models or, since round five, by B's engine, so the
// tests that need illustrative numbers mark every model in a copy as a fixture, and the rest
// assert relations rather than pinned values: they hold on either set.
function asFixture(rec: RecordLike): RecordLike {
	const ests = Object.values(rec.predictions?.estimates ?? {});
	const ids = new Set(ests.map((e) => e?.model_id).filter((x): x is string => !!x));
	return { ...rec, models: [...ids].map((id) => ({ id, fixture: true })) };
}
const goldenPeak = golden.predictions?.estimates?.["memory.peak"];

const ENV = { PREFLIGHT_API_ORIGIN: "https://api.example", PREFLIGHT_EDGE_SECRET: "s3cret", PREFLIGHT_SHARE_CARD_IMAGES: "on" };
const TOKEN = "test-share-token-0001";

function api(record: RecordLike | null, status = 200) {
	return vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit) =>
		record ? new Response(JSON.stringify(record), { status, headers: { "content-type": "application/json" } }) : new Response("{}", { status }),
	);
}

describe("headline", () => {
	it("uses the probes' own measurements when a record has no estimates", () => {
		const h = headline(smoke);
		expect(h.lines.every((l) => l.status === "Measured")).toBe(true);
		// The eval OOM: the line says where, at the eval batch size, and the peak is the eval pass's
		// own (21,902,642,688 bytes), not the train steps' 14.6 GiB.
		expect(h.lines[0]).toEqual({ label: "Ran out of memory in the evaluation pass", value: "eval batch size 32", status: "Measured" });
		expect(h.lines[1]?.value).toBe("20.4 of 22.06 GiB");
		expect(h.lines[2]?.label).toBe("Step time at batch size 4, seq len 1024");
		expect(h.subtitle).toMatch(/^L4/);
		expect(h.illustrative).toBe(false);
	});

	it("calls a fixture model's numbers illustrative, never predicted", () => {
		const h = headline(asFixture(golden));
		expect(h.title).toBe("LoRA fine-tune of Qwen2.5-1.5B on Alpaca");
		expect(h.lines.map((l) => l.label)).toEqual(["Peak memory", "Time to finish", "Run cost", "Chance of running out of memory"]);
		expect(h.lines.every((l) => l.status === "Illustrative")).toBe(true);
		expect(h.illustrative).toBe(true);
		// The number is the record's own, converted: its upper end in GiB.
		expect(typeof goldenPeak?.hi).toBe("number");
		expect(h.lines[0]?.value).toContain(formatValue(goldenPeak?.hi as number, "bytes"));
		expect(h.lines[0]?.value).toMatch(/ to /);
	});

	it("never calls the golden record's numbers illustrative unless its models are fixtures", () => {
		const fixtures = new Set((golden.models ?? []).filter((m) => m.fixture).map((m) => m.id));
		const h = headline(golden);
		// Only the four estimates the card shows decide it.
		const shown = ["memory.peak", "time.total", "cost.total", "risk.oom"].map((k) => golden.predictions?.estimates?.[k]);
		const fromFixture = shown.some((e) => e?.point != null && !!e.model_id && fixtures.has(e.model_id));
		expect(h.illustrative).toBe(fromFixture);
	});

	it("maps each estimate status to the app's word, and gives no number where it abstains", () => {
		const rec: RecordLike = {
			job: { lens: "finetune", source: { entrypoint: "train.py" } },
			predictions: {
				estimates: {
					"memory.peak": { point: 9e9, lo: 8e9, hi: 1e10, unit: "bytes", status: "measured", model_id: "m" },
					"time.total": { point: 4400, lo: 3700, hi: 5100, unit: "s", status: "extrapolated", model_id: "m" },
					"cost.total": { point: null, unit: "usd", status: "abstain", model_id: "m" },
					"risk.oom": { point: 0.1, lo: 0.05, hi: 0.2, unit: "probability", status: "predicted", model_id: "m" },
				},
			},
			models: [{ id: "m", fixture: false }],
		};
		const h = headline(rec);
		expect(h.title).toBe("Fine-tune · train.py");
		expect(h.lines.map((l) => [l.value, l.status])).toEqual([
			["7.45 to 9.31 GiB", "Measured"],
			["1 h 2 min to 1 h 25 min", "Extrapolated"],
			["No number", "Not measured here"],
			["5% to 20%", "Predicted"],
		]);
	});

	it("reads real agent and GRPO records without throwing, every line with a status word", () => {
		for (const r of [agent, grpo]) {
			const h = headline(r);
			expect(h.lines.length).toBeGreaterThan(0);
			for (const l of h.lines) expect(["Measured", "Predicted", "Extrapolated", "Not measured here", "Illustrative"]).toContain(l.status);
		}
	});

	it("never throws on an empty or odd record", () => {
		expect(headline({}).title).toBe("A Preflight flight record");
		expect(headline({ probes: [{ status: "ok" }], predictions: { estimates: {} } }).lines).toEqual([]);
	});

	it("formats units the way the card shows them", () => {
		expect(duration(1.68)).toBe("1.68 s");
		expect(duration(0.041)).toBe("41 ms");
		expect(duration(0.0042)).toBe("4.2 ms");
		expect(duration(405)).toBe("7 min");
		expect(duration(4399.6)).toBe("1 h 13 min");
		expect(formatValue(0.9777, "usd")).toBe("US$0.98");
		expect(formatValue(0.97, "probability")).toBe("97%");
		expect(formatValue(2.5e6, "tokens")).toBe("2.5M tokens");
	});
});

describe("svg", () => {
	it("escapes record text and wraps long titles to two lines", () => {
		const h = { title: `<script>alert("x")</script> ${"word ".repeat(30)}`, subtitle: "L4 & co", lines: [], illustrative: false };
		const svg = cardSvg(h, "preflight.example");
		expect(svg).not.toContain("<script>");
		expect(svg).toContain("&lt;script&gt;");
		expect(svg).toContain("L4 &amp; co");
		expect(wrap("a ".repeat(100), 34, 2)).toHaveLength(2);
	});

	it("labels the badge for screen readers with its number and status", () => {
		const h = headline(asFixture(golden));
		const svg = badgeSvg(h);
		expect(svg).toContain(`aria-label="preflight: peak memory ${h.lines[0]?.value} · illustrative"`);
		expect(h.lines[0]?.value).toMatch(/GiB$/);
	});
});

describe("routes", () => {
	const req = (path: string) => new Request(`https://preflight.example${path}`, { headers: { "CF-Connecting-IP": "203.0.113.7" } });

	it("serves a share page with preview tags that sends people on to the cockpit", async () => {
		const f = api(smoke);
		const res = await sharePage(req(`/s/${TOKEN}`), ENV, TOKEN, f);
		const html = await res.text();
		expect(res.status).toBe(200);
		expect(html).toContain(`<meta property="og:image" content="https://preflight.example/card/${TOKEN}">`);
		expect(html).toContain('<meta name="twitter:card" content="summary_large_image">');
		expect(html).toContain(`url=https://preflight.example/#/cockpit/${TOKEN}`);
		expect(html).toContain("(measured)");
		// The API is asked with the edge secret and the visitor's address, as C's proxy does.
		const [url, init] = f.mock.calls[0] ?? [];
		expect(url).toBe(`https://api.example/v1/records/${TOKEN}`);
		expect(new Headers(init?.headers).get("X-Preflight-Edge")).toBe("s3cret");
		expect(new Headers(init?.headers).get("X-Preflight-Client-IP")).toBe("203.0.113.7");
	});

	it("points previews at the card only when the deploy turns card images on", async () => {
		const off = { PREFLIGHT_API_ORIGIN: "https://api.example", PREFLIGHT_EDGE_SECRET: "s3cret" };
		const html = await (await sharePage(req(`/s/${TOKEN}`), off, TOKEN, api(smoke))).text();
		expect(html).not.toContain("og:image");
		expect(html).toContain('<meta name="twitter:card" content="summary">');
		expect(html).toContain("(measured)");
	});

	it("points a hero record's preview at its pre-rendered card, with card images off", async () => {
		const off = { PREFLIGHT_API_ORIGIN: "https://api.example", PREFLIGHT_EDGE_SECRET: "s3cret" };
		const hero = [...HERO_CARDS][0] as string;
		const html = await (await sharePage(req(`/s/${hero}`), off, hero, api(smoke))).text();
		expect(html).toContain(`<meta property="og:image" content="https://preflight.example/cards/${hero}.png">`);
		expect(html).toContain('<meta name="twitter:card" content="summary_large_image">');
		expect(existsSync(join(here, "..", "static", "cards", `${hero}.png`))).toBe(true);
	});

	it("never renders per request with card images off: a hero goes to its file, others are not served", async () => {
		const off = { PREFLIGHT_API_ORIGIN: "https://api.example", PREFLIGHT_EDGE_SECRET: "s3cret" };
		const png = vi.fn(async () => new Uint8Array(1));
		const f = api(smoke);
		const hero = [...HERO_CARDS][0] as string;
		const toFile = await cardPng(req(`/card/${hero}`), off, hero, { png }, f);
		expect(toFile.status).toBe(302);
		expect(toFile.headers.get("location")).toBe(`https://preflight.example/cards/${hero}.png`);
		expect((await cardPng(req(`/card/${TOKEN}`), off, TOKEN, { png }, f)).status).toBe(404);
		expect(png).not.toHaveBeenCalled();
		expect(f).not.toHaveBeenCalled();
	});

	it("says so when no record matches, with no image tag", async () => {
		const res = await sharePage(req(`/s/${TOKEN}`), ENV, TOKEN, api(null, 404));
		expect(res.status).toBe(404);
		const html = await res.text();
		expect(html).toContain("No flight record matches this link");
		expect(html).not.toContain("og:image");
	});

	it("refuses a malformed token before any fetch, and works without the API configured", async () => {
		const f = api(smoke);
		expect((await sharePage(req("/s/bad%20token"), ENV, "bad token", f)).status).toBe(404);
		expect((await badge(req("/badge/x"), ENV, "../../etc", f)).status).toBe(404);
		expect(f).not.toHaveBeenCalled();
		const res = await sharePage(req(`/s/${TOKEN}`), {}, TOKEN, f);
		expect(res.status).toBe(503);
		expect(res.headers.get("cache-control")).toBe("no-store");
		const html = await res.text();
		expect(html).toContain("Open this flight in Preflight");
		expect(html).not.toContain("og:image");
	});

	it("never caches an API outage, so link previews don't keep a generic card", async () => {
		for (const status of [500, 502, 503]) {
			const res = await badge(req(`/badge/${TOKEN}`), ENV, TOKEN, api(null, status));
			expect(res.status).toBe(502);
			expect(res.headers.get("cache-control")).toBe("no-store");
		}
	});

	it("keeps responses for two minutes only, since a running job revises its record", async () => {
		const res = await badge(req(`/badge/${TOKEN}`), ENV, TOKEN, api(golden));
		expect(res.headers.get("cache-control")).toBe("public, max-age=120");
		expect(res.headers.get("content-type")).toBe("image/svg+xml; charset=utf-8");
	});
});

describe("render", () => {
	const card = join(here, "..", "functions", "_card");
	const wasm = readFileSync(join(card, "vendor", "resvg", "index_bg.wasm"));
	const fonts = ["InstrumentSans-400", "InstrumentSans-600", "InstrumentSerif-400", "JetBrainsMono-500"].map((f) => new Uint8Array(readFileSync(join(card, "fonts", `${f}.bin`))));
	const renderer = { png: (svg: string) => renderPng(svg, wasm, fonts, CARD_W) };

	it("draws a 1200 x 630 PNG for real records", async () => {
		const out = process.env.CARD_OUT;
		if (out) mkdirSync(out, { recursive: true });
		for (const [name, rec] of [["smoke", smoke], ["golden", golden], ["agent", agent], ["grpo", grpo]] as const) {
			const res = await cardPng(new Request(`https://preflight.example/card/${TOKEN}`), ENV, TOKEN, renderer, api(rec));
			expect(res.headers.get("content-type")).toBe("image/png");
			const png = new Uint8Array(await res.arrayBuffer());
			expect([...png.slice(1, 4)].map((c) => String.fromCharCode(c)).join("")).toBe("PNG");
			const view = new DataView(png.buffer, png.byteOffset);
			expect([view.getUint32(16), view.getUint32(20)]).toEqual([CARD_W, CARD_H]);
			if (out) writeFileSync(join(out, `card-${name}.png`), png);
		}
	}, 30000);
});
