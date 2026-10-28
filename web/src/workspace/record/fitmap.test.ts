import { describe, expect, it } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { GridProvider } from "../provider/GridProvider";
import { fitOf } from "../status";
import { defaultAxes, fitMap, MAX_SIDE } from "./fitmap";
import { capacityBytes, currentKnobs } from "./model";
import { pOomFromMemory } from "./oom";
import { FAILS_AT, FITS_BELOW } from "./terms";
import { type FlightRecord, normaliseRecord } from "./types";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const provider = new GridProvider();
const knobs = currentKnobs(dev);

describe("the fit map", () => {
	it("picks the two knobs that move memory most", () => {
		expect(defaultAxes(dev, provider, knobs)).toEqual(["seq_len", "eval_batch_size"]);
	});

	it("marks every square with the status the dials show there, and the visitor's own square", () => {
		const m = fitMap(dev, provider, knobs, "seq_len", "eval_batch_size");
		if (!m) throw new Error("no map");
		expect(m.xs.length).toBeLessThanOrEqual(MAX_SIDE);
		let here = 0;
		for (const row of m.rows) {
			for (const c of row) {
				const k = { ...knobs, seq_len: c.x, eval_batch_size: c.y };
				const p = provider.predict(dev, k);
				const want = p["memory.peak"]?.status === "abstain" && p["risk.oom"]?.status === "abstain" ? "abstain" : fitOf(dev, p, k);
				expect(c.status).toBe(want);
				if (c.here) here++;
			}
		}
		expect(here).toBe(1);
	});

	it("places each boundary between the squares it separates, where memory-derived P(OOM) crosses the cut", () => {
		const m = fitMap(dev, provider, knobs, "seq_len", "eval_batch_size");
		if (!m) throw new Error("no map");
		m.frontier.fits.forEach((c, yi) => {
			if (c === null) return;
			const i = Math.floor(c);
			const row = m.rows[yi]!;
			expect(row[i]!.status).toBe("fits");
			expect(row[i + 1]!.status).not.toBe("fits");
			const f = c - i;
			const x0 = row[i]!.x as number;
			const x1 = row[i + 1]!.x as number;
			const v = Math.exp(Math.log(x0) + f * (Math.log(x1) - Math.log(x0)));
			const k = { ...knobs, seq_len: v, eval_batch_size: m.ys[yi]! };
			const p = pOomFromMemory(provider.predict(dev, k)["memory.peak"], capacityBytes(dev, k));
			if (f !== 0.5) expect(p).toBeCloseTo(FITS_BELOW, 3);
		});
		// "fails" starts at or after "fits" ends on every row.
		m.frontier.fails.forEach((c, yi) => {
			const fits = m.frontier.fits[yi];
			if (c !== null && fits !== null && fits !== undefined) expect(c).toBeGreaterThanOrEqual(fits);
		});
		expect(FAILS_AT).toBeGreaterThan(FITS_BELOW);
	});
});
