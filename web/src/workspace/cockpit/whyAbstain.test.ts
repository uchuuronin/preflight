import { describe, expect, it } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { changedArgs, localOffGridText } from "./whyAbstain";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);

describe("a local record's off-grid line (Chiya's shot 9b, Nov 3)", () => {
	it("names the setting and the command that measures it", () => {
		const knob = dev.knobs.find((k) => !k.needs_code && k.kind === "int")!;
		const v = Number(knob.current) * 2;
		expect(changedArgs(dev, { [knob.id]: v })).toEqual([`--${knob.id} ${v}`]);
		expect(localOffGridText(dev, { [knob.id]: v })).toBe(
			`Not on this record's grids. Check it in on the site to explore every setting, or measure it with preflight run … -- --${knob.id} ${v}.`,
		);
	});
	it("leaves code knobs out of the command, and says it plainly when nothing can be passed", () => {
		const code = dev.knobs.find((k) => k.needs_code);
		if (code) expect(changedArgs(dev, { [code.id]: !code.current })).toEqual([]);
		expect(localOffGridText(dev, {})).toMatch(/measure this setting with preflight run\.$/);
	});
});
