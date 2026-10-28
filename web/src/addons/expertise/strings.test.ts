import { describe, expect, it } from "vitest";
import en from "../../i18n/en.json";
import { EXPERTISE_EN } from "./strings";

type RecordKnobs = { knobs?: Array<{ id: string }> };

const RECORDS = {
	...import.meta.glob<RecordKnobs>("../../../../schema/examples/*.json", { eager: true, import: "default" }),
	...import.meta.glob<RecordKnobs>("../../workspace/fixtures/*.record.json", { eager: true, import: "default" }),
};
const NOT_KNOBS = new Set(["proposed", "pinned", "pinned_hint", "unpin_so_it_moves", "pin", "needs_code", "not_trusted_here"]);
const LABELLED = Object.keys(en)
	.map((k) => /^knob\.([a-z0-9_]+)$/.exec(k)?.[1])
	.filter((id): id is string => !!id && !NOT_KNOBS.has(id));
const KNOB_IDS = [...new Set([...Object.values(RECORDS).flatMap((r) => (r.knobs ?? []).map((k) => k.id)), ...LABELLED])];

describe("expertise strings", () => {
	it("gives every record knob a plain label for the new level", () => {
		const missing = KNOB_IDS.filter((id) => !EXPERTISE_EN[`addon.expertise.label.${id}.new`]);
		expect(missing).toEqual([]);
	});

	it("gives every record knob its code name for the expert level", () => {
		for (const id of KNOB_IDS) expect(EXPERTISE_EN[`addon.expertise.label.${id}.expert`]).toBeTruthy();
	});

	it("only adds keys under addon.expertise, never overriding C's English", () => {
		const clash = Object.keys(EXPERTISE_EN).filter((k) => !k.startsWith("addon.expertise.") || k in en);
		expect(clash).toEqual([]);
	});

	it("uses no em dashes", () => {
		const bad = Object.entries(EXPERTISE_EN).filter(([, v]) => v.includes("\u2014"));
		expect(bad).toEqual([]);
	});
});
