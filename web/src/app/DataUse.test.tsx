import { describe, expect, it } from "vitest";
import en from "../i18n/en.json";
import { dataUseLines, RETENTION } from "./DataUse";

describe("the data-use notice", () => {
	it("only states what this build does: no deletion promise until deletion exists", () => {
		const lines = dataUseLines();
		expect(RETENTION).toBeNull();
		expect(lines).toContain(en["datause.retention.unset"]);
		expect(lines.join(" ")).not.toMatch(/deleted \d/);
	});

	it("mentions Muse Spark reading code or messages only when that behaviour's flag is on", () => {
		const lines = dataUseLines();
		// flags.json ships both off
		expect(lines).not.toContain(en["datause.script_map"]);
		expect(lines).not.toContain(en["datause.guide"]);
	});
});
