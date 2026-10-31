import { describe, expect, it } from "vitest";
import en from "../i18n/en.json";
import { dataUseLines, RETENTION } from "./DataUse";

describe("the data-use notice", () => {
	it("states only the deletion A's runner does: repo code at probe end, records at 30 days", () => {
		const lines = dataUseLines();
		expect(RETENTION).toEqual({ repoDeletedAtProbeEnd: true, uploadDays: null, recordDays: 30 });
		expect(lines).toContain(en["datause.retention.repo"]);
		expect(lines).toContain("Flight records are deleted after 30 days.");
		expect(lines).not.toContain(en["datause.retention.unset"]);
	});

	it("never promises to delete uploaded files, which nothing deletes yet", () => {
		const lines = dataUseLines();
		expect(lines).toContain(en["datause.retention.uploads_kept"]);
		expect(lines.join(" ")).not.toMatch(/Uploaded files are deleted/);
	});

	it("mentions Muse Spark reading code or messages only when that behaviour's flag is on", () => {
		const lines = dataUseLines();
		// flags.json ships both off
		expect(lines).not.toContain(en["datause.script_map"]);
		expect(lines).not.toContain(en["datause.guide"]);
	});
});
