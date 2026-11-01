import { describe, expect, it } from "vitest";
import en from "../i18n/en.json";
import { dataUseLines, RETENTION } from "./DataUse";

describe("the data-use notice", () => {
	it("renders the retention lines for this build's RETENTION", () => {
		expect(RETENTION).toEqual({ repoDeletedAtProbeEnd: true, uploadDays: 30, recordDays: 30, probeDays: 30 });
		const lines = dataUseLines();
		expect(lines).toContain(en["datause.retention.repo"]);
		expect(lines).toContain("Uploaded files and notebooks are deleted about 30 days after the last job that used them, and never before its record.");
		expect(lines).not.toContain(en["datause.retention.uploads_kept"]);
		expect(lines).toContain("Flight records are deleted about 30 days after they are made.");
		expect(lines).toContain("Probe results are kept about 30 days for reuse, error messages included, which can quote your code.");
		expect(lines).not.toContain(en["datause.retention.unset"]);
	});

	it("says uploads and probe results are kept, and records unset, without a number of days", () => {
		const lines = dataUseLines({ repoDeletedAtProbeEnd: false, uploadDays: null, recordDays: null, probeDays: null });
		expect(lines).toContain(en["datause.retention.uploads_kept"]);
		expect(lines).toContain(en["datause.retention.probe_errors"]);
		expect(lines).toContain(en["datause.retention.unset"]);
		expect(lines).not.toContain(en["datause.retention.repo"]);
	});

	it("mentions Muse Spark reading code or messages only when that behaviour's flag is on", () => {
		const lines = dataUseLines();
		// flags.json ships both off
		expect(lines).not.toContain(en["datause.script_map"]);
		expect(lines).not.toContain(en["datause.guide"]);
		expect(lines).not.toContain(en["datause.fix"]);
	});
});
