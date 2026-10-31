import { describe, expect, it } from "vitest";
import en from "../i18n/en.json";
import { dataUseLines, RETENTION } from "./DataUse";

describe("the data-use notice", () => {
	it("renders the retention lines for this build's RETENTION", () => {
		expect(RETENTION).toEqual({ repoDeletedAtProbeEnd: true, uploadDays: null, recordDays: 30 });
		const lines = dataUseLines();
		expect(lines).toContain(en["datause.retention.repo"]);
		expect(lines).toContain(en["datause.retention.uploads_kept"]);
		expect(lines).toContain("Flight records are deleted about 30 days after they are made.");
		expect(lines).toContain(en["datause.retention.probe_errors"]);
		expect(lines).not.toContain(en["datause.retention.unset"]);
	});

	it("says uploads are deleted only once a number of days exists, and records unset without one", () => {
		const lines = dataUseLines({ repoDeletedAtProbeEnd: false, uploadDays: 30, recordDays: null });
		expect(lines).toContain("Uploaded files are deleted about 30 days after they are sent.");
		expect(lines).toContain(en["datause.retention.unset"]);
		expect(lines).not.toContain(en["datause.retention.repo"]);
	});

	it("mentions Muse Spark reading code or messages only when that behaviour's flag is on", () => {
		const lines = dataUseLines();
		// flags.json ships both off
		expect(lines).not.toContain(en["datause.script_map"]);
		expect(lines).not.toContain(en["datause.guide"]);
	});
});
