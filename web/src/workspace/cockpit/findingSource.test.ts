import { describe, expect, it } from "vitest";
import { t } from "../../lib/i18n";
import { findingSourceKey } from "./findingSource";

describe("who found a finding", () => {
	it("names Muse Glimmer for a private check-in's triage, as the live record titles it (Nov 2)", () => {
		const f = { source: "spark_open_triage" as const, title: "Muse Glimmer on probe p1-eval: oom" };
		expect(t(findingSourceKey(f))).toBe("Found by Muse Glimmer");
	});
	it("keeps Spark Open's line for its own triage, and every other source as it was", () => {
		expect(t(findingSourceKey({ source: "spark_open_triage", title: "Spark Open on probe p3: oom" }))).toBe("Found by Muse Spark Open triage");
		expect(t(findingSourceKey({ source: "rule", title: "Muse Glimmer on probe p1: oom" }))).toBe("Found by a rule check");
	});
});
