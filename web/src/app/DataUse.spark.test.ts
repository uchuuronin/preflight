import { afterEach, describe, expect, it, vi } from "vitest";

const on = new Set<string>();
vi.mock("../lib/flags", () => ({ flag: (name: string) => on.has(name) }));

import en from "../i18n/en.json";
import { dataUseLines } from "./DataUse";

const lines = (...flags: string[]) => {
	on.clear();
	for (const f of flags) on.add(f);
	return dataUseLines();
};

describe("the data-use notice for Muse Spark's diagnosis, its claim check and its search (PM, Nov 2)", () => {
	afterEach(() => on.clear());
	it("says nothing about them while their flags are off", () => {
		const l = lines();
		for (const k of ["datause.diagnosis", "datause.claim_check", "datause.search", "datause.fix"] as const) expect(l).not.toContain(en[k]);
	});
	it("says what goes where for each, only under its own flag", () => {
		expect(lines("harness.spark_search")).toContain(en["datause.search"]);
		expect(lines("harness.spark_search")).not.toContain(en["datause.diagnosis"]);
		expect(lines("harness.spark_claim_check")).toContain(en["datause.claim_check"]);
		expect(lines("harness.spark_claim_check")).not.toContain(en["datause.diagnosis"]);
		expect(lines("c.fix_loop")).toContain(en["datause.fix"]);
	});
	it("names the claim check with the diagnosis, which runs it today", () => {
		const l = lines("harness.spark_diagnosis");
		expect(l).toContain(en["datause.diagnosis"]);
		expect(l).toContain(en["datause.claim_check"]);
		expect(en["datause.diagnosis"]).toMatch(/Muse Spark \(Meta\).*plots/);
		expect(en["datause.claim_check"]).toMatch(/Muse Spark Open.*Baseten.*numbers/);
		expect(en["datause.search"]).toMatch(/Muse Spark \(Meta\) searches the web for your error text/);
	});
});
