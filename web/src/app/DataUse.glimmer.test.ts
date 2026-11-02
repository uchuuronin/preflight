import { afterEach, describe, expect, it, vi } from "vitest";

const on = vi.hoisted(() => new Set<string>());
vi.mock("../lib/flags", () => ({ flag: (name: string) => on.has(name) }));

import en from "../i18n/en.json";
import { dataUseLines } from "./DataUse";

afterEach(() => on.clear());

describe("the data-use notice and private triage (harness.glimmer_private)", () => {
	it("says nothing about Glimmer while private triage isn't offered", () => {
		on.add("harness.spark_open_triage");
		expect(dataUseLines()).not.toContain(en["datause.glimmer"]);
		expect(dataUseLines()).not.toContain(en["datause.glimmer_map"]);
	});

	it("says where a private check-in's logs go, and that Muse Spark still builds the map when it does", () => {
		on.add("harness.spark_open_triage").add("harness.glimmer_private");
		expect(dataUseLines()).toContain(en["datause.glimmer"]);
		on.add("harness.spark_script_map");
		expect(dataUseLines()).toContain(en["datause.glimmer_map"]);
		expect(en["datause.glimmer_map"]).toMatch(/Muse Glimmer.*not to Baseten.*Muse Spark \(Meta\)/);
	});
});
