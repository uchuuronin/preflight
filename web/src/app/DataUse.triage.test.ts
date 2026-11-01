import { describe, expect, it, vi } from "vitest";

let on = false;
vi.mock("../lib/flags", () => ({ flag: (name: string) => name === "harness.spark_open_triage" && on }));

import en from "../i18n/en.json";
import { dataUseLines } from "./DataUse";

describe("the data-use notice and Spark Open reading probe logs", () => {
	it("says Spark Open reads the probe's log exactly when A's triage flag is on", () => {
		on = false;
		expect(dataUseLines()).not.toContain(en["datause.triage"]);
		on = true;
		expect(dataUseLines()).toContain(en["datause.triage"]);
		expect(en["datause.triage"]).toMatch(/Muse Spark Open/);
	});
});
