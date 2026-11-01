import { describe, expect, it, vi } from "vitest";

let on = false;
vi.mock("../lib/flags", () => ({ flag: (name: string) => name === "harness.spark_script_map" && on }));

import en from "../i18n/en.json";
import { dataUseLines } from "./DataUse";

describe("the data-use notice and Muse Spark reading the code", () => {
	it("says Muse Spark reads the code exactly when the runner's flag is on", () => {
		expect(en["datause.script_map"]).toBe("Muse Spark (Meta) reads your code to build the map.");
		on = false;
		expect(dataUseLines()).not.toContain(en["datause.script_map"]);
		on = true;
		expect(dataUseLines()).toContain(en["datause.script_map"]);
	});
});
