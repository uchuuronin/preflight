import { describe, expect, it, vi } from "vitest";

vi.mock("../lib/flags", () => ({ flag: (name: string) => name === "c.fix_loop" }));

import en from "../i18n/en.json";
import { dataUseLines } from "./DataUse";

describe("the data-use notice with the fix loop on", () => {
	it("says who writes the fixes", () => {
		expect(en["datause.fix"]).toBe("Fixes are written by Meta’s Muse Code from your script.");
		expect(dataUseLines()).toContain(en["datause.fix"]);
	});
});
