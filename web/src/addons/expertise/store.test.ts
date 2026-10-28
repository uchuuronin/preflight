import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_TIER, STORAGE_KEY, getTier, parseTier, resetTierForTests, setTier, subscribeTier } from "./store";

describe("detail level store", () => {
	beforeEach(() => {
		localStorage.clear();
		resetTierForTests();
	});
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("starts at practitioner when nothing is saved", () => {
		expect(DEFAULT_TIER).toBe("practitioner");
		expect(getTier()).toBe("practitioner");
	});

	it("remembers the chosen level in this browser", () => {
		setTier("new");
		expect(getTier()).toBe("new");
		expect(localStorage.getItem(STORAGE_KEY)).toBe("new");
		resetTierForTests();
		expect(getTier()).toBe("new");
	});

	it("ignores a saved value that is not a level", () => {
		localStorage.setItem(STORAGE_KEY, "wizard");
		resetTierForTests();
		expect(getTier()).toBe("practitioner");
	});

	it("parses only the three levels", () => {
		expect(parseTier("expert")).toBe("expert");
		expect(parseTier("EXPERT")).toBeNull();
		expect(parseTier(null)).toBeNull();
	});

	it("tells listeners when the level changes, and only then", () => {
		const fn = vi.fn();
		const off = subscribeTier(fn);
		setTier("expert");
		setTier("expert");
		expect(fn).toHaveBeenCalledTimes(1);
		off();
		setTier("new");
		expect(fn).toHaveBeenCalledTimes(1);
	});

	it("keeps working when storage throws", () => {
		vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
			throw new Error("blocked");
		});
		vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
			throw new Error("blocked");
		});
		resetTierForTests();
		expect(getTier()).toBe("practitioner");
		setTier("new");
		expect(getTier()).toBe("new");
	});
});
