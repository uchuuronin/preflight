import { describe, expect, it, vi } from "vitest";
import { isStaleChunkError, reloadOnceForStaleChunk } from "./staleChunk";

const memory = () => {
	const m = new Map<string, string>();
	return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) };
};

describe("a page from before a deploy", () => {
	it("knows a missing chunk when it sees one", () => {
		expect(isStaleChunkError(new TypeError("Failed to fetch dynamically imported module: https://x/assets/pretrain-CRuA7Cir.js"))).toBe(true);
		expect(isStaleChunkError(new Error("No flight record matches this link"))).toBe(false);
	});
	it("reloads once, not in a loop, and again after a while", () => {
		const store = memory();
		const reload = vi.fn();
		expect(reloadOnceForStaleChunk(1_000, store, reload)).toBe(true);
		expect(reloadOnceForStaleChunk(2_000, store, reload)).toBe(false);
		expect(reloadOnceForStaleChunk(120_000, store, reload)).toBe(true);
		expect(reload).toHaveBeenCalledTimes(2);
	});
});
