import { afterEach, describe, expect, it, vi } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { loadRecord } from "./load";

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
	new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

afterEach(() => {
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

describe("loadRecord", () => {
	it("retries once after a busy record store, waiting as told", async () => {
		vi.useFakeTimers();
		const fetch = vi
			.fn()
			.mockResolvedValueOnce(json(503, { error: { code: "record_store_busy", message: "Records are a moment behind; try again." } }, { "Retry-After": "2" }))
			.mockResolvedValueOnce(json(200, devRaw));
		vi.stubGlobal("fetch", fetch);
		const p = loadRecord("tok_abcdefghijklmnop");
		await vi.advanceTimersByTimeAsync(1_999);
		expect(fetch).toHaveBeenCalledTimes(1);
		await vi.advanceTimersByTimeAsync(1);
		const rec = await p;
		expect(fetch).toHaveBeenCalledTimes(2);
		expect(rec.knobs.length).toBeGreaterThan(0);
	});

	it("gives up after one retry and shows the API's words", async () => {
		vi.useFakeTimers();
		const busy = () => json(503, { error: { message: "Records are a moment behind; try again." } }, { "Retry-After": "60" });
		const fetch = vi.fn().mockResolvedValueOnce(busy()).mockResolvedValueOnce(busy());
		vi.stubGlobal("fetch", fetch);
		const p = loadRecord("tok_abcdefghijklmnop");
		const done = expect(p).rejects.toThrow("Records are a moment behind; try again.");
		await vi.advanceTimersByTimeAsync(5_000); // capped, not the 60 s asked for
		await done;
		expect(fetch).toHaveBeenCalledTimes(2);
	});

	it("doesn't retry a missing record", async () => {
		const fetch = vi.fn().mockResolvedValueOnce(json(404, { error: { message: "No flight record matches this link." } }));
		vi.stubGlobal("fetch", fetch);
		await expect(loadRecord("tok_abcdefghijklmnop")).rejects.toThrow("No flight record matches this link.");
		expect(fetch).toHaveBeenCalledTimes(1);
	});
});
