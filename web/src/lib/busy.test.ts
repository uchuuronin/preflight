import { afterEach, describe, expect, it, vi } from "vitest";
import { BUSY_MS, installBusyFlag } from "./busy";

describe("the busy flag", () => {
	afterEach(() => vi.useRealTimers());

	it("is set by a key or pointer press and clears BUSY_MS after the last one", () => {
		vi.useFakeTimers();
		const root = document.createElement("div");
		const stop = installBusyFlag(root);
		window.dispatchEvent(new KeyboardEvent("keydown"));
		expect(root.dataset.busy).toBe("");
		vi.advanceTimersByTime(BUSY_MS - 100);
		window.dispatchEvent(new PointerEvent("pointerdown"));
		vi.advanceTimersByTime(BUSY_MS - 100);
		expect(root.dataset.busy).toBe("");
		vi.advanceTimersByTime(200);
		expect(root.dataset.busy).toBeUndefined();
		stop();
	});
});
