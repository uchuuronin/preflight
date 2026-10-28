import { describe, expect, it } from "vitest";
import { whenReady } from "./whenReady";

describe("whenReady", () => {
	it("runs at once when nothing is in the way", async () => {
		let runs = 0;
		await whenReady(() => {
			runs++;
		});
		expect(runs).toBe(1);
	});

	it("retries while the target is not initialised yet", async () => {
		let calls = 0;
		await whenReady(() => {
			calls++;
			if (calls < 3) throw new ReferenceError("Cannot access 'state' before initialization");
		});
		expect(calls).toBe(3);
	});

	it("does not retry other errors", async () => {
		let calls = 0;
		await expect(
			whenReady(() => {
				calls++;
				throw new TypeError("broken add-on");
			}),
		).rejects.toThrow("broken add-on");
		expect(calls).toBe(1);
	});

	it("gives up after the retry limit", async () => {
		let calls = 0;
		await expect(
			whenReady(
				() => {
					calls++;
					throw new ReferenceError("never ready");
				},
				5,
			),
		).rejects.toThrow("never ready");
		expect(calls).toBe(6);
	});
});
