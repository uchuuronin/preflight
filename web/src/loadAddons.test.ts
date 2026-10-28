import { afterEach, describe, expect, it, vi } from "vitest";
import { loadAddons } from "./loadAddons";
import { registerAppPanel, unregister, useRegistry } from "./registry";

describe("loading add-ons", () => {
	afterEach(() => vi.restoreAllMocks());

	it("lets an add-on register while it loads (D's blank-page bug)", async () => {
		const failures = await loadAddons({
			"./addons/selfreg/index.ts": async () => {
				registerAppPanel({ id: "selfreg", slot: "nav-end", titleKey: "x", component: () => null });
			},
		});
		expect(failures).toEqual([]);
		unregister("selfreg");
	});

	it("skips an add-on that throws while loading and loads the rest", async () => {
		const err = vi.spyOn(console, "error").mockImplementation(() => {});
		let ran = false;
		const failures = await loadAddons({
			"./addons/broken/index.ts": async () => {
				throw new Error("load boom");
			},
			"./addons/fine/index.ts": async () => {
				ran = true;
			},
		});
		expect(ran).toBe(true);
		expect(failures.map((f) => f.path)).toEqual(["./addons/broken/index.ts"]);
		expect(err).toHaveBeenCalled();
	});

	it("doesn't wait forever for a slow add-on", async () => {
		const start = Date.now();
		await loadAddons({ "./addons/slow/index.ts": () => new Promise(() => {}) }, 50);
		expect(Date.now() - start).toBeLessThan(1000);
	});

	it("exports the registry hook unchanged", () => {
		expect(typeof useRegistry).toBe("function");
	});
});
