import { renderHook } from "@testing-library/react";
import { beforeAll, describe, expect, it } from "vitest";
import { t } from "../../lib/i18n";
import { useRegistry } from "../../registry";
import { FreeTierPanel } from "./FreeTierPanel";
import { FLAG, LENSES, ready } from "./index";

describe("free-tier add-on registration", () => {
	beforeAll(() => ready);

	it("adds a band under the cockpit's whole run, behind its flag", () => {
		const { result } = renderHook(() => useRegistry());
		const panel = result.current.panels.find((p) => p.id === "freetier.sessions");
		expect(panel).toMatchObject({ slot: "below", flag: FLAG, component: FreeTierPanel, lenses: LENSES });
		expect(t(panel!.titleKey)).toBe("Free GPUs");
		expect(FLAG).toBe("d.freetier");
	});

	it("shows only for jobs that run on a GPU session", () => {
		expect(LENSES).toContain("finetune");
		expect(LENSES).not.toContain("agent");
		expect(LENSES).not.toContain("serving");
	});
});
