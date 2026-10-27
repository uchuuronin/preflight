import { renderHook } from "@testing-library/react";
import { beforeAll, describe, expect, it } from "vitest";
import { t } from "../../lib/i18n";
import { useRegistry } from "../../registry";
import { DetailLevel } from "./DetailLevel";
import { GlossaryCard } from "./GlossaryCard";
import { FLAG, ready } from "./index";

describe("expertise add-on registration", () => {
	beforeAll(() => ready);

	it("docks the detail level in the Logbook behind its flag", () => {
		const { result } = renderHook(() => useRegistry());
		const panel = result.current.appPanels.find((p) => p.id === "expertise.detail-level");
		expect(panel).toMatchObject({ slot: "logbook-settings", flag: FLAG, component: DetailLevel });
		expect(t(panel!.titleKey)).toBe("Detail level");
	});

	it("adds the glossary card to every knob behind the same flag", () => {
		const { result } = renderHook(() => useRegistry());
		const deco = result.current.knobDecorations.find((d) => d.id === "expertise.glossary");
		expect(deco).toMatchObject({ flag: FLAG, component: GlossaryCard });
		expect(deco!.knobIds).toBeUndefined();
	});

	it("uses D's flag name", () => {
		expect(FLAG).toBe("d.expertise");
	});
});
