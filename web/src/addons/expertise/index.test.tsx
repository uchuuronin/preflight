import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { t } from "../../lib/i18n";
import { useRegistry } from "../../registry";
import { DetailLevel } from "./DetailLevel";
import { GlossaryCard } from "./GlossaryCard";
import { FLAG } from "./index";
import { setTier } from "./store";

describe("expertise add-on registration", () => {
	afterEach(() => act(() => setTier("practitioner")));

	it("puts the detail level in the cockpit header behind its flag", () => {
		const { result } = renderHook(() => useRegistry());
		const panel = result.current.panels.find((p) => p.id === "expertise.detail-level");
		expect(panel).toMatchObject({ slot: "cockpit-head", flag: FLAG, component: DetailLevel });
		expect(t(panel!.titleKey)).toBe("Detail level");
		expect(result.current.appPanels.find((p) => p.id === "expertise.detail-level")).toBeUndefined();
	});

	it("adds the glossary card to every knob behind the same flag", () => {
		const { result } = renderHook(() => useRegistry());
		const deco = result.current.knobDecorations.find((d) => d.id === "expertise.glossary");
		expect(deco).toMatchObject({ flag: FLAG, component: GlossaryCard });
		expect(deco!.knobIds).toBeUndefined();
	});

	it("registers its view over the knob dock behind the same flag", () => {
		const { result } = renderHook(() => useRegistry());
		const view = result.current.knobViews.find((v) => v.id === "expertise.level");
		expect(view).toMatchObject({ flag: FLAG, moreKey: "addon.expertise.more" });
		expect(t("addon.expertise.more", { n: 4 })).toBe("Show 4 more settings");
	});

	it("redraws the dock when the level changes", () => {
		const { result } = renderHook(() => useRegistry());
		const before = result.current;
		act(() => setTier("new"));
		expect(result.current).not.toBe(before);
	});

	it("uses D's flag name", () => {
		expect(FLAG).toBe("d.expertise");
	});
});
