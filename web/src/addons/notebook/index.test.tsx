import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { t } from "../../lib/i18n";
import { useRegistry } from "../../registry";
import { NotebookIntake } from "./NotebookIntake";
import { FLAG } from "./index";

describe("notebook intake registration", () => {
	it("adds a Notebook tab to the check-in desk, behind its flag", () => {
		const { result } = renderHook(() => useRegistry());
		const panel = result.current.appPanels.find((p) => p.id === "notebook.intake");
		expect(panel).toMatchObject({ slot: "intake", flag: FLAG, component: NotebookIntake });
		expect(t(panel!.titleKey)).toBe("Notebook");
		expect(FLAG).toBe("d.notebook");
	});
});
