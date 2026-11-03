// @vitest-environment jsdom
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import vs from "../../coverage/vs_model.json";
import { Checks } from "./Checks";

afterEach(cleanup);

describe("the checks page's same-code comparison (B's vs_model.json)", () => {
	it("shows B's summary, a row per script with both sides' ratios, and B's caveats", () => {
		render(<Checks />);
		expect(screen.getByText(vs.summary)).toBeTruthy();
		const table = screen.getByRole("table", { name: /Same code, asked to Muse Spark without probes/ });
		const rows = within(table).getAllByRole("row").slice(1);
		expect(rows.length).toBe(vs.rows.length);
		const gnn = rows.find((r) => r.textContent?.includes("GNN (PyG GraphSAGE)"))!;
		expect(gnn.textContent).toContain("Preflight: 0.92×");
		expect(gnn.textContent).toContain("Muse Spark: 18×");
		// a run that ran out of memory says why it has no peak, not a gap
		const lora = rows.find((r) => r.textContent?.includes("LoRA fine-tune"))!;
		expect(lora.textContent).toContain("never seen");
		for (const c of vs.caveats) expect(screen.getByText(c)).toBeTruthy();
	});
});
