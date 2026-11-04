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
		expect(gnn.textContent).toContain("Preflight: 0.916×");
		expect(gnn.textContent).toContain("Muse Spark: 18.0×");
		// a run that ran out of memory says why it has no peak, not a gap
		const lora = rows.find((r) => r.textContent?.includes("LoRA fine-tune"))!;
		expect(lora.textContent).toContain("ran out of memory at 21.8 GiB reserved");
		for (const c of vs.caveats) expect(screen.getByText(c)).toBeTruthy();
	});

	it("reads first as three tiles and two miss charts, each miss said in words, never by colour alone", () => {
		render(<Checks />);
		const tiles = document.querySelectorAll(".pf-ask__tile");
		expect(tiles.length).toBe(3);
		expect(tiles[0].textContent).toContain("8% to 36%");
		expect(tiles[0].textContent).toContain("1.6× to 18×");
		expect(tiles[1].textContent).toContain("2 of 4");
		expect(tiles[1].textContent).toContain("0 of 4");
		expect(tiles[2].textContent).toContain("1 caught, 1 no answer");
		expect(tiles[2].textContent).toContain("1 caught, 1 missed");
		// every plotted ratio is a lane whose words say how far off and which way, beside a named shape
		const lanes = [...document.querySelectorAll(".pf-miss__lane:not(.is-empty)")];
		const plotted = vs.rows.flatMap((r) => [r.ratio.preflight_step, r.ratio.muse_step, r.ratio.preflight_peak, r.ratio.muse_peak]).filter((x) => x !== null).length;
		expect(lanes.length).toBe(plotted);
		expect(lanes.some((l) => l.textContent === "Muse Spark: 18× too slow" && /full run: 0\.0612 s/.test(l.getAttribute("title") ?? ""))).toBe(true);
		expect(lanes.some((l) => l.textContent === "Preflight: 8% too fast" && l.classList.contains("is-on"))).toBe(true);
		// what isn't plotted is said in place, in B's words
		expect(screen.getByText(/Retrieval \(BEIR\): ran out of memory at 21\.5 GiB reserved, so its full peak was never seen/)).toBeTruthy();
		expect(screen.getByText("no probe ran: no answer, not a miss")).toBeTruthy();
	});
});
