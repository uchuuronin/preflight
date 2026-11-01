// @vitest-environment jsdom
import { act, cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { OrbitGlance } from "../map/OrbitGlance";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { useWorkspace, WorkspaceProvider } from "../state";
import { FitMap } from "./FitMap";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);

function Knobs() {
	const { state } = useWorkspace();
	return <output data-testid="knobs">{JSON.stringify(state.knobs)}</output>;
}

const open = (ui: React.ReactNode) =>
	render(
		<WorkspaceProvider record={dev} provider={new GridProvider()}>
			{ui}
			<Knobs />
		</WorkspaceProvider>,
	);

afterEach(cleanup);

describe("the fit map moves the run, it isn't a table to read", () => {
	it("fills each square by its share of the GPU's memory, and marks the run's square as the puck", () => {
		const { container } = open(<FitMap />);
		const cells = [...container.querySelectorAll<HTMLElement>(".pf-fitcell[data-xi]")];
		expect(cells.length).toBeGreaterThan(4);
		const fills = cells.map((c) => Number(c.style.getPropertyValue("--fill")));
		expect(fills.every((f) => f >= 0 && f <= 1)).toBe(true);
		expect(new Set(fills).size).toBeGreaterThan(2);
		// A square past the GPU is drawn full, with the overflow mark.
		for (const c of cells.filter((x) => x.classList.contains("is-over"))) expect(c.style.getPropertyValue("--fill")).toBe("1");
		expect(container.querySelectorAll(".pf-fitcell.is-here")).toHaveLength(1);
		expect(container.querySelector(".pf-fitmap__puck")).toBeTruthy();
	});

	it("previews a square in the readout on focus without moving any knob, and flies there on press", () => {
		const { container, getByTestId } = open(<FitMap />);
		const before = getByTestId("knobs").textContent;
		const other = container.querySelector<HTMLElement>(".pf-fitcell[data-xi]:not(.is-here)")!;
		act(() => other.focus());
		const readout = container.querySelector(".pf-fit__readout")!.textContent!;
		expect(readout).toContain("Press to fly there");
		expect(readout).toMatch(/of the GPU/);
		expect(getByTestId("knobs").textContent).toBe(before);
		fireEvent.click(other);
		expect(getByTestId("knobs").textContent).not.toBe(before);
	});

	it("drags the run's square across the map, and only the pointer that picked it up moves it", () => {
		const { container, getByTestId } = open(<FitMap />);
		const here = container.querySelector<HTMLElement>(".pf-fitcell.is-here")!;
		const target = container.querySelector<HTMLElement>(".pf-fitcell[data-xi]:not(.is-here)")!;
		const frame = container.querySelector<HTMLElement>(".pf-fitmap__frame")!;
		const before = getByTestId("knobs").textContent;
		// jsdom has no layout: the square under the pointer is the target.
		const efp = document.elementFromPoint;
		document.elementFromPoint = () => target;
		try {
			fireEvent.pointerDown(here, { pointerId: 1, button: 0 });
			fireEvent.pointerMove(frame, { pointerId: 2, clientX: 5, clientY: 5 });
			expect(getByTestId("knobs").textContent).toBe(before);
			fireEvent.pointerMove(frame, { pointerId: 1, clientX: 5, clientY: 5 });
			fireEvent.pointerUp(frame, { pointerId: 1 });
		} finally {
			document.elementFromPoint = efp;
		}
		const k = JSON.parse(getByTestId("knobs").textContent!);
		const dev0 = JSON.parse(before!);
		expect(k).not.toEqual(dev0);
	});

	it("moves focus between squares with the arrow keys", () => {
		const { container } = open(<FitMap />);
		const first = container.querySelector<HTMLElement>('.pf-fitcell[data-xi="0"]')!;
		act(() => first.focus());
		fireEvent.keyDown(first, { key: "ArrowRight" });
		expect((document.activeElement as HTMLElement).dataset.xi).toBe("1");
	});

	it("switches the GPU from the map itself", () => {
		const { container, getByTestId } = open(<FitMap />);
		const sel = container.querySelector<HTMLSelectElement>(".pf-fit__gpu select");
		if (!sel) return; // a record without a GPU knob has nothing to switch
		const other = [...sel.options].map((o) => o.value).find((v) => v !== sel.value)!;
		fireEvent.change(sel, { target: { value: other } });
		expect(JSON.parse(getByTestId("knobs").textContent!).gpu_type).toBe(other);
	});
});

describe("the orbit at a glance", () => {
	it("draws the whole loop simplified: the peak with its mark and the three biggest phases, no nodes", () => {
		const { container } = open(<OrbitGlance />);
		expect(container.querySelector(".pf-onode")).toBeNull();
		expect(container.querySelector(".pf-glance__num")!.textContent).toMatch(/GiB/);
		expect(container.querySelector(".pf-glance .pf-status")).toBeTruthy();
		const phases = container.querySelectorAll(".pf-glance__phase");
		expect(phases.length).toBeGreaterThan(0);
		expect(phases.length).toBeLessThanOrEqual(3);
	});
});
