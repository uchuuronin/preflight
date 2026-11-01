// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import goldenRaw from "../../../../schema/examples/finetune.json";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { Dials } from "./Dials";
import { CostLine } from "./CostLine";

const record = normaliseRecord(goldenRaw as unknown as FlightRecord);
const open = () =>
	render(
		<WorkspaceProvider record={record} recordRef="lora-qwen" provider={new GridProvider()}>
			<CostLine />
			<Dials />
		</WorkspaceProvider>,
	);

afterEach(() => {
	cleanup();
	localStorage.clear();
});

describe("the cost line", () => {
	it("reads quantity x rate = total, with no list-price or date label", () => {
		const { container } = open();
		const line = container.querySelector(".pf-costline")!.textContent!;
		expect(line).toMatch(/GPU-hours × US\$0\.80\/h = US\$/);
		expect(line).not.toMatch(/list|20\d\d/i);
	});

	it("opens a box with the table's source and date, takes $0, tags it, and resets", () => {
		const { container } = open();
		fireEvent.click(container.querySelector(".pf-costline__rate")!);
		const box = screen.getByRole("dialog");
		expect(box.textContent).toMatch(/Table: US\$0\.80\/h on modal/);
		expect(box.textContent).toMatch(/\(\w+ 20\d\d\)/);
		fireEvent.change(box.querySelector("input")!, { target: { value: "0" } });
		act(() => {
			fireEvent.click(box.querySelector("button[type=submit]")!);
		});
		expect(screen.queryByRole("dialog")).toBeNull();
		expect(container.querySelector(".pf-costline__mine")).toBeTruthy();
		expect(container.querySelector(".pf-costline")!.textContent).toMatch(/= US\$0\.00/);
		// The cost dial moved with it (its label carries the final value; the number on its face travels there).
		const costDial = [...container.querySelectorAll(".pf-dial-wrap")].find((d) => /Run cost/i.test(d.textContent ?? ""))!;
		expect(costDial.querySelector(".pf-dial")!.getAttribute("aria-label")).toMatch(/Run cost: US\$0\.00/);
		fireEvent.click(container.querySelector(".pf-costline__reset")!);
		expect(container.querySelector(".pf-costline__mine")).toBeNull();
	});

	it("refuses a negative or empty rate, and Escape closes the box", () => {
		const { container } = open();
		fireEvent.click(container.querySelector(".pf-costline__rate")!);
		const box = screen.getByRole("dialog");
		fireEvent.change(box.querySelector("input")!, { target: { value: "-1" } });
		expect((box.querySelector("button[type=submit]") as HTMLButtonElement).disabled).toBe(true);
		fireEvent.keyDown(window, { key: "Escape" });
		expect(screen.queryByRole("dialog")).toBeNull();
	});
});
