import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it } from "vitest";
import loraRaw from "../fixtures/lora-live-nov3.record.json";
import { useOrbitLayout } from "../map/Orbit";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { useWorkspace, WorkspaceProvider } from "../state";
import { DetailSheet } from "./DetailSheet";

const record = normaliseRecord(loraRaw as unknown as FlightRecord);

function OpenEval() {
	const { actions } = useWorkspace();
	const layout = useOrbitLayout();
	const id = layout.nodes.find((n) => n.graphId && record.findings.filter((f) => f.node_id === n.graphId).length > 1)?.id;
	useEffect(() => {
		if (id) actions.focus(id);
	}, [id, actions]);
	return null;
}

afterEach(() => {
	cleanup();
	localStorage.clear();
});

describe("a node's sheet (Chiya, Nov 3)", () => {
	it("shows one finding at a time, worst first, paged by numbered marks, with close in the header", async () => {
		await act(async () => {
			render(
				<WorkspaceProvider record={record} recordRef="lora-qwen" provider={new GridProvider()}>
					<OpenEval />
					<DetailSheet />
				</WorkspaceProvider>,
			);
		});
		const tabs = screen.getAllByRole("tab");
		expect(tabs.length).toBeGreaterThan(1);
		expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
		expect(tabs[0]!.getAttribute("aria-selected")).toBe("true");
		expect(screen.getByText(/^Finding 1 of \d$/)).toBeTruthy();
		fireEvent.keyDown(tabs[0]!, { key: "ArrowRight" });
		expect(screen.getAllByRole("tab")[1]!.getAttribute("aria-selected")).toBe("true");
		expect(screen.getByRole("tabpanel").textContent).toContain(tabs[1]!.getAttribute("aria-label")!);
		expect(screen.getByRole("button", { name: "Back to the whole job" })).toBeTruthy();
	});
});
