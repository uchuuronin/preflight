// @vitest-environment jsdom
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import devRaw from "./fixtures/dev-finetune.record.json";
import { GridProvider } from "./provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "./record/types";
import { useWorkspace, type Workspace, WorkspaceProvider } from "./state";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
afterEach(cleanup);

function mount() {
	const ref: { ws: Workspace | null } = { ws: null };
	function Probe() {
		ref.ws = useWorkspace();
		return null;
	}
	render(
		<WorkspaceProvider record={dev} recordRef="lora-7b" provider={new GridProvider()}>
			<Probe />
		</WorkspaceProvider>,
	);
	return ref;
}

describe("the visitor's own knob move always wins", () => {
	it("sets a pending suggestion aside instead of locking the knob", () => {
		const r = mount();
		act(() => {
			r.ws!.actions.setKnob("batch_size", 8, "guide");
			r.ws!.actions.propose({ knobId: "batch_size", from: 4, to: 8 });
		});
		act(() => {
			r.ws!.actions.setKnob("batch_size", 2, "user");
		});
		expect(r.ws!.state.knobs.batch_size).toBe(2);
		expect(r.ws!.state.proposal).toBeNull();
		expect(r.ws!.state.setAside).toEqual({ knobId: "batch_size", from: 4, to: 8 });
	});

	it("puts the guide's previewed move back when the visitor moves a different knob", () => {
		const r = mount();
		act(() => {
			r.ws!.actions.setKnob("batch_size", 8, "guide");
			r.ws!.actions.propose({ knobId: "batch_size", from: 4, to: 8 });
		});
		act(() => {
			r.ws!.actions.setKnob("eval_batch_size", 8, "user");
		});
		expect(r.ws!.state.knobs.eval_batch_size).toBe(8);
		expect(r.ws!.state.knobs.batch_size).toBe(4);
		expect(r.ws!.state.proposal).toBeNull();
	});

	it("'Not this' rules the value out and sets the knob aside for the tours", () => {
		const r = mount();
		act(() => r.ws!.actions.decline("batch_size", 8));
		act(() => r.ws!.actions.decline("batch_size", 8));
		expect(r.ws!.state.limits.excluded.batch_size).toEqual([8]);
		expect(r.ws!.state.limits.declined).toEqual(["batch_size"]);
	});
});
