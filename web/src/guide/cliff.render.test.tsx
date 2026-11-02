// @vitest-environment jsdom
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../lib/flags", () => ({ flag: () => false }));

import en from "../i18n/en.json";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import liveRaw from "../workspace/fixtures/lora-live-nov1.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { capacityBytes, targetFor } from "../workspace/record/model";
import { type FlightRecord, type LoadedRecord, normaliseRecord } from "../workspace/record/types";
import { fitOf, trusted } from "../workspace/status";
import { useWorkspace, type Workspace, WorkspaceProvider } from "../workspace/state";
import { fmt } from "../lib/fmt";
import { GuidePanel } from "./GuidePanel";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const live = normaliseRecord(liveRaw as unknown as FlightRecord);
afterEach(cleanup);

function mount(record: LoadedRecord = dev) {
	const ref: { ws: Workspace | null } = { ws: null };
	function Probe() {
		ref.ws = useWorkspace();
		return null;
	}
	render(
		<WorkspaceProvider record={record} recordRef="test" provider={new GridProvider()}>
			<Probe />
			<GuidePanel />
		</WorkspaceProvider>,
	);
	return ref;
}

const cliffShown = () => document.querySelector(".pf-guide__cliff");

describe("the memory-limit nudge", () => {
	it("speaks up when the visitor moves a knob over the limit, with a real peak", () => {
		const r = mount();
		act(() => r.ws!.actions.setKnob("eval_batch_size", 4, "user"));
		expect(fitOf(dev, r.ws!.estimates, r.ws!.state.knobs)).not.toBe("fails");
		act(() => r.ws!.actions.setKnob("eval_batch_size", 32, "user"));
		expect(fitOf(dev, r.ws!.estimates, r.ws!.state.knobs)).toBe("fails");
		expect(cliffShown()).toBeTruthy();
		expect(cliffShown()!.textContent).not.toContain(en["value.none"]);
		expect(screen.getByRole("button", { name: /^Back to 4/ })).toBeTruthy();
	});

	it("stays quiet when a suggestion is put back (Not this, Keep, a guide revert)", () => {
		const r = mount();
		act(() => r.ws!.actions.setKnob("eval_batch_size", 4, "user"));
		act(() => r.ws!.actions.setKnob("eval_batch_size", 32, "user", { putBack: true }));
		expect(fitOf(dev, r.ws!.estimates, r.ws!.state.knobs)).toBe("fails");
		expect(cliffShown()).toBeNull();
	});

	it("still speaks up for a pinned knob the visitor moves themselves", () => {
		const r = mount();
		act(() => r.ws!.actions.setKnob("eval_batch_size", 4, "user"));
		act(() => r.ws!.actions.pin("eval_batch_size", true));
		act(() => r.ws!.actions.setKnob("eval_batch_size", 32, "user"));
		expect(cliffShown()).toBeTruthy();
	});

	it("without a trusted peak, names the GPU's memory alone (no 'No value')", () => {
		const r = mount(live);
		// On this record only the largest GPUs clear the eval at its settings (B300: tight).
		const big = "B300-SXM-288GB";
		act(() => r.ws!.actions.setKnob("gpu_type", big, "user"));
		expect(fitOf(live, r.ws!.estimates, r.ws!.state.knobs)).not.toBe("fails");
		act(() => r.ws!.actions.setKnob("gpu_type", "L4", "user"));
		const memT = targetFor(live, "memory")!;
		expect(fitOf(live, r.ws!.estimates, r.ws!.state.knobs)).toBe("fails");
		expect(trusted(r.ws!.estimates[memT.id])).toBe(false);
		const text = cliffShown()!.textContent ?? "";
		expect(text).not.toContain(en["value.none"]);
		expect(text).not.toContain("{");
		expect(text).toContain(fmt(capacityBytes(live, r.ws!.state.knobs)!, "bytes"));
		expect(text).toContain("likely to run out of memory");
	});
});
