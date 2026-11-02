// @vitest-environment jsdom
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../lib/flags", () => ({ flag: () => false }));

import en from "../i18n/en.json";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../workspace/record/types";
import { fitOf } from "../workspace/status";
import { useWorkspace, type Workspace, WorkspaceProvider } from "../workspace/state";
import { GuidePanel } from "./GuidePanel";

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

	it("stays quiet when the knob was pinned there on purpose (Keep on a guide move)", () => {
		const r = mount();
		act(() => r.ws!.actions.setKnob("eval_batch_size", 4, "user"));
		act(() => {
			r.ws!.actions.setKnob("eval_batch_size", 32, "user");
			r.ws!.actions.pin("eval_batch_size", true);
		});
		expect(fitOf(dev, r.ws!.estimates, r.ws!.state.knobs)).toBe("fails");
		expect(cliffShown()).toBeNull();
	});
});
