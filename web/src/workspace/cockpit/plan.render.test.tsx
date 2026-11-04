import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import agentRaw from "../fixtures/agent-eval-oct31.record.json";
import grpoRaw from "../fixtures/grpo-l4.record.json";
import loraRaw from "../fixtures/lora-live-nov3.record.json";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { FlightPlanBand } from "./Bands";

const head = (raw: unknown, ref: string) => {
	const { container } = render(
		<WorkspaceProvider record={normaliseRecord(raw as FlightRecord)} recordRef={ref} provider={new GridProvider()}>
			<FlightPlanBand />
		</WorkspaceProvider>,
	);
	return container.querySelector(".pf-band__head")?.textContent ?? "";
};

afterEach(() => {
	cleanup();
	localStorage.clear();
});

describe("the flight plan's head (PM's audit, Nov 3)", () => {
	it("prices a run that stops up to the crash, with the full run beside it", () => {
		const text = head(loraRaw, "lora-qwen");
		expect(text).toMatch(/Stops at the first eval, step 200/);
		expect(text).toMatch(/About US\$[\d.]+ to US\$[\d.]+ over .+ until it stops \(the full run: US\$[\d.]+ to US\$[\d.]+\)/);
	});
	it("says what the script does say when the steps can't be counted", () => {
		expect(head(grpoRaw, "grpo")).toMatch(/Runs 1 epoch, but the script doesn't say how many examples it has/);
		expect(head(agentRaw, "agent-eval")).toMatch(/Runs 500 tasks, not training steps/);
	});
});
