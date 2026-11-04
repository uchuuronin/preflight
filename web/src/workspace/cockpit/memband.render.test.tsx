import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import campaignRaw from "../fixtures/campaign-grid.record.json";
import loraRaw from "../fixtures/lora-live-nov3.record.json";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { MemoryBand } from "./Bands";

const band = (raw: unknown, ref: string) =>
	render(
		<WorkspaceProvider record={normaliseRecord(raw as FlightRecord)} recordRef={ref} provider={new GridProvider()}>
			<MemoryBand />
		</WorkspaceProvider>,
	).container.textContent ?? "";

afterEach(() => {
	cleanup();
	localStorage.clear();
});

describe("the memory band's eval line (PM, Nov 3)", () => {
	it("says why a campaign's peak is far past the GPU when the eval spike leads it", () => {
		expect(band(campaignRaw, "campaign-grid")).toContain("The eval at batch 32 holds every logit; the grid never ran its eval, so this is extrapolated.");
	});
	it("stays off a single run's record", () => {
		expect(band(loraRaw, "lora-qwen")).not.toContain("the grid never ran its eval");
	});
});
