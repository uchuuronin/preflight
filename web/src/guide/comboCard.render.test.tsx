// @vitest-environment jsdom
// The card and the guide's words agree (live check, Nov 3): on PF102 (L4, eval batch 32, out of memory)
// the only single move that clears it is a GPU change, so the guide's sensitivity offers the
// combination on the L4 and says so; the card it names must be there to apply it.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../lib/flags", () => ({ flag: (name: string) => name === "c.guide.live" }));

import en from "../i18n/en.json";
import liveRaw from "../workspace/fixtures/lora-live-nov1.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../workspace/record/types";
import { WorkspaceProvider } from "../workspace/state";
import { NO_LIMITS } from "./limits";
import { guideCombo } from "./liveGuide";
import { GuidePanel } from "./GuidePanel";

const qwen = normaliseRecord(liveRaw as unknown as FlightRecord);
afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});
const frame = () => act(() => new Promise<void>((r) => requestAnimationFrame(() => r())));
const knobs = () => Object.fromEntries(qwen.knobs.map((k) => [k.id, k.current]));

describe("the combination on the current GPU", () => {
	it("is offered without a GPU change on PF102", () => {
		const c = guideCombo(qwen, new GridProvider(), knobs(), "min_cost", NO_LIMITS, new Set());
		expect(c).not.toBeNull();
		expect(c!.changes.map((x) => x.knobId)).not.toContain("gpu_type");
	});

	it("has its card beside the guide's words", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: string) =>
				url.startsWith("/v1/guide/turns")
					? new Response(JSON.stringify({ text: "No single knob on this GPU clears the failure.", withheld: null, calls: [], awaiting: "visitor", state: "s", provenance: {} }), { status: 200 })
					: new Response(JSON.stringify({ limits: null, saved: true }), { status: 200 }),
			),
		);
		render(
			<WorkspaceProvider record={qwen} recordRef="lora-qwen" provider={new GridProvider()}>
				<GuidePanel />
			</WorkspaceProvider>,
		);
		await act(async () => {
			fireEvent.click(screen.getByRole("button", { name: en["guide.start"] }));
		});
		await frame();
		expect(screen.getAllByRole("button", { name: /^Apply all / }).length).toBe(1);
	});
});
