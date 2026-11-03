// @vitest-environment jsdom
// The runner's own word that nothing was measured (contract 0.6.6, A's `nothing_measured` on the done event).
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../lib/flags", () => ({ flag: () => false }));
vi.mock("../route", async (orig) => ({ ...(await orig<object>()), go: vi.fn() }));

import en from "../../i18n/en.json";
import { Boarding } from "./Boarding";

class FakeSource {
	static last: FakeSource | null = null;
	onmessage: ((e: { data: string }) => void) | null = null;
	onerror: (() => void) | null = null;
	onopen: (() => void) | null = null;
	constructor() {
		FakeSource.last = this;
	}
	addEventListener() {}
	close() {}
}

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("Boarding, a job that measured nothing", () => {
	it("says so from the done event, with the stop's own words", async () => {
		vi.stubGlobal("EventSource", FakeSource);
		vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 200 })));
		render(<Boarding jobId="01JOBTEST" />);
		await act(async () => {
			FakeSource.last!.onmessage?.({ data: JSON.stringify({ job_id: "01JOBTEST", seq: 9, at: "2026-11-03T21:00:00Z", stage: "done", record_id: "r", nothing_measured: true, error_code: "gpu.unavailable" }) });
		});
		expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(en["boarding.nothing_measured"]);
		expect(screen.getByText(/No GPU was free in time/)).toBeTruthy();
	});
});
