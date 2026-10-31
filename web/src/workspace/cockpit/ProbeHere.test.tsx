// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/flags", () => ({ flag: (name: string) => name === "runner.queue" }));
const go = vi.fn();
vi.mock("../../app/route", () => ({ go: (...a: unknown[]) => go(...a) }));

import en from "../../i18n/en.json";
import devRaw from "../fixtures/dev-finetune.record.json";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { ProbeHere } from "./ProbeHere";

const raw = devRaw as unknown as FlightRecord;
const TOKEN = "AbCdEfGhIjKlMnOpQrSt";
// The same record as a live one (no fixture model ids), with peak memory abstaining here.
function live(abstain: boolean) {
	const est = { ...(raw.predictions?.estimates ?? {}) } as Record<string, { status: string; model_id?: string } | undefined>;
	for (const [k, e] of Object.entries(est)) if (e) est[k] = { ...e, model_id: "memory-analytical@0.2.0" };
	if (abstain) est["memory.peak"] = { status: "abstain", reason: "off_grid", model_id: "memory-analytical@0.2.0" } as never;
	return normaliseRecord({ ...raw, models: [], predictions: { ...raw.predictions, estimates: est, grids: [] } } as unknown as FlightRecord);
}

const open = (record: ReturnType<typeof live>, ref = TOKEN) =>
	render(
		<WorkspaceProvider record={record} recordRef={ref} provider={new GridProvider()}>
			<ProbeHere />
		</WorkspaceProvider>,
	);

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	go.mockReset();
});

describe("run a probe here", () => {
	it("offers to measure a setting where a readout abstains, and sends one probe of it", async () => {
		const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "01JOB" }), { status: 202 }));
		vi.stubGlobal("fetch", fetch);
		open(live(true));
		expect(screen.getByText(/peak memory/)).toBeTruthy();
		await act(async () => {
			fireEvent.click(screen.getByRole("button", { name: en["probehere.go"] }));
		});
		const [url, init] = fetch.mock.calls[0]!;
		expect(url).toBe("/v1/jobs");
		const body = JSON.parse(init.body as string);
		expect(body.source).toEqual({ kind: "record", record_ref: TOKEN });
		expect(body.mode).toBe("probe");
		expect(Object.keys(body.knobs).length).toBeGreaterThan(0);
		expect(init.headers["Idempotency-Key"]).toMatch(/^probe-/);
		expect(go).toHaveBeenCalledWith("boarding", "01JOB");
		expect(screen.getByText(en["probehere.cost"]).textContent).not.toMatch(/\$|US\$/);
	});

	it("stays away when everything is trusted, on samples, and on a ref that isn't a share token", () => {
		open(live(false));
		expect(screen.queryByRole("button", { name: en["probehere.go"] })).toBeNull();
		cleanup();
		open(normaliseRecord(raw));
		expect(screen.queryByRole("button", { name: en["probehere.go"] })).toBeNull();
		cleanup();
		open(live(true), "lora-7b");
		expect(screen.queryByRole("button", { name: en["probehere.go"] })).toBeNull();
	});

	it("says why when the queue is full", async () => {
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { message: "The probe queue is full." } }), { status: 429, headers: { "Retry-After": "60" } })));
		open(live(true));
		await act(async () => {
			fireEvent.click(screen.getByRole("button", { name: en["probehere.go"] }));
		});
		expect(screen.getByRole("alert").textContent).toBe("The probe queue is full.");
		expect(go).not.toHaveBeenCalled();
	});
});
