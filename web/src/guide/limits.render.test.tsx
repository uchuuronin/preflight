// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

let live = false;
vi.mock("../lib/flags", () => ({ flag: (name: string) => name === "c.guide.live" && live }));

import en from "../i18n/en.json";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../workspace/record/types";
import { useWorkspace, type Workspace, WorkspaceProvider } from "../workspace/state";
import { GuidePanel } from "./GuidePanel";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	live = false;
});

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

const frame = () => act(() => new Promise<void>((r) => requestAnimationFrame(() => r())));

describe("limits and choices in the rule-based tour", () => {
	it("with the GPU kept on A100, the tour never moves the GPU, and 'Not this' on batch size says nothing is left", async () => {
		const r = mount();
		// Confirm A100 as the only GPU through the panel.
		fireEvent.click(screen.getByRole("button", { name: "A100-SXM-80GB" }));
		expect(r.ws!.state.limits.allowed.gpu_type).toEqual(["A100-SXM-80GB"]);
		fireEvent.click(screen.getByRole("button", { name: en["guide.start"] }));
		await frame();
		// Step 1: the eval batch fix. Apply it.
		fireEvent.click(screen.getByRole("button", { name: /^Apply / }));
		await frame();
		expect(r.ws!.state.proposal?.knobId).toBe("batch_size");
		expect(r.ws!.state.knobs.gpu_type).toBe("A100-SXM-80GB");
		// The choices name the value each leaves the knob at.
		expect(screen.getByRole("button", { name: "Keep 4" })).toBeTruthy();
		expect(screen.getByRole("button", { name: "Pin 8" })).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: en["guide.not_this"] }));
		await frame();
		expect(r.ws!.state.knobs.batch_size).toBe(4);
		expect(r.ws!.state.limits.excluded.batch_size).toEqual([8]);
		expect(screen.getByText(en["guide.done.limits"])).toBeTruthy();
		expect(r.ws!.state.knobs.gpu_type).toBe("A100-SXM-80GB");
	});

	it("'Keep' pins the visitor's own value; 'Pin' pins the suggestion", async () => {
		const r = mount();
		fireEvent.click(screen.getByRole("button", { name: en["guide.start"] }));
		await frame();
		fireEvent.click(screen.getByRole("button", { name: "Keep 32" }));
		await frame();
		expect(r.ws!.state.pinned.has("eval_batch_size")).toBe(true);
		expect(r.ws!.state.knobs.eval_batch_size).toBe(32);
		cleanup();
		const r2 = mount();
		fireEvent.click(screen.getByRole("button", { name: en["guide.start"] }));
		await frame();
		fireEvent.click(screen.getByRole("button", { name: "Pin 8" }));
		await frame();
		expect(r2.ws!.state.pinned.has("eval_batch_size")).toBe(true);
		expect(r2.ws!.state.knobs.eval_batch_size).toBe(8);
	});

	it("the visitor's own move sets the suggestion aside and the tour carries on from their setting", async () => {
		const r = mount();
		fireEvent.click(screen.getByRole("button", { name: en["guide.start"] }));
		await frame();
		act(() => {
			r.ws!.actions.setKnob("eval_batch_size", 4, "user");
		});
		await frame();
		expect(r.ws!.state.knobs.eval_batch_size).toBe(4);
		expect(r.ws!.state.proposal).toBeNull();
		expect(screen.getByText(/so the guide's suggestion is set aside/)).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: en["guide.carry_on"] }));
		await frame();
		expect(r.ws!.state.proposal?.knobId).not.toBe("eval_batch_size");
	});
});

describe("limits in plain words", () => {
	it("come back as chips that apply only once confirmed, and say what wasn't understood", async () => {
		live = true;
		const fetch = vi.fn(async (url: string) =>
			url.startsWith("/v1/guide/limits/state") ? new Response(JSON.stringify({ limits: null, saved: true }), { status: 200 }) : new Response(
				JSON.stringify({
					limits: [
						{ kind: "allowed", knob_id: "gpu_type", values: ["A100-SXM-80GB"] },
						{ kind: "budget_usd", value: 40 },
					],
					not_understood: ["can't touch the dataset"],
				}),
				{ status: 200 },
			),
		);
		vi.stubGlobal("fetch", fetch);
		const r = mount();
		fireEvent.change(screen.getByLabelText(en["limits.words.label"]), { target: { value: "must stay on A100, under $40, can't touch the dataset" } });
		await act(async () => {
			fireEvent.click(screen.getByRole("button", { name: en["limits.words.read"] }));
		});
		const read = fetch.mock.calls.find((c) => (c as unknown[])[0] === "/v1/guide/limits") as unknown as [string, RequestInit];
		expect(JSON.parse(read[1].body as string)).toEqual({ record: "lora-7b", text: "must stay on A100, under $40, can't touch the dataset" });
		expect(screen.getByText(/^Under US\$40/)).toBeTruthy();
		expect(r.ws!.state.limits.budgetUsd).toBeNull(); // nothing applies before it's confirmed
		expect(screen.getByText(/^Not turned into a limit: "can't touch the dataset"/)).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: en["limits.confirm_all"] }));
		expect(r.ws!.state.limits.budgetUsd).toBe(40);
		expect(r.ws!.state.limits.allowed.gpu_type).toEqual(["A100-SXM-80GB"]);
	});
});

describe("the server keeps this session's limits", () => {
	it("saves each change and restores them on a reload", async () => {
		const puts: unknown[] = [];
		const fetch = vi.fn(async (url: string, init?: RequestInit) => {
			if (init?.method === "PUT") {
				puts.push(JSON.parse(init.body as string));
				return new Response(JSON.stringify({ saved: true }), { status: 200 });
			}
			return new Response(JSON.stringify({ limits: { frozen: ["packing"], allowed: {}, excluded: {}, declined: [] } }), { status: 200 });
		});
		vi.stubGlobal("fetch", fetch);
		const r = mount();
		await act(async () => {
			await new Promise((res) => setTimeout(res, 20));
		});
		expect(r.ws!.state.limits.frozen).toEqual(["packing"]); // restored from the server
		fireEvent.click(screen.getByRole("button", { name: "A100-SXM-80GB" }));
		await act(async () => {
			await new Promise((res) => setTimeout(res, 350));
		});
		expect(puts.at(-1)).toMatchObject({ record: "lora-7b", limits: { allowed: { gpu_type: ["A100-SXM-80GB"] }, frozen: ["packing"] } });
	});
});

describe("the live guide falling back", () => {
	it("starts the rule-based tour at once when the live guide can't be reached, so the visitor's press still gives a move", async () => {
		live = true;
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response(JSON.stringify({ error: { code: "unavailable", message: "Preflight's server isn't connected yet. Samples still work." } }), { status: 503 })),
		);
		mount();
		await act(async () => {
			fireEvent.click(screen.getByRole("button", { name: en["guide.start"] }));
		});
		for (let i = 0; i < 5; i++) await frame();
		// The notice says why, and the tour's first step is there with its move.
		expect(document.querySelector(".pf-guide__notice")?.textContent).toContain("isn't connected");
		expect(screen.getAllByRole("button", { name: /^Apply / }).length).toBeGreaterThan(0);
		// Focus moves to the step's move, since the pressed button is gone.
		for (let i = 0; i < 3; i++) await frame();
		expect((document.activeElement as HTMLElement | null)?.textContent).toMatch(/^Apply /);
	});
});
