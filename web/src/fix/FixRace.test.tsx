// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../lib/flags", () => ({ flag: (name: string) => name === "c.fix_loop" }));

import en from "../i18n/en.json";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../workspace/record/types";
import { WorkspaceProvider } from "../workspace/state";
import { FixFinding } from "./FixFinding";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const f2 = dev.findings.find((f) => f.id === "f2")!;

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

describe("the fix race on screen", () => {
	it("shows the cap before the click, races the candidates in lanes and never calls a fix verified", async () => {
		vi.useFakeTimers();
		let asked: { candidates: Array<{ template_id: string }> } | null = null;
		const fetch = vi.fn(async (url: string, init?: RequestInit) => {
			if (url.startsWith("/v1/guide/limits")) return new Response(JSON.stringify({ saved: true, limits: null }), { status: 200 });
			if (url === "/v1/patches") {
				asked = JSON.parse(init?.body as string);
				return new Response(JSON.stringify({ patch_id: "fx_abcdefghijklmnopqr", status: "running", candidates: [], cap_usd: 0.39, candidates_asked: asked!.candidates }), { status: 202 });
			}
			const cands = asked!.candidates.map((c, i) => ({
				id: `c${i + 1}`,
				template_id: c.template_id,
				value: null,
				status: i === 0 ? "patched" : "no_change",
				diff: i === 0 ? "--- a/train.py\n+++ b/train.py\n" : "",
				files: i === 0 ? ["train.py"] : [],
				final: null,
				seconds: 21,
				problem: null,
				blocked: [],
			}));
			return new Response(JSON.stringify({ patch_id: "fx_abcdefghijklmnopqr", status: "done", candidates: cands, spent_usd: 0.0912 }), { status: 200 });
		});
		vi.stubGlobal("fetch", fetch);
		render(
			<WorkspaceProvider record={dev} recordRef="lora-7b" provider={new GridProvider()}>
				<FixFinding finding={f2} />
			</WorkspaceProvider>,
		);
		fireEvent.click(screen.getByRole("button", { name: en["guide.fix_it"] }));
		expect(screen.getByText(/paid by Preflight/).textContent).toMatch(/US\$0\.\d\d/);
		await act(async () => {
			fireEvent.click(screen.getByRole("button", { name: en["fix.start"] }));
		});
		// The limits are saved to the server first; the fix request itself carries none.
		expect(fetch.mock.calls.some((c) => c[0] === "/v1/guide/limits/state" && c[1]?.method === "PUT")).toBe(true);
		const call = fetch.mock.calls.find((c) => c[0] === "/v1/patches")!;
		const body = JSON.parse(call[1]!.body as string);
		expect(body).not.toHaveProperty("limits");
		expect(body).toMatchObject({ record: "lora-7b", finding: "f2" });
		expect(body.candidates[0].template_id).toBe("eval.reduce_batch");
		expect(screen.getAllByText(/Writing/).length).toBe(body.candidates.length);
		await act(async () => {
			await vi.advanceTimersByTimeAsync(2100);
		});
		expect(screen.getByText(en["fix.status.patched"])).toBeTruthy();
		expect(screen.getByText(en["fix.not_verified"])).toBeTruthy();
		expect(screen.queryByText(/verified\b(?!:)/i)?.textContent ?? "").not.toMatch(/^Verified/);
		expect(screen.getByText(/Muse Code used US\$0\.09/)).toBeTruthy();
	});

	it("stays away when the fix loop is off or the finding has no fix", () => {
		render(
			<WorkspaceProvider record={dev} recordRef="lora-7b" provider={new GridProvider()}>
				<FixFinding finding={{ ...f2, fix_template: null }} />
			</WorkspaceProvider>,
		);
		expect(screen.queryByRole("button", { name: en["guide.fix_it"] })).toBeNull();
	});
});

describe("the fix waits for the limits to be saved", () => {
	it("doesn't start when the server's copy can't be read or saved, and never blanks it", async () => {
		const { _resetLoaded, ensureSaved, NO_LIMITS } = await import("../guide/limits");
		_resetLoaded();
		const down = vi.fn(async () => new Response("", { status: 503 })) as unknown as typeof fetch;
		expect(await ensureSaved("lora-7b", NO_LIMITS, down)).toBe(false);
		_resetLoaded();
		const puts: string[] = [];
		const held = vi.fn(async (_url: string, init?: RequestInit) => {
			if (init?.method === "PUT") puts.push(init.body as string);
			return new Response(JSON.stringify({ limits: { frozen: ["packing"] }, saved: true }), { status: 200 });
		}) as unknown as typeof fetch;
		// The server holds limits and this tab has none yet: the server's copy stands.
		expect(await ensureSaved("lora-7b", NO_LIMITS, held)).toBe(true);
		expect(puts).toEqual([]);
	});
});
