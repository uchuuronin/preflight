// @vitest-environment jsdom
// The quick read (PM, Nov 3): while Muse Spark reads the record, a progress line and the rule-based
// tour's first step show at once; the live guide's answer replaces them; a slow or failed answer leaves
// the quick step in place, and taking it stops the live guide, so a late answer moves nothing.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../lib/flags", () => ({ flag: (name: string) => name === "c.guide.live" }));

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
const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });
const answer = (over: Record<string, unknown> = {}) => ({ text: "Muse Spark's first move.", withheld: null, calls: [], awaiting: "visitor", state: "s1", provenance: {}, ...over });

/** The guide's turns wait for `release`; everything else (limits) answers at once. */
function slowGuide() {
	let release: (r: Response) => void = () => {};
	const turn = new Promise<Response>((r) => {
		release = r;
	});
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string) => (url.startsWith("/v1/guide/turns") ? turn : ok({ limits: null, saved: true }))),
	);
	return (r: Response) => act(async () => release(r));
}

const quickText = () => document.querySelector(".pf-guide__say--quick")?.textContent ?? "";

async function start() {
	await act(async () => {
		fireEvent.click(screen.getByRole("button", { name: en["guide.start"] }));
	});
	await frame();
}

describe("the quick read while Muse Spark reads the record", () => {
	it("shows the progress line and the rule-based tour's first step at once, and moves nothing", async () => {
		slowGuide();
		const r = mount();
		const before = { ...r.ws!.state.knobs };
		await start();
		expect(screen.getByText(en["guide.live.reading"])).toBeTruthy();
		expect(screen.getByText(en["guide.quick.label"])).toBeTruthy();
		expect(quickText()).toMatch(/eval batch/i);
		expect(r.ws!.state.knobs).toEqual(before);
		expect(r.ws!.state.proposal).toBeNull();
	});

	it("a slow answer leaves the quick step in place; the answer, when it comes, takes over", async () => {
		const release = slowGuide();
		mount();
		await start();
		for (let i = 0; i < 5; i++) await frame();
		expect(quickText()).not.toBe("");
		await release(ok(answer()));
		await frame();
		expect(screen.getByText("Muse Spark's first move.")).toBeTruthy();
		expect(document.querySelector(".pf-guide__quick")).toBeNull();
	});

	it("a failed answer starts the rule-based tour on the same step", async () => {
		const release = slowGuide();
		mount();
		await start();
		const said = quickText();
		await release(new Response(JSON.stringify({ error: { code: "unavailable", message: "Muse Spark isn't answering." } }), { status: 503 }));
		for (let i = 0; i < 4; i++) await frame();
		expect(document.querySelector(".pf-guide__quick")).toBeNull();
		const step = [...document.querySelectorAll(".pf-guide__say")].map((e) => e.textContent);
		expect(step).toContain(said);
		expect(screen.getAllByRole("button", { name: /^Apply / }).length).toBeGreaterThan(0);
	});

	it("taking the quick read stops the live guide: its late answer shows nothing and moves no knob", async () => {
		const release = slowGuide();
		const r = mount();
		await start();
		const said = quickText();
		fireEvent.click(screen.getByRole("button", { name: en["guide.quick.take"] }));
		for (let i = 0; i < 3; i++) await frame();
		const step = [...document.querySelectorAll(".pf-guide__say")].map((e) => e.textContent);
		expect(step).toContain(said);
		const knobs = { ...r.ws!.state.knobs };
		await release(
			ok(
				answer({
					text: "Late answer.",
					awaiting: "results",
					calls: [{ call_id: "c1", name: "set_knob", arguments: { knob_id: "batch_size", value: 1 }, runs_in: "browser" }],
				}),
			),
		);
		for (let i = 0; i < 3; i++) await frame();
		expect(screen.queryByText("Late answer.")).toBeNull();
		expect(r.ws!.state.knobs).toEqual(knobs);
	});
});

describe("the quick read goes once Muse Spark has done anything", () => {
	it("a first round that moves a knob hides it, so it can't be taken from the guide's setting", async () => {
		let second: (r: Response) => void = () => {};
		let n = 0;
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: string) => {
				if (!url.startsWith("/v1/guide/turns")) return ok({ limits: null, saved: true });
				n++;
				if (n === 1) return ok(answer({ text: "", awaiting: "results", calls: [{ call_id: "c1", name: "set_knob", arguments: { knob_id: "eval_batch_size", value: 8 }, runs_in: "browser" }] }));
				return new Promise<Response>((r) => {
					second = r;
				});
			}),
		);
		mount();
		await start();
		for (let i = 0; i < 4; i++) await frame();
		expect(n).toBe(2);
		expect(document.querySelector(".pf-guide__quick")).toBeNull();
		await act(async () => second(ok(answer())));
	});

	it("follows the visitor's own move while it shows", async () => {
		slowGuide();
		const r = mount();
		await start();
		const first = quickText();
		act(() => r.ws!.actions.setKnob("eval_batch_size", 8, "user"));
		await frame();
		expect(quickText()).not.toBe("");
		expect(quickText()).not.toBe(first);
	});
});

describe("what the visitor wrote on New flight", () => {
	it("waits in the guide's own box, and goes to Muse Spark when they ask", async () => {
		const release = slowGuide();
		const said = "Fine-tune on my support tickets; cheapest that fits";
		const withWords = { ...dev, job: { ...dev.job, objective: { ...(dev.job.objective ?? {}), description: said } } };
		render(
			<WorkspaceProvider record={withWords} recordRef="lora-7b" provider={new GridProvider()}>
				<GuidePanel />
			</WorkspaceProvider>,
		);
		await start();
		await release(ok(answer()));
		await frame();
		const box = screen.getByLabelText(en["guide.ask.label"]) as HTMLTextAreaElement;
		expect(box.value).toBe(said);
		await act(async () => {
			fireEvent.click(screen.getByRole("button", { name: en["guide.ask.send"] }));
		});
		const calls = (fetch as unknown as { mock: { calls: Array<[string, RequestInit]> } }).mock.calls.filter((c) => c[0].startsWith("/v1/guide/turns"));
		expect(JSON.parse(calls.at(-1)![1].body as string).message).toBe(said);
	});
});

describe("a first answer cut off after a round of reading tools (live, Nov 3)", () => {
	const twoRounds = (first: Record<string, unknown>) => {
		let n = 0;
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: string) => {
				if (!url.startsWith("/v1/guide/turns")) return ok({ limits: null, saved: true });
				n++;
				if (n === 1) return ok(answer({ text: "Starting your cost tour.", awaiting: "results", ...first }));
				return new Response(JSON.stringify({ error: { code: "timeout", message: "The guide model took too long to answer.", fallback: "rule_based", reason: "timeout" } }), { status: 504 });
			}),
		);
	};

	it("still hands over to the rule-based tour, with its first move", async () => {
		twoRounds({ calls: [{ call_id: "c1", name: "highlight", arguments: { nodes: [] }, runs_in: "browser" }] });
		mount();
		await start();
		for (let i = 0; i < 6; i++) await frame();
		expect(screen.getAllByRole("button", { name: /^Apply / }).length).toBeGreaterThan(0);
	});

	it("doesn't, once Muse Spark has moved a knob: the visitor decides from there", async () => {
		twoRounds({ calls: [{ call_id: "c1", name: "set_knob", arguments: { knob_id: "eval_batch_size", value: 8 }, runs_in: "browser" }] });
		mount();
		await start();
		for (let i = 0; i < 6; i++) await frame();
		expect(screen.queryAllByRole("button", { name: /^Apply / }).length).toBe(0);
		expect(document.querySelector(".pf-guide__notice")?.textContent).toMatch(/too long/);
	});
});
