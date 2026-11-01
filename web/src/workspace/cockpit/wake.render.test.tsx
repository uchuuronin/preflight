// @vitest-environment jsdom
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import devRaw from "../fixtures/dev-finetune.record.json";
import { cubicBezier, durMs, valueEase } from "../../lib/tween";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { useWorkspace, WorkspaceProvider } from "../state";
import { deltaText, Dials, WAKE_MS, WakeStrip, worthMarking } from "./Dials";

const record = normaliseRecord(devRaw as unknown as FlightRecord);

let ws: ReturnType<typeof useWorkspace> | null = null;
function Grab() {
	ws = useWorkspace();
	return null;
}
const open = () =>
	render(
		<WorkspaceProvider record={record} provider={new GridProvider()}>
			<Grab />
			<Dials />
			<WakeStrip />
		</WorkspaceProvider>,
	);

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
	cleanup();
	ws = null;
	vi.useRealTimers();
});

describe("the value clock's curve", () => {
	it("matches the CSS curve's ends and is monotonic", () => {
		expect(valueEase(0)).toBe(0);
		expect(valueEase(1)).toBe(1);
		let last = 0;
		for (let i = 1; i <= 20; i++) {
			const v = valueEase(i / 20);
			expect(v).toBeGreaterThanOrEqual(last);
			last = v;
		}
		// A strong ease-out: most of the way there by a third of the time.
		expect(valueEase(1 / 3)).toBeGreaterThan(0.75);
	});

	it("is the identity for a linear bezier", () => {
		const lin = cubicBezier(0, 0, 1, 1);
		for (const x of [0.1, 0.37, 0.5, 0.9]) expect(lin(x)).toBeCloseTo(x, 4);
	});
});

describe("a wake's text and threshold", () => {
	it("signs the change with a real minus", () => {
		const f = (v: number) => `US$${v.toFixed(2)}`;
		expect(deltaText(11, 14.42, f)).toBe("+US$3.42");
		expect(deltaText(14.42, 11, f)).toBe("−US$3.42");
	});

	it("marks changes above 1% of the old value or of the dial's span, never none", () => {
		const scale = { kind: "linear", lo: 0, hi: 1, source: "probability" } as const;
		expect(worthMarking(100, 100, null)).toBe(false);
		expect(worthMarking(100, 100.5, null)).toBe(false);
		expect(worthMarking(100, 102, null)).toBe(true);
		// A probability moving off zero is measured against the dial's span.
		expect(worthMarking(0, 0.05, scale)).toBe(true);
		expect(worthMarking(0, 0.005, scale)).toBe(false);
	});
});

describe("the wake after a knob move", () => {
	it("shows nothing on first draw", () => {
		const { container } = open();
		expect(container.querySelectorAll(".pf-dial__wake")).toHaveLength(0);
		expect(container.querySelectorAll(".pf-wakes__item")).toHaveLength(0);
	});

	it("marks the readouts a move changed, in the dials and the strip, then clears", () => {
		const { container } = open();
		const knob = record.knobs.find((k) => k.id === "gradient_checkpointing") ?? record.knobs.find((k) => k.kind === "bool");
		expect(knob).toBeTruthy();
		const before = ws!.estimates;
		act(() => ws!.actions.setKnob(knob!.id, !(ws!.state.knobs[knob!.id] ?? knob!.current), "user"));
		const changed = Object.keys(before).filter((id) => before[id]?.point !== ws!.estimates[id]?.point);
		expect(changed.length).toBeGreaterThan(0);
		const wakes = [...container.querySelectorAll(".pf-dial__wake")];
		expect(wakes.length).toBeGreaterThan(0);
		for (const w of wakes) expect(w.textContent).toMatch(/^[+−]/);
		expect(wakes.every((w) => w.getAttribute("aria-hidden") === "true")).toBe(true);
		expect(container.querySelectorAll(".pf-wakes__item").length).toBe(wakes.length);
		act(() => {
			vi.advanceTimersByTime(WAKE_MS + 10);
		});
		expect(container.querySelectorAll(".pf-dial__wake")).toHaveLength(0);
		expect(container.querySelectorAll(".pf-wakes__item")).toHaveLength(0);
	});

	it("colours a slower, dearer run as worse", () => {
		const { container } = open();
		const knob = record.knobs.find((k) => k.id === "gradient_checkpointing");
		expect(knob).toBeTruthy();
		act(() => ws!.actions.setKnob(knob!.id, true, "user"));
		expect(container.querySelectorAll(".pf-dial__wake.is-worse").length).toBeGreaterThan(0);
	});

	it("clears the last move's marks when the next move changes nothing", () => {
		const { container } = open();
		const knob = record.knobs.find((k) => k.id === "gradient_checkpointing")!;
		act(() => ws!.actions.setKnob(knob.id, true, "user"));
		expect(container.querySelectorAll(".pf-dial__wake").length).toBeGreaterThan(0);
		// Off and back on in one go: a new setting object with the same values, so nothing moved.
		const knobsBefore = ws!.state.knobs;
		act(() => {
			ws!.actions.setKnob(knob.id, false, "user");
			ws!.actions.setKnob(knob.id, true, "user");
		});
		expect(ws!.state.knobs).not.toBe(knobsBefore);
		// Cleared at once, not left to describe the last move.
		expect(container.querySelectorAll(".pf-dial__wake")).toHaveLength(0);
		act(() => {
			vi.advanceTimersByTime(WAKE_MS + 10);
		});
		expect(container.querySelectorAll(".pf-dial__wake")).toHaveLength(0);
		expect(container.querySelector(".pf-wakes__empty")).toBeTruthy();
	});

	it("shows a move's wake in a strip that mounts with the move (the guide opening the knob it moved)", () => {
		let showStrip = false;
		function Late() {
			return showStrip ? <WakeStrip /> : null;
		}
		const { container, rerender } = render(
			<WorkspaceProvider record={record} provider={new GridProvider()}>
				<Grab />
				<Dials />
				<Late />
			</WorkspaceProvider>,
		);
		const knob = record.knobs.find((k) => k.id === "gradient_checkpointing")!;
		act(() => ws!.actions.setKnob(knob.id, true, "guide"));
		showStrip = true;
		rerender(
			<WorkspaceProvider record={record} provider={new GridProvider()}>
				<Grab />
				<Dials />
				<Late />
			</WorkspaceProvider>,
		);
		expect(container.querySelectorAll(".pf-wakes__item").length).toBe(container.querySelectorAll(".pf-dial__wake").length);
		expect(container.querySelectorAll(".pf-wakes__item").length).toBeGreaterThan(0);
	});

	it("ends the tween on the final text and size class", () => {
		vi.useRealTimers();
		const { container } = open();
		const knob = record.knobs.find((k) => k.id === "gradient_checkpointing")!;
		act(() => ws!.actions.setKnob(knob.id, true, "user"));
		return new Promise<void>((done) => {
			setTimeout(() => {
				const b = container.querySelector(".pf-dial__text b")!;
				const want = b.textContent;
				expect(want).toMatch(/h|min/);
				expect(b.className === "" || /^is-long(er)?$/.test(b.className)).toBe(true);
				done();
			}, 800);
		});
	});
});
