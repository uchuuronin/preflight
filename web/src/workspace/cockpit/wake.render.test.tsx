// @vitest-environment jsdom
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import devRaw from "../fixtures/dev-finetune.record.json";
import { cubicBezier, valueEase } from "../../lib/tween";
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
		const knob = record.knobs.find((k) => k.id === "memory.gradient_checkpointing") ?? record.knobs.find((k) => k.kind === "bool");
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
		const knob = record.knobs.find((k) => k.id === "memory.gradient_checkpointing");
		if (!knob) return;
		act(() => ws!.actions.setKnob(knob.id, true, "user"));
		const worse = container.querySelectorAll(".pf-dial__wake.is-worse");
		expect(worse.length).toBeGreaterThan(0);
	});
});
