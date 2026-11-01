/** Test helpers for the guide: WorkspaceActions that record what the guide did. */
import { vi } from "vitest";
import type { KnobValue } from "../workspace/record/types";
import type { TraceEntry, WorkspaceActions } from "../workspace/state";

export function fakeActions(pinned = new Set<string>()) {
	const moves: Array<[string, KnobValue]> = [];
	const trace: TraceEntry[] = [];
	const actions: WorkspaceActions = {
		setKnob: (id, v, by = "user") => {
			if (by === "guide" && pinned.has(id)) return false;
			moves.push([id, v]);
			return true;
		},
		pin: (id, on) => {
			if (on) pinned.add(id);
			else pinned.delete(id);
		},
		setObjective: vi.fn(),
		highlight: vi.fn(),
		propose: vi.fn(),
		focus: vi.fn(),
		openKnob: vi.fn(),
		snapshot: vi.fn(() => ({ id: "s1", name: "s1", knobs: {}, estimates: {} })),
		compare: vi.fn(() => []),
		showGhost: vi.fn(),
		setPlayhead: vi.fn(),
		trace: (e) => trace.push(e),
		reset: vi.fn(),
		setLimits: vi.fn(),
		decline: vi.fn(),
		setRate: vi.fn(),
	};
	return { actions, moves, trace, pinned };
}
