/**
 * Workspace state and the action surface. The person's controls and the guide's tools call
 * the same actions (setKnob, pin, highlight, snapshot, compare, focus), so the guide can never
 * do anything the person couldn't, and pins are enforced here as well as in the tool schema.
 */
import {
	createContext,
	useCallback,
	useContext,
	useEffect,
	useMemo,
	useReducer,
	useRef,
	type ReactNode,
} from "react";
import { currentKnobs } from "./record/model";
import type { KnobValue, LoadedRecord, Objective, ObjectiveKind, Prediction } from "./record/types";
import type { PredictionProvider } from "./provider/PredictionProvider";
import { recordKnobLatency } from "./telemetry";

export interface Snapshot {
	id: string;
	name: string;
	knobs: Record<string, KnobValue>;
	estimates: Prediction;
}

export interface Proposal {
	knobId: string;
	from: KnobValue;
	to: KnobValue;
}

/** One guide tool call as shown in the trace: what was called, and what came back. */
export interface TraceEntry {
	id: number;
	tool: string;
	args: string;
	result: string;
	ok: boolean;
	pending?: boolean;
}

interface State {
	knobs: Record<string, KnobValue>;
	pinned: ReadonlySet<string>;
	objective: Objective;
	/** Estimates before the most recent change, for delta badges. */
	previous: Prediction | null;
	lastChange: { knobId: string; by: "user" | "guide" } | null;
	highlight: string[];
	proposal: Proposal | null;
	focus: string | null;
	activeKnob: string | null;
	snapshots: Snapshot[];
	ghost: string | null;
	playhead: number;
	trace: TraceEntry[];
}

type Action =
	| { type: "set"; knobId: string; value: KnobValue; by: "user" | "guide"; previous: Prediction }
	| { type: "pin"; knobId: string; pinned: boolean }
	| { type: "objective"; kind: ObjectiveKind }
	| { type: "highlight"; nodes: string[] }
	| { type: "propose"; proposal: Proposal | null }
	| { type: "focus"; node: string | null }
	| { type: "activeKnob"; knobId: string | null }
	| { type: "snapshot"; snapshot: Snapshot }
	| { type: "ghost"; id: string | null }
	| { type: "playhead"; at: number }
	| { type: "trace"; entry: TraceEntry }
	| { type: "reset"; knobs: Record<string, KnobValue> };

const TRACE_LIMIT = 40;

function reducer(state: State, action: Action): State {
	switch (action.type) {
		case "set":
			if (action.by === "guide" && state.pinned.has(action.knobId)) return state;
			return {
				...state,
				knobs: { ...state.knobs, [action.knobId]: action.value },
				previous: action.previous,
				lastChange: { knobId: action.knobId, by: action.by },
			};
		case "pin": {
			const pinned = new Set(state.pinned);
			if (action.pinned) pinned.add(action.knobId);
			else pinned.delete(action.knobId);
			return { ...state, pinned };
		}
		case "objective":
			return { ...state, objective: { ...state.objective, kind: action.kind } };
		case "highlight":
			return { ...state, highlight: action.nodes };
		case "propose":
			return { ...state, proposal: action.proposal };
		case "focus":
			return { ...state, focus: action.node };
		case "activeKnob":
			return { ...state, activeKnob: action.knobId };
		case "snapshot":
			return { ...state, snapshots: [...state.snapshots, action.snapshot] };
		case "ghost":
			return { ...state, ghost: action.id };
		case "playhead":
			return { ...state, playhead: Math.min(1, Math.max(0, action.at)) };
		case "trace": {
			const rest = state.trace.filter((e) => e.id !== action.entry.id);
			return { ...state, trace: [...rest, action.entry].slice(-TRACE_LIMIT) };
		}
		case "reset":
			return { ...state, knobs: action.knobs, previous: null, lastChange: null, proposal: null, highlight: [] };
	}
}

export interface WorkspaceActions {
	setKnob(knobId: string, value: KnobValue, by?: "user" | "guide"): boolean;
	pin(knobId: string, pinned: boolean): void;
	setObjective(kind: ObjectiveKind): void;
	highlight(nodes: string[]): void;
	propose(proposal: Proposal | null): void;
	focus(node: string | null): void;
	openKnob(knobId: string | null): void;
	snapshot(name?: string): Snapshot;
	compare(a: string, b: string): Array<{ targetId: string; a: number | null; b: number | null }>;
	showGhost(id: string | null): void;
	setPlayhead(at: number): void;
	trace(entry: TraceEntry): void;
	reset(): void;
}

export interface Workspace {
	record: LoadedRecord;
	provider: PredictionProvider;
	state: State;
	estimates: Prediction;
	actions: WorkspaceActions;
}

const Ctx = createContext<Workspace | null>(null);

export function WorkspaceProvider({
	record,
	provider,
	children,
}: {
	record: LoadedRecord;
	provider: PredictionProvider;
	children: ReactNode;
}) {
	const [state, dispatch] = useReducer(reducer, record, (r) => ({
		knobs: currentKnobs(r),
		pinned: new Set<string>(),
		objective: r.job.objective ?? { kind: "min_time" },
		previous: null,
		lastChange: null,
		highlight: [],
		proposal: null,
		focus: null,
		activeKnob: null,
		snapshots: [],
		ghost: null,
		playhead: 0,
		trace: [],
	}));

	const changeStart = useRef<number | null>(null);
	const estimates = useMemo(() => provider.predict(record, state.knobs), [provider, record, state.knobs]);

	// Knob latency is measured to the frame after the new readouts commit, which is what the
	// person sees; the contract's target is under 400 ms at p95.
	useEffect(() => {
		const start = changeStart.current;
		if (start === null) return;
		changeStart.current = null;
		const id = requestAnimationFrame(() => recordKnobLatency(performance.now() - start));
		return () => cancelAnimationFrame(id);
	}, [estimates]);

	const latest = useRef({ state, estimates });
	latest.current = { state, estimates };

	const setKnob = useCallback(
		(knobId: string, value: KnobValue, by: "user" | "guide" = "user"): boolean => {
			const { state: s, estimates: e } = latest.current;
			if (by === "guide" && s.pinned.has(knobId)) return false;
			if (!record.knobs.some((k) => k.id === knobId)) return false;
			changeStart.current = performance.now();
			dispatch({ type: "set", knobId, value, by, previous: e });
			return true;
		},
		[record],
	);

	const actions = useMemo<WorkspaceActions>(
		() => ({
			setKnob,
			pin: (knobId, pinned) => dispatch({ type: "pin", knobId, pinned }),
			setObjective: (kind) => dispatch({ type: "objective", kind }),
			highlight: (nodes) => dispatch({ type: "highlight", nodes }),
			propose: (proposal) => dispatch({ type: "propose", proposal }),
			focus: (node) => dispatch({ type: "focus", node }),
			openKnob: (knobId) => dispatch({ type: "activeKnob", knobId }),
			snapshot: (name) => {
				const { state: s, estimates: e } = latest.current;
				const snap: Snapshot = {
					id: `s${s.snapshots.length + 1}`,
					name: name || `Snapshot ${s.snapshots.length + 1}`,
					knobs: { ...s.knobs },
					estimates: e,
				};
				dispatch({ type: "snapshot", snapshot: snap });
				return snap;
			},
			compare: (a, b) => {
				const snaps = latest.current.state.snapshots;
				const sa = snaps.find((s) => s.id === a);
				const sb = snaps.find((s) => s.id === b);
				if (!sa || !sb) return [];
				return Object.keys(sa.estimates).map((targetId) => ({
					targetId,
					a: sa.estimates[targetId]?.point ?? null,
					b: sb.estimates[targetId]?.point ?? null,
				}));
			},
			showGhost: (id) => dispatch({ type: "ghost", id }),
			setPlayhead: (at) => dispatch({ type: "playhead", at }),
			trace: (entry) => dispatch({ type: "trace", entry }),
			reset: () => dispatch({ type: "reset", knobs: currentKnobs(record) }),
		}),
		[setKnob, record],
	);

	const value = useMemo<Workspace>(
		() => ({ record, provider, state, estimates, actions }),
		[record, provider, state, estimates, actions],
	);
	return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useWorkspace(): Workspace {
	const ws = useContext(Ctx);
	if (!ws) throw new Error("useWorkspace outside WorkspaceProvider");
	return ws;
}

export function useOptionalWorkspace(): Workspace | null {
	return useContext(Ctx);
}
