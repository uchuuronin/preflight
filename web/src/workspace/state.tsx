/**
 * Workspace state and the action surface. The user's controls and the guide's tools call
 * the same actions (setKnob, pin, highlight, snapshot, compare), so the guide can never do
 * anything the user couldn't, and pins are enforced here as well as in the tool schema.
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
import type { FlightRecord, KnobValue, Prediction } from "./record/types";
import type { PredictionProvider } from "./provider/PredictionProvider";
import { currentKnobs } from "./provider/PredictionProvider";
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
	by: "guide";
}

interface State {
	knobs: Record<string, KnobValue>;
	pinned: ReadonlySet<string>;
	/** Estimates before the most recent change, for delta badges. */
	previous: Prediction | null;
	/** Which knob moved last and by whom, for the rail's trace. */
	lastChange: { knobId: string; by: "user" | "guide" } | null;
	highlight: string[];
	proposal: Proposal | null;
	snapshots: Snapshot[];
	ghost: string | null;
	/** Playhead position along the run, 0..1. */
	playhead: number;
}

type Action =
	| { type: "set"; knobId: string; value: KnobValue; by: "user" | "guide"; previous: Prediction }
	| { type: "pin"; knobId: string; pinned: boolean }
	| { type: "highlight"; nodes: string[] }
	| { type: "propose"; proposal: Proposal | null }
	| { type: "snapshot"; snapshot: Snapshot }
	| { type: "ghost"; id: string | null }
	| { type: "playhead"; at: number }
	| { type: "reset"; knobs: Record<string, KnobValue> };

function reducer(state: State, action: Action): State {
	switch (action.type) {
		case "set":
			if (state.pinned.has(action.knobId) && action.by === "guide") return state;
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
		case "highlight":
			return { ...state, highlight: action.nodes };
		case "propose":
			return { ...state, proposal: action.proposal };
		case "snapshot":
			return { ...state, snapshots: [...state.snapshots, action.snapshot] };
		case "ghost":
			return { ...state, ghost: action.id };
		case "playhead":
			return { ...state, playhead: Math.min(1, Math.max(0, action.at)) };
		case "reset":
			return { ...state, knobs: action.knobs, previous: null, lastChange: null, proposal: null };
	}
}

export interface WorkspaceActions {
	setKnob(knobId: string, value: KnobValue, by?: "user" | "guide"): boolean;
	pin(knobId: string, pinned: boolean): void;
	highlight(nodes: string[]): void;
	propose(proposal: Proposal | null): void;
	snapshot(name?: string): Snapshot;
	compare(a: string, b: string): Array<{ targetId: string; a: number; b: number }>;
	showGhost(id: string | null): void;
	setPlayhead(at: number): void;
	reset(): void;
}

export interface Workspace {
	record: FlightRecord;
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
	record: FlightRecord;
	provider: PredictionProvider;
	children: ReactNode;
}) {
	const [state, dispatch] = useReducer(reducer, record, (r) => ({
		knobs: currentKnobs(r),
		pinned: new Set<string>(),
		previous: null,
		lastChange: null,
		highlight: [],
		proposal: null,
		snapshots: [],
		ghost: null,
		playhead: 0,
	}));

	const changeStart = useRef<number | null>(null);

	const estimates = useMemo(() => provider.predict(record, state.knobs), [provider, record, state.knobs]);

	// Knob latency is measured to the frame after the new readouts commit, which is what
	// the person sees; the contract's target is under 400 ms at p95.
	useEffect(() => {
		const start = changeStart.current;
		if (start === null) return;
		changeStart.current = null;
		const id = requestAnimationFrame(() => recordKnobLatency(performance.now() - start));
		return () => cancelAnimationFrame(id);
	}, [estimates]);

	// Actions read the latest state through refs so their identity stays stable.
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
			highlight: (nodes) => dispatch({ type: "highlight", nodes }),
			propose: (proposal) => dispatch({ type: "propose", proposal }),
			snapshot: (name) => {
				const { state: s, estimates: e } = latest.current;
				const snap: Snapshot = {
					id: `s${s.snapshots.length + 1}`,
					name: name ?? `Snapshot ${s.snapshots.length + 1}`,
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
					a: sa.estimates[targetId]?.point ?? Number.NaN,
					b: sb.estimates[targetId]?.point ?? Number.NaN,
				}));
			},
			showGhost: (id) => dispatch({ type: "ghost", id }),
			setPlayhead: (at) => dispatch({ type: "playhead", at }),
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
