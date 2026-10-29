/**
 * UI registry: panels, readouts, knob decorations and map overlays add themselves here,
 * and C's layout renders whatever is registered. Add-ons (web/src/addons/<name>/index.ts)
 * are imported automatically at start-up and call these functions; they never edit
 * C's components.
 *
 * Every entry may name a flag; it renders only when that flag is on.
 */
import type { ComponentType } from "react";
import { useSyncExternalStore } from "react";
import type { Estimate, JobRequest, Knob, KnobValue, Lens, LoadedRecord } from "./workspace/record/types";
import type { PredictionProvider } from "./workspace/provider/PredictionProvider";
import { guidedKnobs, useOptionalWorkspace, useRanked } from "./workspace/state";
import { flag } from "./lib/flags";

export interface WorkspaceContextValue {
	record: LoadedRecord;
	knobs: Readonly<Record<string, KnobValue>>;
	estimates: Readonly<Record<string, Estimate | undefined>>;
}

export type { KnobValue };

/**
 * Where a panel docks:
 * - "below": a band under the cockpit's whole-run section
 * - "drawer": inside a node's detail sheet
 * - "guide-footer": under the guide
 * - "cockpit-head": beside the cockpit title
 */
export type PanelSlot = "below" | "drawer" | "guide-footer" | "cockpit-head";

/**
 * What the check-in desk gives an "intake" panel. `setSource` hands the desk the panel's
 * `JobRequest.source` (null clears it); while the active tab has one, "Check in" sends it with
 * the goal, GPU and description. It is absent while check-in isn't open (the `runner.queue`
 * flag), so a panel can say so. Other slots get no props.
 */
export interface IntakePanelProps {
	setSource?: (source: JobRequest["source"] | null) => void;
}

/** Panels outside the cockpit: no record in hand. */
export interface AppPanelEntry {
	id: string;
	slot: "intake" | "logbook-settings" | "nav-end" | "home-beats";
	titleKey: string;
	order?: number;
	flag?: string;
	component: ComponentType<IntakePanelProps>;
}

export interface PanelEntry {
	id: string;
	slot: PanelSlot;
	/** i18n key for the tab or heading. */
	titleKey: string;
	lenses?: Lens[];
	order?: number;
	flag?: string;
	component: ComponentType<WorkspaceContextValue & { nodeId?: string }>;
}

export interface ReadoutEntry {
	id: string;
	/** Prediction target this readout shows. */
	targetId: string;
	titleKey: string;
	lenses?: Lens[];
	order?: number;
	flag?: string;
}

export interface KnobDecorationEntry {
	id: string;
	/** Which knobs it decorates; omit for all. */
	knobIds?: string[];
	flag?: string;
	component: ComponentType<{ knob: Knob; value: KnobValue } & WorkspaceContextValue & CockpitExtras>;
}

/** Read-only extras for add-ons inside the cockpit (D's request): the provider and the ranking. */
export interface CockpitExtras {
	provider: PredictionProvider;
	/** Knob ids by impact on the current objective at the current setting, most first. */
	ranked: readonly string[];
}

export interface MapOverlayEntry {
	id: string;
	lenses?: Lens[];
	flag?: string;
	order?: number;
	/** Rendered inside the map viewport, above the graph. */
	component: ComponentType<WorkspaceContextValue>;
}

/** What a knob view sees when it decides which knobs the dock shows (D's spec, Oct 27). */
export interface KnobViewContext {
	record: LoadedRecord;
	lens: Lens;
	/** Current knob values. */
	knobs: Readonly<Record<string, KnobValue>>;
	pinned: ReadonlySet<string>;
	/** Knob ids the guide is on right now: the one it is moving, and any it points at. */
	highlight: ReadonlySet<string>;
	/** Knob ids by impact on the current objective at the current setting, most first (sensitivity()). */
	ranked: readonly string[];
	/** The knob whose control is open, if any. */
	active: string | null;
}

/**
 * A view over the knob dock, e.g. D's "New to this" expertise level showing fewer knobs with
 * plainer names. Views apply in `order`, each filtering what the previous one left. Whatever a
 * view returns, the dock keeps knobs that are pinned, highlighted by the guide, open, or moved
 * from the record's setting, so nothing the person or the guide is acting on can vanish, and
 * it offers a "show all" capsule for the rest. A view that throws is skipped.
 */
export interface KnobViewEntry {
	id: string;
	flag?: string;
	lenses?: Lens[];
	order?: number;
	/** The knobs to show, from the knobs given (unknown knobs in the result are ignored). */
	filter?: (knobs: Knob[], ctx: KnobViewContext) => Knob[];
	/** i18n key for a knob's name in the dock and its control; unknown keys fall back to the record's. */
	labelKey?: (knob: Knob, ctx: KnobViewContext) => string;
	/** i18n key for the "show all" capsule, with {n} = how many are hidden. */
	moreKey?: string;
}

interface Registry {
	appPanels: AppPanelEntry[];
	panels: PanelEntry[];
	readouts: ReadoutEntry[];
	knobDecorations: KnobDecorationEntry[];
	mapOverlays: MapOverlayEntry[];
	knobViews: KnobViewEntry[];
}

let state: Registry = { appPanels: [], panels: [], readouts: [], knobDecorations: [], mapOverlays: [], knobViews: [] };
const listeners = new Set<() => void>();

function emit(next: Registry): void {
	state = next;
	for (const l of listeners) l();
}

function upsert<T extends { id: string }>(list: T[], entry: T): T[] {
	const rest = list.filter((e) => e.id !== entry.id);
	return [...rest, entry];
}

export function registerAppPanel(entry: AppPanelEntry): void {
	emit({ ...state, appPanels: upsert(state.appPanels, entry) });
}

export function registerPanel(entry: PanelEntry): void {
	emit({ ...state, panels: upsert(state.panels, entry) });
}

export function registerReadout(entry: ReadoutEntry): void {
	emit({ ...state, readouts: upsert(state.readouts, entry) });
}

export function registerKnobDecoration(entry: KnobDecorationEntry): void {
	emit({ ...state, knobDecorations: upsert(state.knobDecorations, entry) });
}

export function registerMapOverlay(entry: MapOverlayEntry): void {
	emit({ ...state, mapOverlays: upsert(state.mapOverlays, entry) });
}

export function registerKnobView(entry: KnobViewEntry): void {
	emit({ ...state, knobViews: upsert(state.knobViews, entry) });
}

/** Remove an entry by id from every list (for an add-on that turns itself off). */
export function unregister(id: string): void {
	const drop = <T extends { id: string }>(l: T[]) => l.filter((e) => e.id !== id);
	emit({
		appPanels: drop(state.appPanels),
		panels: drop(state.panels),
		readouts: drop(state.readouts),
		knobDecorations: drop(state.knobDecorations),
		mapOverlays: drop(state.mapOverlays),
		knobViews: drop(state.knobViews),
	});
}

function subscribe(fn: () => void): () => void {
	listeners.add(fn);
	return () => listeners.delete(fn);
}

function visible<T extends { flag?: string; lenses?: Lens[]; order?: number }>(
	list: T[],
	lens: Lens,
): T[] {
	return list
		.filter((e) => (e.flag ? flag(e.flag) : true) && (!e.lenses || e.lenses.includes(lens)))
		.sort((a, b) => (a.order ?? 100) - (b.order ?? 100));
}

export function useRegistry(): Registry {
	return useSyncExternalStore(subscribe, () => state, () => state);
}

export function useAppPanels(slot: AppPanelEntry["slot"]): AppPanelEntry[] {
	return useRegistry()
		.appPanels.filter((p) => p.slot === slot && (p.flag ? flag(p.flag) : true))
		.sort((a, b) => (a.order ?? 100) - (b.order ?? 100));
}

export function usePanels(slot: PanelSlot, lens: Lens): PanelEntry[] {
	const reg = useRegistry();
	return visible(
		reg.panels.filter((p) => p.slot === slot),
		lens,
	);
}

export function useReadouts(lens: Lens): ReadoutEntry[] {
	return visible(useRegistry().readouts, lens);
}

export function useMapOverlays(lens: Lens): MapOverlayEntry[] {
	return visible(useRegistry().mapOverlays, lens);
}

export function useKnobDecorations(knobId: string): KnobDecorationEntry[] {
	return useRegistry().knobDecorations.filter(
		(d) => (d.flag ? flag(d.flag) : true) && (!d.knobIds || d.knobIds.includes(knobId)),
	);
}

/**
 * Add-ons load from loadAddons.ts, not here: an eager glob in this file would be hoisted above
 * `state`, so an add-on that registers as it loads would hit it before initialisation and
 * blank the page (D's bug, Oct 27).
 */
export interface KnobViewResult {
	shown: Knob[];
	hidden: Knob[];
	/** Label key per knob id from the views (only keys that exist; others use the record's). */
	labelKeys: Readonly<Record<string, string>>;
	/** The "show all" capsule's key from the last view that hid anything. */
	moreKey: string | null;
}

const reported = new Set<string>();

function report(id: string, error: unknown): void {
	if (reported.has(id)) return;
	reported.add(id);
	console.error(`[preflight] knob view "${id}" failed and was skipped`, error);
}

/**
 * Apply the registered knob views to the record's knobs. Pure, for tests; `useKnobView` is
 * the hook. `exists` says whether an i18n key exists (label keys that don't are ignored).
 */
export function applyKnobViews(
	knobs: Knob[],
	ctx: KnobViewContext,
	views: KnobViewEntry[],
	exists: (key: string) => boolean = () => true,
): KnobViewResult {
	const kept = (k: Knob) =>
		ctx.pinned.has(k.id) ||
		ctx.highlight.has(k.id) ||
		ctx.active === k.id ||
		(k.id in ctx.knobs && ctx.knobs[k.id] !== k.current);
	let shown = [...knobs];
	let moreKey: string | null = null;
	const labelKeys: Record<string, string> = {};
	for (const view of visible(views, ctx.lens)) {
		try {
			if (view.filter) {
				const byId = new Map(shown.map((k) => [k.id, k]));
				const picked = view.filter([...shown], ctx).flatMap((k) => (byId.has(k?.id) ? [byId.get(k.id) as Knob] : []));
				const ids = new Set(picked.map((k) => k.id));
				// Keep the view's order, then add back what must stay, where it was.
				const forced = shown.filter((k) => !ids.has(k.id) && kept(k));
				const next = [...picked, ...forced].sort((a, b) => {
					const ia = picked.indexOf(a);
					const ib = picked.indexOf(b);
					if (ia >= 0 && ib >= 0) return ia - ib;
					return shown.indexOf(a) - shown.indexOf(b);
				});
				if (next.length < shown.length && view.moreKey) moreKey = view.moreKey;
				shown = next;
			}
			if (view.labelKey) {
				for (const k of knobs) {
					const key = view.labelKey(k, ctx);
					if (typeof key === "string" && exists(key)) labelKeys[k.id] = key;
				}
			}
		} catch (error) {
			report(view.id, error);
		}
	}
	const ids = new Set(shown.map((k) => k.id));
	return { shown, hidden: knobs.filter((k) => !ids.has(k.id)), labelKeys, moreKey };
}

export function useKnobViews(): KnobViewEntry[] {
	return useRegistry().knobViews;
}

/** What an add-on may read about the open cockpit. No actions: add-ons don't move knobs. */
export interface CockpitView extends WorkspaceContextValue, CockpitExtras {
	lens: Lens;
	pinned: ReadonlySet<string>;
	highlight: ReadonlySet<string>;
	objective: { kind?: string | null; target_id?: string | null };
}

/**
 * For any add-on component, app panels included (D's request): the open cockpit, read-only,
 * with the provider and the knob ranking; null when no cockpit is open (e.g. the logbook).
 * The ranking is computed only when `withRanking` is true.
 */
export function useCockpit(withRanking = false): CockpitView | null {
	const ws = useOptionalWorkspace();
	const ranked = useRanked(withRanking && ws !== null);
	if (!ws) return null;
	return {
		record: ws.record,
		lens: ws.record.job.lens,
		knobs: ws.state.knobs,
		estimates: ws.estimates,
		provider: ws.provider,
		ranked,
		pinned: ws.state.pinned,
		highlight: guidedKnobs(ws.state),
		objective: ws.state.objective,
	};
}
