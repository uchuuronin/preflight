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
import type { Estimate, Knob, KnobValue, Lens, LoadedRecord } from "./workspace/record/types";
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

/** Panels outside the cockpit: no record in hand. */
export interface AppPanelEntry {
	id: string;
	slot: "intake" | "logbook-settings" | "nav-end" | "home-beats";
	titleKey: string;
	order?: number;
	flag?: string;
	component: ComponentType;
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
	component: ComponentType<{ knob: Knob; value: KnobValue } & WorkspaceContextValue>;
}

export interface MapOverlayEntry {
	id: string;
	lenses?: Lens[];
	flag?: string;
	order?: number;
	/** Rendered inside the map viewport, above the graph. */
	component: ComponentType<WorkspaceContextValue>;
}

interface Registry {
	appPanels: AppPanelEntry[];
	panels: PanelEntry[];
	readouts: ReadoutEntry[];
	knobDecorations: KnobDecorationEntry[];
	mapOverlays: MapOverlayEntry[];
}

let state: Registry = { appPanels: [], panels: [], readouts: [], knobDecorations: [], mapOverlays: [] };
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

/** Import every add-on once. Called from main.tsx before the first render. */
export function loadAddons(): void {
	import.meta.glob("./addons/*/index.ts", { eager: true });
	import.meta.glob("./addons/*/index.tsx", { eager: true });
}
