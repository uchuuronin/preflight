/**
 * In-flight mode, C's half: the live record `preflight watch` writes on the visitor's own machine
 * (A's recorder/inflight.py) reaches the cockpit without leaving it. The visitor picks the file;
 * where the browser can keep a handle on it (the File System Access API, Chromium), it is read
 * again whenever it changes, so the cockpit follows the run. Elsewhere the file is read once, and
 * picking it again refreshes it. Nothing is uploaded.
 */
import { useSyncExternalStore } from "react";
import { t } from "../../lib/i18n";
import type { FlightRecord } from "../record/types";

/** Source: derived, A's collector rewrites the live record at most every 5 s (inflight.WRITE_EVERY_S). */
export const POLL_MS = 3_000;
/** The route parameter the cockpit opens the live record by. */
export const LIVE_REF = "live";

export interface LiveState {
	record: FlightRecord | null;
	name: string | null;
	/** When the file last changed, as the file system says (ms since the epoch). */
	modified: number | null;
	error: string | null;
	/** A handle is kept and polled (Chromium); otherwise the file was read once. */
	following: boolean;
}

interface FileHandle {
	getFile(): Promise<File>;
}

const EMPTY: LiveState = { record: null, name: null, modified: null, error: null, following: false };
let state: LiveState = EMPTY;
let handle: FileHandle | null = null;
let timer: ReturnType<typeof setInterval> | null = null;
const listeners = new Set<() => void>();

function set(next: Partial<LiveState>): void {
	state = { ...state, ...next };
	for (const l of listeners) l();
}

/** The live record's own shape: a record whose source is `inflight` and that has the `live` probe. */
export function asLiveRecord(text: string): FlightRecord {
	let raw: unknown;
	try {
		raw = JSON.parse(text);
	} catch {
		throw new Error(t("live.error.not_json"));
	}
	const rec = raw as Partial<FlightRecord> | null;
	if (!rec || typeof rec !== "object" || rec.job?.source?.kind !== "inflight" || !rec.probes?.some((p) => p.id === "live")) {
		throw new Error(t("live.error.not_live"));
	}
	return rec as FlightRecord;
}

/** True when this file is a live record (now on screen). */
async function read(file: File, opening = false): Promise<boolean> {
	if (!opening && state.modified === file.lastModified && state.name === file.name && state.record) return true;
	try {
		const record = asLiveRecord(await file.text());
		set({ record, name: file.name, modified: file.lastModified, error: null });
		return true;
	} catch (e) {
		// a file caught mid-write keeps the last good copy on screen (A writes it atomically, so this is rare)
		if (!opening) set({ error: e instanceof Error ? e.message : String(e) });
		return false;
	}
}

/** A file from an <input type="file">: read once. */
export async function openLiveFile(file: File): Promise<boolean> {
	stopFollowing();
	return read(file, true);
}

/** The browser can keep a handle on a picked file (Chromium): checked before any await, so a
 * fallback file input still opens inside the visitor's click. */
export function canFollow(): boolean {
	return typeof (window as unknown as { showOpenFilePicker?: unknown }).showOpenFilePicker === "function";
}

/** Pick the file with a handle the page can read again (Chromium): "opened", "closed" (the
 * visitor closed the picker) or "wrong" (not a live record; nothing changes). */
export async function pickAndFollow(): Promise<"opened" | "closed" | "wrong"> {
	const picker = (window as unknown as { showOpenFilePicker: (o: unknown) => Promise<FileHandle[]> }).showOpenFilePicker;
	let picked: FileHandle[];
	try {
		picked = await picker({ types: [{ description: t("live.picker"), accept: { "application/json": [".json"] } }], multiple: false });
	} catch {
		return "closed";
	}
	const h = picked[0];
	if (!h || !(await read(await h.getFile(), true))) return "wrong";
	stopFollowing();
	handle = h;
	set({ following: true });
	timer = setInterval(() => {
		const cur = handle;
		if (!cur) return;
		cur.getFile().then(
			(f) => void read(f),
			() => {
				stopFollowing(); // moved or deleted: stop asking
				set({ error: t("live.error.gone") });
			},
		);
	}, POLL_MS);
	return "opened";
}

export function stopFollowing(): void {
	if (timer) clearInterval(timer);
	timer = null;
	handle = null;
	if (state.following) set({ following: false });
}

export function liveRecord(): FlightRecord | null {
	return state.record;
}

/** For tests. */
export function resetLive(): void {
	stopFollowing();
	state = EMPTY;
	for (const l of listeners) l();
}

function subscribe(fn: () => void): () => void {
	listeners.add(fn);
	return () => listeners.delete(fn);
}

export function useLive(): LiveState {
	return useSyncExternalStore(subscribe, () => state, () => EMPTY);
}
