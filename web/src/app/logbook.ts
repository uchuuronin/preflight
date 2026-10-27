/**
 * The logbook: flights this browser has opened, kept in localStorage only (no account).
 * Storage can be missing or throw (private windows, blocked site data), so every access is
 * guarded and the page works without it.
 */
import type { LoadedRecord } from "../workspace/record/types";

const KEY = "pf.logbook.v1";
const MAX = 40;

export interface LogEntry {
	ref: string;
	title: string;
	lens: string;
	gpu: string | null;
	openedAt: string;
	/** What Preflight caught, from the record's worst finding. */
	caught: string | null;
	worst: "info" | "warn" | "error" | "critical" | null;
}

export function readLogbook(): LogEntry[] {
	try {
		const raw = localStorage.getItem(KEY);
		if (!raw) return [];
		const parsed = JSON.parse(raw) as unknown;
		return Array.isArray(parsed) ? (parsed as LogEntry[]) : [];
	} catch {
		return [];
	}
}

const RANK = { info: 0, warn: 1, error: 2, critical: 3 } as const;

export function logFlight(ref: string, record: LoadedRecord): void {
	try {
		const worst = [...record.findings].sort((a, b) => RANK[b.severity] - RANK[a.severity])[0];
		const entry: LogEntry = {
			ref,
			title: record.job.title ?? record.job.lens,
			lens: record.job.lens,
			gpu: record.env?.gpu?.name ?? (record.env?.backend === "api" ? "API" : null),
			openedAt: new Date().toISOString(),
			caught: worst?.title ?? null,
			worst: worst?.severity ?? null,
		};
		const rest = readLogbook().filter((e) => e.ref !== ref);
		localStorage.setItem(KEY, JSON.stringify([entry, ...rest].slice(0, MAX)));
	} catch {
		// Storage unavailable: the logbook simply stays empty.
	}
}

export function clearLogbook(): void {
	try {
		localStorage.removeItem(KEY);
	} catch {
		// nothing to clear
	}
}
