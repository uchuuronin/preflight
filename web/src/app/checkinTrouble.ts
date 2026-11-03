/**
 * When a check-in stops while reading the code or installing what it imports, which step it was,
 * what's missing, and the way past it (PM, Nov 2: the unseen runs, 0 of 9 checked in: 7 stopped at
 * an import, 2 repos were over the read limit, 1 over the map's).
 *
 * Read from A's fields where they exist (proposed in docs/requests/from-c.md: `submission.too_large`,
 * `map.too_large`, `Probe.missing_module` and `missing_package`), else from today's messages, so the
 * page explains these stops now. The ways past need A's runner to take `extra_packages` and
 * `source.scope`: behind `c.checkin_help` until it does; without it the page says what to change by
 * hand. Also keeps each check-in's request for this visit, so a way past can send it again.
 */
import type { JobRequest } from "../workspace/record/types";

/** A probe as far as this reads it (record schema Probe, plus A's proposed fields). */
export interface ProbeSeen {
	id?: string;
	status?: string;
	error_code?: string | null;
	error?: string | null;
	missing_module?: string | null;
	missing_package?: string | null;
	steps?: number | null;
}

export type Trouble =
	/** The repo is over what Preflight reads (A's runner: 2,000 files, 25 MB of text). */
	| { step: "read"; limit: "repo" }
	/** The Python is over what the script map reads (2 MB). */
	| { step: "read"; limit: "map" }
	/** The script imports packages the probe doesn't have. `pkg` is the name Preflight installs it by, when it's on its list. */
	| { step: "install"; missing: Array<{ module: string; pkg: string | null; noWheels?: boolean }> }
	/**
	 * The job finished but no probe measured anything (no GPU in time, the repo's own module, a
	 * notebook-only call, a too-old torch, ...): `code` is the stop they share, when they share one
	 * (PM, Nov 3: a check-in whose probes never ran must not look finished).
	 */
	| { step: "measure"; code: string | null };

/**
 * A probe that measured something: it ran (ok), its numbers came from an earlier identical run (cached),
 * or it ran out of memory, which is the answer at that setting (B's engine reads it as "fails here").
 * Any probe that took steps measured them too (review, Nov 3).
 */
const MEASURED = new Set(["ok", "cached", "oom"]);
/** Still going (a live, watched run): not finished, so not "nothing measured". */
const GOING = new Set(["queued", "running"]);

/**
 * A finished job's record where no probe measured anything, and why (the stop they share). Null when
 * any probe measured, or there are no probes. The import stop with its own way past is
 * `troubleFromProbes`; this is every other way a check-in ends with nothing measured.
 */
/** Stops an identical resend can get past: a GPU that may be free now, a run that was cut short. */
export const TRANSIENT_STOPS = new Set(["gpu.unavailable", "timeout", "killed"]);

export function nothingMeasured(probes: readonly ProbeSeen[] | undefined, doneCode?: string | null): Trouble | null {
	const all = probes ?? [];
	if (!all.length || all.some((p) => MEASURED.has(p.status ?? "") || GOING.has(p.status ?? "") || (p.steps ?? 0) > 0)) return null;
	const codes = new Set(all.map((p) => p.error_code ?? null));
	const shared = codes.size === 1 ? [...codes][0]! : null;
	return { step: "measure", code: doneCode ?? shared };
}

/** A failed job's stop, from its error code and message. */
export function troubleFromFailure(code: string | null | undefined, message: string | null | undefined): Trouble | null {
	const m = message ?? "";
	if (code === "submission.too_large" || (code?.startsWith("submission") && /larger than Preflight reads/i.test(m))) return { step: "read", limit: "repo" };
	if (code === "map.too_large" || /too large to map/i.test(m)) return { step: "read", limit: "map" };
	return null;
}

const IMPORT_NAME = /^[A-Za-z_][A-Za-z0-9_.]*$/;

/** The import name in a probe's missing-package message, or null. Only the recorder's "The script
 * imports X" names an import (recorder/errors.py); its "A library in the script needs X" form comes
 * from a library's own words, which can be anything ("needs the protobuf library"), so it isn't read. */
function moduleIn(p: ProbeSeen): string | null {
	if (p.missing_module && IMPORT_NAME.test(p.missing_module)) return p.missing_module;
	const m = /The script imports ([A-Za-z_][A-Za-z0-9_.]*), which is not installed/.exec(p.error ?? "");
	return m ? m[1]! : null;
}

/**
 * A finished job whose probes all stopped at a missing import: nothing was measured, and the way
 * past is installing it. Null when any probe ran, or when a probe stopped for another reason.
 */
export function troubleFromProbes(
	probes: readonly ProbeSeen[] | undefined,
	known: Readonly<Record<string, string>> = {},
	noWheels: readonly string[] = [],
): Trouble | null {
	const all = probes ?? [];
	// Every probe stopped at an import (the runner runs each point, so each stops the same way): a
	// run where any probe measured something, or stopped for another reason, isn't this stop (review, Nov 2).
	if (!all.length || !all.every((p) => p.error_code === "missing_package")) return null;
	const stopped = all;
	const seen = new Map<string, string | null>();
	for (const p of stopped) {
		const mod = moduleIn(p);
		if (!mod || seen.has(mod)) continue;
		const top = mod.split(".")[0]!;
		seen.set(mod, noWheels.includes(top) ? null : (p.missing_package ?? known[top] ?? known[mod] ?? null));
	}
	if (!seen.size) return null;
	return { step: "install", missing: [...seen].map(([module, pkg]) => (noWheels.includes(module.split(".")[0]!) ? { module, pkg, noWheels: true } : { module, pkg })) };
}

/** A package name the box accepts: a PyPI project name, nothing else (no versions, URLs or flags). */
export const PACKAGE_NAME = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$/;
export const MAX_EXTRA_PACKAGES = 10;

/** The box's text as package names: split on commas and spaces, deduplicated; bad names are returned apart. */
export function parsePackages(text: string): { names: string[]; bad: string[] } {
	const parts = text.split(/[\s,]+/).filter(Boolean);
	const names: string[] = [];
	const bad: string[] = [];
	for (const p of parts) {
		if (!PACKAGE_NAME.test(p)) bad.push(p);
		else if (!names.some((n) => n.toLowerCase() === p.toLowerCase())) names.push(p);
	}
	return { names: names.slice(0, MAX_EXTRA_PACKAGES), bad };
}

/** Each check-in's request, by job id, for this visit (a way past sends it again with a change). */
const sent = new Map<string, JobRequest>();

/** Only the last few: a request can hold up to 5 MB of uploaded files. */
const KEEP = 3;

export function rememberRequest(jobId: string, body: JobRequest): void {
	sent.delete(jobId);
	sent.set(jobId, body);
	while (sent.size > KEEP) sent.delete(sent.keys().next().value!);
}

export function requestFor(jobId: string | null): JobRequest | null {
	return jobId ? (sent.get(jobId) ?? null) : null;
}

/** The request again, installing these packages too. */
export function withPackages(body: JobRequest, names: readonly string[]): JobRequest {
	return { ...body, extra_packages: [...names] } as JobRequest;
}

/** The request again, reading only this file and what it imports. */
export function fromFile(body: JobRequest, entrypoint: string): JobRequest {
	return { ...body, source: { ...(body.source as Record<string, unknown>), entrypoint, scope: "entrypoint" } } as JobRequest;
}
