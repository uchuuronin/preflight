/**
 * What the boarding screen shows, from the job's events (pure, so it's tested without a stream).
 *
 * A record is read by its share token (GET /v1/records/{share_token}); A sends the token with the
 * first `skeleton` event, and later events only bump `record_rev`. A failure is shown by its
 * stable `error_code` (t('error.<code>'), falling back to the family), and A's own visitor-safe
 * message, which says what to change, is the hint.
 */
import { errorCopy, eventText, type ErrorCopy } from "../lib/errors";
import type { JobEvent } from "../workspace/record/types";

/** JobEvent plus A's additive fields (contract "Unreleased": share_token, error_code). */
export type BoardingEvent = JobEvent & {
	share_token?: string | null;
	error_code?: string | null;
	/** Contract 0.6.5 (A, behind runner.free_resend): the queued event of a resend that didn't count against the day's check-ins. */
	free_resend?: boolean | null;
};

/**
 * The route's five legs, in order, and the job stages each covers (A's JobEvent.stage). `calls`
 * names the model-call purposes (the record's provenance) that belong to a leg: its model tag
 * shows only once the record says such a call was made (audit finding 12).
 */
export const STAGES: Array<{ stage: JobEvent["stage"][]; key: string; calls?: string[] }> = [
	{ stage: ["queued", "admitted"], key: "boarding.stage.queue" },
	{ stage: ["fetching", "mapping", "skeleton"], key: "boarding.stage.read", calls: ["script_map"] },
	{ stage: ["harnessing", "probing"], key: "boarding.stage.probe" },
	{ stage: ["triage"], key: "boarding.stage.triage", calls: ["triage", "claim_check", "diagnosis"] },
	{ stage: ["predicting", "done"], key: "boarding.stage.fit" },
];

/** A model call as the record's provenance keeps it (schema ModelCall). */
export interface ModelCallSeen {
	provider?: string;
	model?: string;
	purpose?: string;
	ok?: boolean;
}

/** The on-screen name of a Meta model, from its id; null for any other model. */
export function metaModelName(model: string | undefined): string | null {
	const m = (model ?? "").toLowerCase();
	if (m.includes("muse-spark-open")) return "Muse Spark Open";
	if (m.includes("muse-spark")) return "Muse Spark";
	if (m.includes("muse-glimmer")) return "Muse Glimmer";
	return null;
}

/** What a model call did, by its purpose (A's and B's `ModelCall.purpose` values), as words for the record's header. */
const DID: Readonly<Record<string, string>> = {
	script_map: "models.did.script_map",
	triage: "models.did.triage",
	diagnosis: "models.did.diagnosis",
	diagnosis_check: "models.did.diagnosis_check",
	claim_check: "models.did.claim_check",
	search: "models.did.search",
	guide: "models.did.guide",
	patch: "models.did.patch",
	command: "models.did.command",
};

/**
 * The Meta models a record's own model calls show (its provenance, successful calls only): each with
 * what it did, in the order first called, and what those calls cost in all (PM, Nov 3: "Meta models on
 * this record"). Null when no Meta model was called.
 */
export function metaModelsOnRecord(provenance: ReadonlyArray<ModelCallSeen & { cost_usd?: number }> | undefined): { models: Array<{ name: string; did: string[] }>; usd: number } | null {
	const by = new Map<string, Set<string>>();
	let usd = 0;
	for (const c of provenance ?? []) {
		const name = metaModelName(c.model);
		if (!name) continue;
		// A failed call is billed too (a timed-out Glimmer call by its seconds): counted in the cost, not in what it did.
		usd += typeof c.cost_usd === "number" && Number.isFinite(c.cost_usd) ? c.cost_usd : 0;
		if (c.ok === false) continue;
		if (!by.has(name)) by.set(name, new Set());
		const key = c.purpose ? DID[c.purpose] : undefined;
		if (key) by.get(name)!.add(key);
	}
	if (!by.size) return null;
	return { models: [...by].map(([name, did]) => ({ name, did: [...did] })), usd };
}

/** Per leg, the Meta models the record shows were actually called for it (successful calls only). */
export function modelsRan(provenance: readonly ModelCallSeen[] | undefined): string[][] {
	return STAGES.map((s) => {
		const names = (provenance ?? [])
			.filter((c) => c.ok !== false && c.purpose && s.calls?.includes(c.purpose))
			.map((c) => metaModelName(c.model))
			.filter((n): n is string => !!n);
		return [...new Set(names)];
	});
}

const TERMINAL: ReadonlySet<string> = new Set(["done", "failed", "cancelled"]);

export interface BoardingState {
	last: BoardingEvent | undefined;
	done: boolean;
	failed: boolean;
	cancelled: boolean;
	/** The share token to open the cockpit on, once one has arrived. */
	open: string | null;
	error: ErrorCopy | null;
	/** The leg the job is on, or the one it stopped on after a failure or cancel (index into STAGES). */
	at: number;
	/** What the runner last said about the leg it's on (A, Nov 2: "waiting for a free L4 GPU (3 min so far)"), while it's on it. */
	note: string | null;
	/** The job is a free resend: the one before it stopped at an import, before measuring anything (A's 0.6.5). */
	free: boolean;
}

/**
 * The job as GET /v1/jobs/{id} reports it, turned into the event the stream would have sent at its
 * end, for when the stream is lost (Nov 2: a check-in's boarding page lost its stream and never
 * learned the job was done, so "Open the cockpit" stayed off). Null while the job is still going.
 */
export function eventFromStatus(st: { status?: string; share_token?: string | null; error_code?: string | null; error?: string | null } | null): BoardingEvent | null {
	if (!st || !st.status || !TERMINAL.has(st.status)) return null;
	return { stage: st.status as JobEvent["stage"], share_token: st.share_token ?? null, error_code: st.error_code ?? null, message: st.error ?? null } as BoardingEvent;
}

export function boardingState(events: readonly BoardingEvent[]): BoardingState {
	const last = events[events.length - 1];
	let open: string | null = null;
	for (const e of events) if (e.share_token) open = e.share_token;
	const done = last?.stage === "done";
	const failed = last?.stage === "failed" || last?.stage === "cancelled";
	const cancelled = last?.stage === "cancelled";
	let error: ErrorCopy | null = null;
	// A plain cancel (the visitor's own, or a deadline) has no code and no message: it isn't an
	// error to explain, the heading says it.
	if (failed && last && !(cancelled && !last.error_code && !last.message)) {
		const said = eventText(last.message);
		const copy = errorCopy(last.error_code, said?.text);
		const hint = said && said.text !== copy.message && !last.message?.startsWith("error.") ? said.text : copy.hint;
		error = { ...copy, hint };
	}
	// A failed or cancelled event carries no leg of its own: the job stopped on the leg of the last
	// stage it reached (the queue, if it never got further).
	let at = -1;
	for (const e of events) {
		if (TERMINAL.has(e.stage) && e.stage !== "done") continue;
		const i = STAGES.findIndex((s) => s.stage.includes(e.stage));
		if (i >= 0) at = i;
	}
	if (failed && at < 0) at = 0;
	// The runner's own words on the leg it's on (a GPU wait, say); gone once a later event moves on.
	const note = last && !TERMINAL.has(last.stage) && last.message ? (eventText(last.message)?.text ?? null) : null;
	const free = events.some((e) => e.stage === "queued" && e.free_resend === true);
	return { last, done, failed, cancelled, open, error, at, note, free };
}
