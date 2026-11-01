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
export type BoardingEvent = JobEvent & { share_token?: string | null; error_code?: string | null };

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
	return null;
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
	return { last, done, failed, cancelled, open, error, at };
}
