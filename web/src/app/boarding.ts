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

export interface BoardingState {
	last: BoardingEvent | undefined;
	done: boolean;
	failed: boolean;
	cancelled: boolean;
	/** The share token to open the cockpit on, once one has arrived. */
	open: string | null;
	error: ErrorCopy | null;
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
	return { last, done, failed, cancelled, open, error };
}
