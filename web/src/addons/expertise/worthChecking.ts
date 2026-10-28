/**
 * "Things worth checking" at "New to this": the record's warnings, listed as findings instead of
 * their knobs taking a place in the five. Opening one shows its knob (knobView.ts keeps a knob
 * once opened). The list renders in C's findings display, under the dock, once C offers that
 * spot (docs/requests/from-d.md); this module is the pure part both sides share.
 */
import type { Finding, LoadedRecord } from "../../workspace/record/types";

/** The record's warn findings, in the record's order; knob_id is null when the record doesn't declare it. */
export function worthChecking(record: LoadedRecord): Finding[] {
	const declared = new Set(record.knobs.map((k) => k.id));
	return record.findings
		.filter((f) => f.severity === "warn")
		.map((f) => ({ ...f, knob_id: f.knob_id && declared.has(f.knob_id) ? f.knob_id : null }));
}

/** The i18n key for the list's heading, with {n} = how many. */
export function worthCheckingKey(n: number): string {
	return n === 1 ? "addon.expertise.worth_checking.one" : "addon.expertise.worth_checking.other";
}
