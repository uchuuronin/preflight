/**
 * Why a readout has no number, with the record in view (Chiya's shot 9b, Nov 3): a record opened from
 * disk never reaches the server, so the engine can't fill a setting its grids don't hold. There an
 * off-grid cell names the setting and the two ways to get its number: check the job in on the site,
 * or measure that setting with `preflight run`.
 */
import { t } from "../../lib/i18n";
import { isLocalRef } from "../provider/engineFallback";
import type { KnobValue, LoadedRecord } from "../record/types";
import { useWorkspace } from "../state";
import { abstainText } from "./abstain";

const OFF_GRID = new Set(["off_grid", "between_measured"]);
const MAX_ARGS = 3;

/** The settings that differ from the record's own, as script arguments ("--batch_size 32"). Code knobs can't be passed. */
export function changedArgs(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): string[] {
	const out: string[] = [];
	for (const k of record.knobs) {
		if (k.needs_code || !(k.id in knobs) || knobs[k.id] === k.current) continue;
		const v = knobs[k.id];
		if (typeof v === "boolean") {
			if (v) out.push(`--${k.id}`);
			continue;
		}
		out.push(`--${k.id} ${String(v)}`);
	}
	return out.slice(0, MAX_ARGS);
}

export function localOffGridText(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): string {
	const args = changedArgs(record, knobs);
	return args.length ? t("abstain.local", { cmd: `preflight run … -- ${args.join(" ")}` }) : t("abstain.local_plain");
}

/** abstainText, but a local record's off-grid cell says what to do there. */
export function useWhyAbstain(): (reason: string | null | undefined) => string {
	const { record, recordRef, state } = useWorkspace();
	return (reason) => (isLocalRef(recordRef) && OFF_GRID.has(String(reason ?? "")) ? localOffGridText(record, state.knobs) : abstainText(reason));
}
