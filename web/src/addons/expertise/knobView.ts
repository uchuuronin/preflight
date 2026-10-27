/**
 * The expertise dial's view over C's knob dock (registerKnobView). At "New to this" it shows
 * the lens baseline, the top knobs for the visitor's current goal (C's `ranked`, from
 * sensitivity() on the active objective), the knobs behind the current failure and anything the
 * visitor has touched; practitioners and experts see every knob. C's dock also keeps pinned,
 * guide-highlighted, open and moved knobs whatever a view returns.
 */
import { hasKey } from "../../lib/i18n";
import type { KnobViewContext, KnobViewEntry } from "../../registry";
import type { Knob, LoadedRecord } from "../../workspace/record/types";
import { getTier } from "./store";
import { labelKeyFor, visibleKnobs } from "./tiers";

const FAILING = new Set(["error", "critical"]);

/** Knob ids named by the record's error and critical findings, on the grid or not. */
export function failingKnobs(record: LoadedRecord): Set<string> {
	const declared = new Set(record.knobs.map((k) => k.id));
	const out = new Set<string>();
	for (const f of record.findings) {
		if (FAILING.has(f.severity) && f.knob_id && declared.has(f.knob_id)) out.add(f.knob_id);
	}
	return out;
}

function recordKey(record: LoadedRecord): string {
	return record.record_id ?? record.share_token ?? "record";
}

export function createKnobView(): KnobViewEntry {
	// Knobs moved this visit, per record, so one put back stays where the visitor left it.
	const touched = new Map<string, Set<string>>();

	function touchedFor(ctx: KnobViewContext): Set<string> {
		const key = recordKey(ctx.record);
		let set = touched.get(key);
		if (!set) {
			set = new Set();
			touched.set(key, set);
		}
		for (const k of ctx.record.knobs) {
			if (k.id in ctx.knobs && ctx.knobs[k.id] !== k.current) set.add(k.id);
		}
		return set;
	}

	return {
		id: "expertise.level",
		order: 10,
		moreKey: "addon.expertise.more",
		filter(knobs: Knob[], ctx: KnobViewContext): Knob[] {
			const moved = touchedFor(ctx);
			const tier = getTier();
			if (tier !== "new") return knobs;
			const ids = new Set(
				visibleKnobs({
					lens: ctx.lens,
					tier,
					knobIds: knobs.map((k) => k.id),
					pinned: ctx.pinned,
					highlighted: ctx.highlight,
					touched: moved,
					failing: failingKnobs(ctx.record),
					ranked: ctx.ranked,
				}),
			);
			return knobs.filter((k) => ids.has(k.id));
		},
		labelKey(knob: Knob): string {
			return labelKeyFor(knob.id, knob.label_key, getTier(), KNOWN);
		},
	};
}

/** Keys t() can resolve; C's dock also falls back to the record's label for unknown keys. */
const KNOWN: ReadonlySet<string> = { has: hasKey } as unknown as ReadonlySet<string>;
