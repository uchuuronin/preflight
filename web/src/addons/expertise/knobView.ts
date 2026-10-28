/**
 * The expertise dial's view over C's knob dock (registerKnobView). At "New to this" it shows up
 * to five knobs (the failure's, the top two for the visitor's current goal from C's `ranked`,
 * then the lens baseline; tiers.ts) plus anything the visitor has opened or moved this visit.
 * Warnings' knobs stay in the dock until C's dock lists the warnings (`findingsShown`); after that
 * a warning's knob shows once opened. Practitioners and experts see every knob. C's dock also keeps
 * pinned, guide-highlighted, open and moved knobs whatever a view returns.
 */
import { hasKey } from "../../lib/i18n";
import type { KnobViewContext, KnobViewEntry } from "../../registry";
import type { Finding, Knob, LoadedRecord } from "../../workspace/record/types";
import { getTier } from "./store";
import { POLICY, type Policy, labelKeyFor, visibleKnobs } from "./tiers";
import { worthChecking, worthCheckingKey } from "./worthChecking";

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

/**
 * What D's view adds for C's dock, per the request in docs/requests/from-d.md: the warnings to list
 * under the row and the list's heading. C's dock sets `findingsShown` on the context in a render
 * where it lists them; until it does, D keeps the warnings' knobs in the dock.
 */
export interface ExpertiseKnobView extends KnobViewEntry {
	findings: (ctx: KnobViewContext) => Finding[];
	findingsKey: (n: number) => string;
}

/** Whether C's dock lists this view's findings in this render (absent until C ships it). */
export function findingsShown(ctx: KnobViewContext): boolean {
	return (ctx as KnobViewContext & { findingsShown?: boolean }).findingsShown === true;
}

export function createKnobView(policy: Policy = POLICY): ExpertiseKnobView {
	// Knobs moved this visit, per record, so one put back stays where the visitor left it; and
	// knobs opened this visit (a warning's "open" button, say), so one revealed stays revealed.
	const touched = new Map<string, Set<string>>();
	const opened = new Map<string, Set<string>>();

	function setFor(store: Map<string, Set<string>>, ctx: KnobViewContext): Set<string> {
		const key = recordKey(ctx.record);
		let set = store.get(key);
		if (!set) {
			set = new Set();
			store.set(key, set);
		}
		return set;
	}

	function touchedFor(ctx: KnobViewContext): Set<string> {
		const set = setFor(touched, ctx);
		for (const k of ctx.record.knobs) {
			if (k.id in ctx.knobs && ctx.knobs[k.id] !== k.current) set.add(k.id);
		}
		return set;
	}

	function openedFor(ctx: KnobViewContext): Set<string> {
		const set = setFor(opened, ctx);
		if (ctx.active) set.add(ctx.active);
		return set;
	}

	return {
		id: "expertise.level",
		order: 10,
		moreKey: "addon.expertise.more",
		findings: (ctx: KnobViewContext) => worthChecking(ctx.record),
		findingsKey: worthCheckingKey,
		filter(knobs: Knob[], ctx: KnobViewContext): Knob[] {
			const moved = touchedFor(ctx);
			const seen = openedFor(ctx);
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
					opened: seen,
					failing: failingKnobs(ctx.record),
					warnings: new Set(worthChecking(ctx.record).flatMap((f) => (f.knob_id ? [f.knob_id] : []))),
					warningsListed: findingsShown(ctx),
					ranked: ctx.ranked,
				}, policy),
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
