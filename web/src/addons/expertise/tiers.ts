/**
 * Which knobs each detail level shows, and how they are named.
 *
 * Practitioner and expert see every knob. "New to this" picks at most MAX_NEW_KNOBS by itself, in
 * this order: the knobs behind the current failure (error and critical findings), the
 * NEW_TOP_SENSITIVE knobs that matter most for the visitor's current goal (sensitivity()), then
 * the lens baseline (the knobs a newcomer recognises). Warnings' knobs are not picked: the
 * warnings are listed as findings, and a warning's knob shows once the visitor opens it. On top of
 * the five, whatever the visitor opened, moved or pinned and whatever the guide is on stays.
 * A failure's knob is never dropped to keep under the cap. docs/decisions/expertise-new-level.md;
 * dial.zoo.test.ts checks it against the zoo's planted problems.
 */
import type { Lens } from "../../workspace/record/types";
import type { Tier } from "./store";

/**
 * Top knobs for the current goal added at the new level.
 * [guess: replace with the count that maximises planted-problem coverage in dial.zoo.test.ts on
 * B's real grids, once make_golden.py runs (B, Wed Oct 28)]
 */
export const NEW_TOP_SENSITIVE = 2;

/**
 * Most knobs the new level picks by itself.
 * [cited: Cowan 2001, "The magical number 4 in short-term memory": about 4 chunks;
 * 5 is the upper end of 4 ± 1]
 */
export const MAX_NEW_KNOBS = 5;

/**
 * Knobs a newcomer recognises, per lens. Lenses without a list fall back to the top knobs.
 * [guess: from the Users and views tab; replace with the knobs first-time visitors open first,
 * from session telemetry after Nov 4]
 */
export const BASELINE: Partial<Record<Lens, readonly string[]>> = {
	finetune: ["gpu_type", "batch_size", "seq_len"],
	pretrain: ["batch_size", "seq_len", "max_steps"],
	agent: ["model", "n_tasks", "concurrency"],
	rl: ["group_size", "max_completion_length", "gpu_type"],
	eval: ["model", "gpu_type", "n_tasks"],
	serving: ["gpu_type", "max_model_len", "concurrency"],
	vision: ["gpu_type", "batch_size", "num_workers"],
};

export interface VisibilityInput {
	lens: Lens;
	tier: Tier;
	/** Every knob the record declares, in the record's order. */
	knobIds: readonly string[];
	pinned: ReadonlySet<string>;
	/** Knobs the guide is on right now. */
	highlighted: ReadonlySet<string>;
	/** Knobs the visitor has moved this visit, even if put back since. */
	touched: ReadonlySet<string>;
	/** Knobs the visitor has opened this visit, e.g. from a warning's "open" button. */
	opened: ReadonlySet<string>;
	/** Knobs behind a current failure (error or critical findings), on the grid or not. */
	failing: ReadonlySet<string>;
	/** Knob ids by impact on the current objective, most first (sensitivity()). */
	ranked: readonly string[];
}

export function visibleKnobs(input: VisibilityInput): string[] {
	const { lens, tier, knobIds, pinned, highlighted, touched, opened, failing, ranked } = input;
	if (tier !== "new") return [...knobIds];
	const declared = new Set(knobIds);
	const baseline = BASELINE[lens] ?? [];
	// Order of claim on the five [guess: rerun dial.zoo.test.ts on B's real grids to compare
	// orders; on A's fixtures the goal ranking is mostly empty, so every order ties].
	const candidates = [
		...knobIds.filter((id) => failing.has(id)),
		...ranked.slice(0, NEW_TOP_SENSITIVE),
		...baseline,
		...ranked.slice(NEW_TOP_SENSITIVE),
	];
	const show = new Set<string>();
	for (const id of candidates) {
		if (!declared.has(id) || show.has(id)) continue;
		if (show.size >= MAX_NEW_KNOBS && !failing.has(id)) break;
		show.add(id);
	}
	for (const set of [pinned, highlighted, touched, opened]) for (const id of set) show.add(id);
	return knobIds.filter((id) => show.has(id));
}

/**
 * The label to show for a knob at a level: D's plainer wording at "new" (or the code name at
 * "expert") when D has one, otherwise the record's own label_key.
 */
export function labelKeyFor(knobId: string, recordKey: string, tier: Tier, known: ReadonlySet<string>): string {
	if (tier === "practitioner") return recordKey;
	const key = `addon.expertise.label.${knobId}.${tier}`;
	return known.has(key) ? key : recordKey;
}
