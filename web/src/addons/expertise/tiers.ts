/**
 * Which knobs each detail level shows, and how they are named.
 *
 * Practitioner and expert see every knob. "New to this" sees the lens baseline (the knobs a
 * newcomer recognises), plus the NEW_TOP_SENSITIVE knobs that matter most right now by
 * sensitivity() for the visitor's current goal, the knobs behind the current failure and any the
 * visitor touched (knobView.ts). docs/decisions/expertise-new-level.md; dial.zoo.test.ts checks it
 * against the zoo's planted problems.
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
 * Most knobs the new level picks by itself when a lens has no baseline.
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
	/** Knobs behind a current failure (error or critical findings), on the grid or not. */
	failing: ReadonlySet<string>;
	/** Knob ids by impact on the current objective, most first (sensitivity()). */
	ranked: readonly string[];
}

export function visibleKnobs(input: VisibilityInput): string[] {
	const { lens, tier, knobIds, pinned, highlighted, touched, failing, ranked } = input;
	if (tier !== "new") return [...knobIds];
	const baseline = BASELINE[lens];
	const show = new Set<string>(baseline ?? []);
	for (const id of ranked.slice(0, baseline ? NEW_TOP_SENSITIVE : MAX_NEW_KNOBS)) show.add(id);
	for (const id of pinned) show.add(id);
	for (const id of highlighted) show.add(id);
	for (const id of touched) show.add(id);
	for (const id of failing) show.add(id);
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
