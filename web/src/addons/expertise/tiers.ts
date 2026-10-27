/**
 * Which knobs each detail level shows, and how they are named.
 *
 * Practitioner and expert see every knob. "New to this" sees the lens baseline (the knobs a
 * newcomer recognises), plus the NEW_TOP_SENSITIVE knobs that matter most right now by
 * sensitivity(), plus anything pinned or under the guide's hand, so the guide never moves a knob
 * the person can't see. The count was chosen by a Monte Carlo on the sample records
 * (docs/decisions/expertise-new-level.md); tiers.test.ts keeps it honest.
 *
 * Live once C adds the knob-view hook (docs/requests.md); the dock shows every knob until then.
 */
import type { Lens } from "../../workspace/record/types";
import type { Tier } from "./store";

/** Top knobs by impact added at the new level: the smallest count that always shows the fix. */
export const NEW_TOP_SENSITIVE = 2;

/** Most knobs the new level puts on screen, pins and guide highlights aside. */
export const MAX_NEW_KNOBS = 5;

/** Knobs a newcomer recognises, per lens. Lenses without a list fall back to the top knobs. */
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
	/** Knob ids by impact on the current objective, most first (sensitivity()). */
	ranked: readonly string[];
}

export function visibleKnobs(input: VisibilityInput): string[] {
	const { lens, tier, knobIds, pinned, highlighted, ranked } = input;
	if (tier !== "new") return [...knobIds];
	const baseline = BASELINE[lens];
	const show = new Set<string>(baseline ?? []);
	for (const id of ranked.slice(0, baseline ? NEW_TOP_SENSITIVE : MAX_NEW_KNOBS)) show.add(id);
	for (const id of pinned) show.add(id);
	for (const id of highlighted) show.add(id);
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
