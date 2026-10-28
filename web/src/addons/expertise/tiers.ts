/**
 * Which knobs each detail level shows, and how they are named.
 *
 * Practitioner and expert see every knob. "New to this" picks at most MAX_NEW_KNOBS by itself, in
 * this order: the knobs behind the current failure (error and critical findings), the
 * NEW_TOP_SENSITIVE knobs that matter most for the visitor's current goal (sensitivity()), then
 * the lens baseline (the knobs a newcomer recognises). Warnings' knobs are not picked once C's
 * dock lists the warnings ("things worth checking"), and a warning's knob shows when the visitor
 * opens it; until that list is on screen they stay in the dock like failures. On top of
 * the five, whatever the visitor opened, moved or pinned and whatever the guide is on stays.
 * A failure's knob is never dropped to keep under the cap. docs/decisions/expertise-new-level.md;
 * dial.zoo.test.ts checks it against the zoo's planted problems.
 */
import type { Lens } from "../../workspace/record/types";
import census from "./census.json";
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
 * Knobs a newcomer recognises, per lens, most common first; the cap decides how many show.
 * Training lenses come from census.json [measured: settings set per example across the bug zoo,
 * TRL, transformers, PEFT and nanoGPT examples, built by python/preflight/addons/knob_census.py;
 * counts and sources in docs/decisions/expertise-new-level.md]. Hardware (gpu_type) is not a
 * script setting, so the census can't rank it.
 */
const CENSUSED = census.lenses as Record<string, { ranking: string[] }>;

/**
 * Lenses with no training examples to count.
 * [guess: from the Users and views tab; replace with the knobs first-time visitors open first,
 * from session telemetry after Nov 4]
 */
const GUESSED: Partial<Record<Lens, readonly string[]>> = {
	agent: ["model", "n_tasks", "concurrency"],
	eval: ["model", "gpu_type", "n_tasks"],
	serving: ["gpu_type", "max_model_len", "concurrency"],
};

export const BASELINE: Partial<Record<Lens, readonly string[]>> = {
	...GUESSED,
	...Object.fromEntries(Object.entries(CENSUSED).map(([lens, row]) => [lens, row.ranking])),
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
	/** Knobs behind a warning (warn findings). */
	warnings: ReadonlySet<string>;
	/** Whether the warnings are listed on screen; if not, their knobs stay like failures. */
	warningsListed: boolean;
	/** Knob ids by impact on the current objective, most first (sensitivity()). */
	ranked: readonly string[];
}

/** Where each group claims places in the five; "must" = failures, and warnings nobody can see listed. */
export type Claim = "must" | "top" | "baseline" | "rest";

export interface Policy {
	/** How many of the goal's top knobs claim a place before the baseline. */
	top: number;
	order: readonly Claim[];
}

/**
 * [guess: failures first is the review's rule; top before baseline follows "rank by the active
 * goal". Settled by the comparison in dial.zoo.test.ts when B's grids land (Wed Oct 28): the
 * policy that shows the planted fixes most often wins, and that test fails if another does.]
 */
export const POLICY: Policy = { top: NEW_TOP_SENSITIVE, order: ["must", "top", "baseline", "rest"] };

export function visibleKnobs(input: VisibilityInput, policy: Policy = POLICY): string[] {
	const { lens, tier, knobIds, pinned, highlighted, touched, opened, ranked } = input;
	if (tier !== "new") return [...knobIds];
	const must = new Set([...input.failing, ...(input.warningsListed ? [] : input.warnings)]);
	const groups: Record<Claim, readonly string[]> = {
		must: knobIds.filter((id) => must.has(id)),
		top: ranked.slice(0, policy.top),
		baseline: BASELINE[lens] ?? [],
		rest: ranked.slice(policy.top),
	};
	const declared = new Set(knobIds);
	const show = new Set<string>();
	for (const id of policy.order.flatMap((g) => groups[g])) {
		if (!declared.has(id) || show.has(id)) continue;
		if (show.size >= MAX_NEW_KNOBS && !must.has(id)) continue;
		show.add(id);
	}
	for (const id of must) if (declared.has(id)) show.add(id);
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
