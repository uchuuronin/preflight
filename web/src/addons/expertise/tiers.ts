// D1 · Expertise dial: which knobs show at each tier, and how they are worded.
//
// Provisional until the schema freezes (Tue Oct 27, noon ET):
// - Knob ids below are D's guesses from the Users and views lens table. When A
//   publishes the knob ids in python/preflight/schema, this file is updated to
//   match and these local types are replaced by the generated TS types.
// - Labels are string keys; English lives in strings.en.json until C confirms
//   how add-on strings reach t() (see docs/requests.md).

export type Tier = "new" | "practitioner" | "expert";

export type Lens =
	| "fine_tuning"
	| "pretraining"
	| "rl"
	| "agent"
	| "eval_sweep"
	| "serving"
	| "retrieval"
	| "gnn"
	| "vision";

export const TIERS: readonly Tier[] = ["new", "practitioner", "expert"];
export const DEFAULT_TIER: Tier = "practitioner";

// How many of the knobs ranked highest by B's sensitivity() are always shown
// at the "new" tier, on top of that lens's baseline list.
export const NEW_TIER_TOP_SENSITIVE = 3;

// Knobs shown at each tier, per lens. A tier shows its own list plus every
// list below it. Anything not listed shows only at "expert".
export const KNOBS_BY_TIER: Record<Lens, Record<Exclude<Tier, "expert">, readonly string[]>> = {
	fine_tuning: {
		new: ["gpu_type", "method", "seq_len", "batch_size"],
		practitioner: ["eval_batch_size", "grad_checkpointing", "lora_rank", "weight_bits", "packing"],
	},
	pretraining: {
		new: ["gpu_type", "gpu_count", "batch_size", "seq_len"],
		practitioner: ["precision", "torch_compile", "activation_checkpointing", "checkpoint_interval", "dataloader_workers"],
	},
	rl: {
		new: ["group_size", "completion_len", "gpu_type"],
		practitioner: ["vllm_generation", "batch_size", "learning_rate"],
	},
	agent: {
		new: ["model", "dataset_size", "concurrency"],
		practitioner: ["reasoning_effort", "max_steps", "prompt_caching"],
	},
	eval_sweep: {
		new: ["model", "gpu_type", "attempts"],
		practitioner: ["thinking_cap", "engine", "activation_layers"],
	},
	serving: {
		new: ["gpu_type", "max_context", "concurrency"],
		practitioner: ["quantisation", "tensor_parallel"],
	},
	retrieval: {
		new: ["gpu_type", "batch_size", "corpus_size"],
		practitioner: ["negatives_per_query", "embedding_dim", "index_type"],
	},
	gnn: {
		new: ["gpu_type", "fan_out", "hops"],
		practitioner: ["seed_batch", "sampler_workers"],
	},
	vision: {
		new: ["gpu_type", "batch_size", "dataloader_workers"],
		practitioner: ["precision", "gpu_augmentation", "channels_last"],
	},
};

export interface VisibilityInput {
	lens: Lens;
	tier: Tier;
	knobIds: readonly string[]; // every knob the record declares, in rail order
	pinned: ReadonlySet<string>;
	highlighted: ReadonlySet<string>; // knobs the guide is touring right now
	sensitivityRanked: readonly string[]; // from sensitivity(), most impact first
}

// Knobs to show, in the record's rail order. Three rules never bend:
// pinned knobs always show, knobs the guide highlights always show (so the
// tour never moves a knob the visitor can't see), and "expert" shows all.
export function visibleKnobs(input: VisibilityInput): string[] {
	const { lens, tier, knobIds, pinned, highlighted, sensitivityRanked } = input;
	if (tier === "expert") return [...knobIds];

	const lists = KNOBS_BY_TIER[lens];
	const allowed = new Set<string>(lists.new);
	if (tier === "practitioner") for (const id of lists.practitioner) allowed.add(id);
	if (tier === "new") {
		for (const id of sensitivityRanked.slice(0, NEW_TIER_TOP_SENSITIVE)) allowed.add(id);
	}
	for (const id of pinned) allowed.add(id);
	for (const id of highlighted) allowed.add(id);

	return knobIds.filter((id) => allowed.has(id));
}

// Number of knobs hidden at this tier, for the rail's "Show N more" control.
export function hiddenCount(input: VisibilityInput): number {
	return input.knobIds.length - visibleKnobs(input).length;
}

// Label key for a knob or readout at a tier. Falls back to the practitioner
// wording, then to the id, so a new knob from A never renders blank.
export function labelKey(id: string, tier: Tier, known: ReadonlySet<string>): string {
	const exact = `addon.expertise.label.${id}.${tier}`;
	if (known.has(exact)) return exact;
	const fallback = `addon.expertise.label.${id}.practitioner`;
	return known.has(fallback) ? fallback : id;
}
