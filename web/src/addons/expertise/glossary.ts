/**
 * Glossary for the expertise dial: one explanation per term at each detail level, and the
 * map from the record's knob ids to those terms. The text lives in glossary.en.json and is
 * registered into t() as addon.expertise.term.<term>.<label|new|practitioner|expert|act>.
 */
import content from "./glossary.en.json";
import type { Tier } from "./store";

export interface Term {
	label: string;
	new: string;
	practitioner: string;
	expert: string;
	/** What a person usually changes, when there is a clear fix. */
	act?: string;
}

export const GLOSSARY: Readonly<Record<string, Term>> = content.terms;

/**
 * Knob id (as declared in records) to glossary term. Ids follow A's golden records and B's
 * gpu_type request; RL, serving, retrieval, GNN and vision ids are added as A declares them.
 */
export const KNOB_TERMS: Readonly<Record<string, string>> = {
	// fine-tune and pretrain
	batch_size: "batch_size",
	seq_len: "sequence_length",
	grad_accum: "gradient_accumulation",
	eval_batch_size: "eval_batch",
	eval_no_grad: "eval_no_grad",
	lora_r: "lora_rank",
	gradient_checkpointing: "gradient_checkpointing",
	precision: "precision",
	packing: "packing",
	num_workers: "dataloader_workers",
	lr: "learning_rate",
	checkpoint_every: "checkpoint_interval",
	max_steps: "max_steps",
	gpu_type: "gpu_type",
	gpu: "gpu_type",
	torch_compile: "torch_compile",
	// agent
	model: "agent_model",
	prompt_caching: "prompt_caching",
	max_turns: "max_turns",
	context_limit: "context_window",
	concurrency: "concurrency",
	n_tasks: "n_tasks",
	tool_retry_cap: "tool_retry_cap",
	reasoning_effort: "reasoning_effort",
	// RL
	group_size: "group_size",
	num_generations: "group_size",
	max_completion_length: "truncation",
	completion_len: "truncation",
	// GNN, retrieval
	fan_out: "fan_out",
	num_neighbors: "fan_out",
};

export function termForKnob(knobId: string): string | null {
	const term = KNOB_TERMS[knobId];
	return term && GLOSSARY[term] ? term : null;
}

export interface GlossaryEntry {
	term: string;
	titleKey: string;
	bodyKey: string;
	actKey: string | null;
}

const PREFIX = "addon.expertise.term";

export function glossaryEntry(knobId: string, tier: Tier): GlossaryEntry | null {
	const term = termForKnob(knobId);
	if (!term) return null;
	return {
		term,
		titleKey: `${PREFIX}.${term}.label`,
		bodyKey: `${PREFIX}.${term}.${tier}`,
		actKey: GLOSSARY[term]?.act ? `${PREFIX}.${term}.act` : null,
	};
}

/** Flatten the glossary into t() keys. */
export function glossaryStrings(): Record<string, string> {
	const out: Record<string, string> = {};
	for (const [id, term] of Object.entries(GLOSSARY)) {
		for (const [field, text] of Object.entries(term)) out[`${PREFIX}.${id}.${field}`] = text;
	}
	return out;
}
