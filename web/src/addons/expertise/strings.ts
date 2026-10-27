/**
 * English copy for the expertise add-on, registered into C's t() under addon.expertise.*.
 * British spelling, no em dashes. Other languages (D7) add a sibling strings file.
 */
import { registerStrings } from "../../lib/i18n";
import { glossaryStrings } from "./glossary";

export const EXPERTISE_EN: Record<string, string> = {
	"addon.expertise.dial.title": "Detail level",
	"addon.expertise.tier.new": "New to this",
	"addon.expertise.tier.new.hint": "Plain words, only the settings that matter most",
	"addon.expertise.tier.practitioner": "Practitioner",
	"addon.expertise.tier.practitioner.hint": "Standard terms and the usual settings",
	"addon.expertise.tier.expert": "Expert",
	"addon.expertise.tier.expert.hint": "Every setting, with formulas and model details",
	"addon.expertise.card.summary": "What this setting does",
	"addon.expertise.card.act": "What to change",
	"addon.expertise.card.more": "More detail",
	...levelLabels({
		batch_size: ["Examples at once", "per_device_train_batch_size"],
		seq_len: ["Longest example", "max_seq_length"],
		grad_accum: ["Batches added up per step", "gradient_accumulation_steps"],
		eval_batch_size: ["Checks at once", "per_device_eval_batch_size"],
		eval_no_grad: ["Lighter checks", "eval under torch.no_grad()"],
		lora_r: ["Add-on size", "lora_r"],
		gradient_checkpointing: ["Save memory (slower)", "gradient_checkpointing"],
		precision: ["Number precision", "torch_dtype / amp"],
		packing: ["Pack short examples together", "packing"],
		gpu_type: ["GPU", "gpu_type"],
		gpu: ["GPU", "gpu_type"],
		num_workers: ["Data helpers", "DataLoader(num_workers)"],
		lr: ["Learning speed", "learning_rate"],
		checkpoint_every: ["Save progress every", "save_steps"],
		max_steps: ["Training steps", "max_steps"],
		model: ["AI model", "model"],
		prompt_caching: ["Reuse the prompt start", "prompt caching"],
		max_turns: ["Most steps per task", "max_turns"],
		context_limit: ["How much it can read", "context_limit (tokens)"],
		concurrency: ["Running at once", "concurrency"],
		n_tasks: ["Number of tasks", "n_tasks"],
		tool_retry_cap: ["Tool retries", "tool_retry_cap"],
	}),
};

/** knob id -> [new-level label, expert-level code name]. */
function levelLabels(table: Record<string, [string, string]>): Record<string, string> {
	const out: Record<string, string> = {};
	for (const [id, [plain, code]] of Object.entries(table)) {
		out[`addon.expertise.label.${id}.new`] = plain;
		out[`addon.expertise.label.${id}.expert`] = code;
	}
	return out;
}

export function registerExpertiseStrings(): void {
	registerStrings("en", { ...EXPERTISE_EN, ...glossaryStrings() });
}
