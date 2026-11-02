import type { LoadedRecord } from "../workspace/record/types";

/**
 * Knobs that change what the model learns, not just how fast or how cheaply. The tour offers
 * them only to clear a failure. Requested as knobs[].changes_results.
 */
const CHANGES_RESULTS = new Set([
	// what the model sees
	"seq_len",
	"packing",
	"group_by_length",
	"max_prompt_len",
	"max_completion_len",
	"max_new_tokens",
	"context_limit",
	"image_size",
	// what it can learn
	"lora_r",
	"lora_alpha",
	"lora_dropout",
	"quantization",
	"precision",
	"n_layer",
	"d_model",
	"n_head",
	"hidden_channels",
	"gnn_layers",
	"model",
	// how it learns
	"lr",
	"lr_schedule",
	"warmup_steps",
	"weight_decay",
	"optimizer",
	"grad_clip",
	"epochs",
	"max_steps",
	// RL and agents
	"group_size",
	"temperature",
	"reasoning_effort",
	"kl_beta",
	"clip_low",
	"clip_high",
	"loss_type",
	"max_turns",
	// what an eval measures: its tasks are its result, so running fewer is a different eval
	"n_tasks",
]);

/** A's `Knob.changes_results` when the record says so, else this table. */
export function changesResults(knobId: string, record?: LoadedRecord): boolean {
	if (record?.knobs.find((k) => k.id === knobId)?.changes_results) return true;
	return CHANGES_RESULTS.has(knobId);
}
