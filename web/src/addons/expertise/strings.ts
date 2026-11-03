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
	"addon.expertise.more": "Show {n} more settings",
	"addon.expertise.worth_checking.one": "1 thing worth checking",
	"addon.expertise.worth_checking.other": "{n} things worth checking",
	...levelLabels({
		// B's list, Nov 3: knobs the engine reads that had no label on the site.
		dtype: ["Number precision", "dtype"],
		torch_dtype: ["Weights' number precision", "torch_dtype"],
		ptdtype: ["Number precision", "ptdtype"],
		compile: ["Compile for speed", "torch.compile"],
		prefetch_factor: ["Batches ready in advance", "prefetch_factor"],
		group_by_length: ["Group similar lengths", "group_by_length"],
		pin_memory: ["Faster copies to the GPU", "pin_memory"],
		nodes: ["Machines", "num_nodes"],
		inter_node_gbps: ["Network between machines", "inter-node bandwidth (Gbit/s)"],
		pipeline_parallel: ["Model split in stages", "pipeline_parallel_size"],
		rl_iterations: ["Training rounds", "rl_iterations"],
		request_rate: ["Requests per second", "request_rate"],
		max_batched_tokens: ["Tokens processed at once", "max_num_batched_tokens"],
		deepspeed_config: ["Memory-sharing setup", "deepspeed config"],
		tf32: ["Faster maths on newer GPUs", "allow_tf32"],
		dump_attentions: ["Save attention maps", "output_attentions"],
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
		lr_schedule: ["How learning speed changes", "lr_scheduler_type"],
		optimizer: ["Update method", "optim"],
		checkpoint_every: ["Save progress every", "save_steps"],
		max_steps: ["Training steps", "max_steps"],
		model: ["AI model", "model"],
		prompt_caching: ["Reuse the prompt start", "prompt caching"],
		max_turns: ["Most steps per task", "max_turns"],
		context_limit: ["How much it can read", "context_limit (tokens)"],
		concurrency: ["Running at once", "concurrency"],
		n_tasks: ["Number of tasks", "n_tasks"],
		tool_retry_cap: ["Tool retries", "tool_retry_cap"],
		reasoning_effort: ["Thinking time", "reasoning_effort"],
		// RL
		group_size: ["Answers per question", "num_generations"],
		max_completion_len: ["Longest answer", "max_completion_length"],
		max_prompt_len: ["Longest question", "max_prompt_length"],
		clip_high: ["Step limit (up)", "epsilon_high"],
		clip_low: ["Step limit (down)", "epsilon"],
		kl_beta: ["Stay close to the start", "beta"],
		loss_type: ["Learning rule", "loss_type"],
		temperature: ["Answer variety", "temperature"],
		use_vllm: ["Fast answer writing", "use_vllm"],
		// fine-tune and pretrain, wider set
		lora_alpha: ["Add-on strength", "lora_alpha"],
		lora_dropout: ["Add-on dropout", "lora_dropout"],
		attn: ["Attention code", "attn_implementation"],
		quantization: ["Model squeeze", "quantization_config"],
		epochs: ["Passes over the data", "num_train_epochs"],
		eval_every: ["Check progress every", "eval_steps"],
		log_every: ["Log every", "logging_steps"],
		grad_clip: ["Step size cap", "max_grad_norm"],
		warmup_steps: ["Gentle start", "warmup_steps"],
		weight_decay: ["Keep weights small", "weight_decay"],
		d_model: ["Model width", "d_model"],
		n_head: ["Attention heads", "n_head"],
		n_layer: ["Layers", "n_layer"],
		// multi-GPU
		gpus: ["Number of GPUs", "nproc_per_node"],
		fsdp_mode: ["Split the model across GPUs", "fsdp sharding_strategy"],
		tensor_parallel: ["GPUs per model copy", "tensor_parallel_size"],
		// serving and eval
		max_model_len: ["Longest conversation", "max_model_len"],
		max_num_seqs: ["Requests at once", "max_num_seqs"],
		gpu_memory_utilization: ["GPU memory share", "gpu_memory_utilization"],
		kv_cache_dtype: ["Memory precision", "kv_cache_dtype"],
		prefix_caching: ["Reuse shared starts", "enable_prefix_caching"],
		max_new_tokens: ["Longest answer", "max_new_tokens"],
		dump_hidden_states: ["Save model internals", "output_hidden_states"],
		// retrieval
		encode_batch_size: ["Texts encoded at once", "encode batch_size"],
		index_m: ["Index links", "HNSW M"],
		index_nlist: ["Index buckets", "IVF nlist"],
		// GNN
		gnn_layers: ["How far to look", "num_layers"],
		hidden_channels: ["Detail per node", "hidden_channels"],
		"fanout.hop1": ["Neighbours, first step out", "num_neighbors[0]"],
		"fanout.hop2": ["Neighbours, second step out", "num_neighbors[1]"],
		"fanout.hop3": ["Neighbours, third step out", "num_neighbors[2]"],
		"fanout.hop4": ["Neighbours, fourth step out", "num_neighbors[3]"],
		// vision
		image_size: ["Image size", "image_size"],
		channels_last: ["Image memory layout", "channels_last"],
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
