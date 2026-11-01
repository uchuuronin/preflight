/**
 * Rule-based tour: the guide when Muse Spark is unavailable or the visitor's live turns are
 * spent, labelled "Rule-based tour" on screen. It plans from the provider's ranked moves and
 * re-plans after every Apply, Revert or Pin, so a step never goes stale.
 */
import { hasKey } from "../lib/i18n";
import { type CampaignPlan, finishBand } from "../workspace/campaign/plan";
import { probeCap, rlProbe, spreadCount } from "../workspace/rl";
import type { Move, PredictionProvider } from "../workspace/provider/PredictionProvider";
import { costStraddlesBudget, failureTarget } from "../workspace/record/model";
import { FAILS_AT, FITS_BELOW } from "../workspace/record/terms";
import type { Finding, KnobValue, LoadedRecord, Objective } from "../workspace/record/types";
import { type Limits, NO_LIMITS, withinLimits } from "./limits";

/**
 * Map nodes a knob acts on, by knob id, for records from before A's `knobs[].affects`
 * (harness/knob_meta.py). A knob with its own `affects` uses that and the record's findings only.
 */
const KNOB_NODES: Record<string, string[]> = {
	eval_batch_size: ["eval"],
	eval_no_grad: ["eval"],
	batch_size: ["data", "model"],
	seq_len: ["model", "lm_head"],
	gradient_checkpointing: ["model"],
	gpu_type: ["model"],
	gpu: ["model"],
	gpus: ["model"],
	fsdp_mode: ["model", "optimizer"],
	lora_r: ["optimizer", "model"],
	lora_alpha: ["model"],
	lora_dropout: ["model"],
	quantization: ["model"],
	attn: ["model"],
	precision: ["model"],
	optimizer: ["optimizer"],
	weight_decay: ["optimizer"],
	lr: ["optimizer"],
	grad_clip: ["optimizer"],
	packing: ["data"],
	group_by_length: ["data"],
	num_workers: ["data"],
	grad_accum: ["optimizer"],
	checkpoint_every: ["checkpoint"],
	eval_every: ["eval"],
	max_new_tokens: ["rollout"],
	group_size: ["rollout"],
	use_vllm: ["rollout"],
	gpu_memory_utilization: ["rollout"],
	kv_cache_dtype: ["rollout"],
};

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

export function nodesForKnob(record: LoadedRecord, knobId: string): string[] {
	// A record that says what a knob acts on (even nothing) is the answer; the table is only for
	// records from before A wrote it, since its coarse ids ("model") would light a whole group.
	const declared = record.knobs.find((k) => k.id === knobId)?.affects;
	const fromFindings = record.findings.filter((f) => f.knob_id === knobId && f.node_id).map((f) => f.node_id as string);
	const known = new Set(record.graph.nodes.map((n) => n.id));
	const base = Array.isArray(declared) ? declared : (KNOB_NODES[knobId] ?? []);
	return [...new Set([...base, ...fromFindings])].filter((n) => known.has(n));
}

export interface TourStep {
	/**
	 * move: apply a "fits" move. probe_offer: a "tight" setting would save enough to be worth
	 * measuring first (the person's click). blocked_by_pin, nothing_safe, done: as named.
	 */
	kind: "move" | "probe_offer" | "sample_offer" | "finding" | "campaign" | "rl" | "blocked_by_pin" | "nothing_safe" | "done";
	/** campaign: which part of the record's campaign plan this step walks (the calendar lights up). */
	campaignPart?: CampaignPart;
	/** rl: which measured RL signal this step walks (the RL view lights up). */
	rlPart?: RlPart;
	/** finding: a problem A's harness or a diagnosis found in the code, with its fix template. */
	finding?: Finding;
	move?: Move;
	pinnedKnob?: string;
	nodes: string[];
	textKey: string;
	whyKey: string;
}

export interface TourContext {
	record: LoadedRecord;
	provider: PredictionProvider;
	knobs: Readonly<Record<string, KnobValue>>;
	pinned: ReadonlySet<string>;
	objective: Objective;
	visited: ReadonlySet<string>;
	/** The visitor's limits: no step ever suggests a value outside them. */
	limits?: Limits;
	/** The record's campaign plan when its calendar is on screen: the tour walks it first. */
	campaign?: CampaignPlan | null;
}

export type RlPart = "spread" | "length";
export const rlVisit = (part: RlPart) => `rl:${part}`;

/** The measured RL signals the tour still has to walk: prompts with no reward spread, then completion length against the cap. */
export function rlParts(ctx: Pick<TourContext, "record" | "visited">): RlPart[] {
	if (ctx.record.job.lens !== "rl") return [];
	const p = rlProbe(ctx.record);
	if (!p?.rl) return [];
	const parts: RlPart[] = [];
	if (spreadCount(p)) parts.push("spread");
	if (p.rl.response_length?.max !== undefined && p.rl.response_length?.max !== null) parts.push("length");
	return parts.filter((x) => !ctx.visited.has(rlVisit(x)));
}

export type CampaignPart = "grid" | "cut" | "finish";
export const campaignVisit = (part: CampaignPart) => `campaign:${part}`;

/** The parts of the campaign plan the tour still has to walk, in order: what it costs, what the budget cuts, when it ends. */
export function campaignParts(ctx: Pick<TourContext, "campaign" | "visited">): CampaignPart[] {
	const c = ctx.campaign;
	if (!c || !c.cells.length) return [];
	const parts: CampaignPart[] = ["grid"];
	if (c.cut_list.length) parts.push("cut");
	parts.push("finish");
	return parts.filter((p) => !ctx.visited.has(campaignVisit(p)));
}

/** Provider options for this tour: the pins given, the visitor's limits, declined knobs skipped. */
function opts(ctx: TourContext, pinned: ReadonlySet<string>, extra: { includeTight?: boolean } = {}) {
	return withinLimits(ctx.record, ctx.provider, ctx.limits ?? NO_LIMITS, ctx.knobs, { pinned, ...extra });
}

function keyFor(knobId: string, kind: "line" | "why", fixing: boolean): string {
	const specific = `guide.rule.${knobId}.${fixing ? "fix" : "save"}.${kind}`;
	if (hasKey(specific)) return specific;
	return `guide.rule.generic.${fixing ? "fix" : "save"}.${kind}`;
}

/** A "tight" setting is worth a probe when it improves the aim by at least this share. */
export const PROBE_WORTH = 0.1;

function riskNow(ctx: Pick<TourContext, "record" | "provider" | "knobs">): number {
	const fail = failureTarget(ctx.record);
	if (!fail) return 0;
	return ctx.provider.predict(ctx.record, ctx.knobs)[fail]?.point ?? 0;
}

/** Any real chance of failing (not "fits"): the tour deals with it before saving money or time. */
export function isFailing(ctx: Pick<TourContext, "record" | "provider" | "knobs">): boolean {
	return riskNow(ctx) >= FITS_BELOW;
}

/** Likely to fail ("fails"): the knob-specific lines that say the run would stop apply. */
export function likelyFails(ctx: Pick<TourContext, "record" | "provider" | "knobs">): boolean {
	return riskNow(ctx) >= FAILS_AT;
}

function candidates(ctx: TourContext, pinned: ReadonlySet<string>): Move[] {
	const failing = isFailing(ctx);
	return ctx.provider
		.moves(ctx.record, ctx.objective, opts(ctx, pinned))
		.filter((m) => m.fit === "fits")
		.filter((m) => !ctx.visited.has(m.knobId))
		.filter((m) => failing || !changesResults(m.knobId, ctx.record));
}

/** The best "tight" setting worth measuring: beats the best "fits" move by PROBE_WORTH. */
function probeOffer(ctx: TourContext, bestFits: Move | undefined): Move | undefined {
	const failing = isFailing(ctx);
	if (failing) return undefined;
	const offers = ctx.provider
		.moves(ctx.record, ctx.objective, opts(ctx, ctx.pinned, { includeTight: true }))
		.filter((m) => m.fit === "tight" && !ctx.visited.has(`probe:${m.knobId}`) && !ctx.visited.has(m.knobId))
		.filter((m) => !changesResults(m.knobId, ctx.record));
	// moves() returns only improvements here, so |relative| is the share the aim improves by.
	const base = Math.abs(bestFits?.relative ?? 0);
	return offers.find((m) => Math.abs(m.relative ?? 0) - base >= PROBE_WORTH);
}

/** Visited marker for the sample offer, so Skip moves on. */
export const SAMPLE_VISIT = "sample:more";

/** Visited marker for a finding the tour has shown. */
export const findingVisit = (f: Finding) => `finding:${f.id}`;

const FINDING_SEVERITIES = new Set(["critical", "error", "warn"]);

/**
 * The next problem found in the code that no knob on the dock shows: one with a fix template whose
 * knob, if it names one, isn't a knob of this record (a retry loop's cap is in the code, not a
 * setting). Findings tied to a real knob are the moves' to show, so they never come here.
 */
function codeFindings(ctx: Pick<TourContext, "record" | "visited">): Finding[] {
	const knobs = new Set(ctx.record.knobs.map((k) => k.id));
	return ctx.record.findings.filter(
		(f) =>
			FINDING_SEVERITIES.has(f.severity) &&
			!!f.fix_template &&
			!(f.knob_id && knobs.has(f.knob_id)) &&
			!ctx.visited.has(findingVisit(f)),
	);
}

function nextFinding(ctx: TourContext): Finding | undefined {
	return codeFindings(ctx)[0];
}

/** Code findings the tour has still to show (each is a step of its own). */
export function pendingFindings(ctx: TourContext): number {
	return isFailing(ctx) ? 0 : codeFindings(ctx).length;
}

export function nextStep(ctx: TourContext): TourStep {
	// A campaign record: the plan comes first (what the grid costs, what the budget cuts, when it
	// ends), since no knob move shows it; then the run's own tour.
	// An RL run: what the probe measured about the reward signal comes first, since no knob move
	// shows a prompt that teaches nothing.
	const rlPart = rlParts(ctx)[0];
	if (rlPart) {
		let textKey = `guide.rl.${rlPart}`;
		if (rlPart === "length") {
			const p = rlProbe(ctx.record);
			// Against the cap the run was measured at, not the knob as it is now.
			const cap = p ? probeCap(p) : null;
			const longest = p?.rl?.response_length?.max;
			// "Reached the cap" only when the longest completion did.
			if (!(typeof cap === "number" && typeof longest === "number" && longest >= cap)) textKey = "guide.rl.length_under";
		}
		return { kind: "rl", rlPart, nodes: [], textKey, whyKey: `guide.rl.${rlPart}.why` };
	}
	const part = campaignParts(ctx)[0];
	if (part) {
		const abstains = part === "finish" && (!ctx.campaign?.finish_s || ctx.campaign.finish_s.status === "abstain" || !finishBand(ctx.campaign));
		return {
			kind: "campaign",
			campaignPart: part,
			nodes: [],
			textKey: abstains ? "guide.campaign.finish_abstain" : `guide.campaign.${part}`,
			whyKey: `guide.campaign.${part}.why`,
		};
	}
	// An agent run whose cost range straddles the budget: whether it fits depends on where the
	// real cost lands, so measuring more tasks comes before any saving (sample_more, a click).
	if (!ctx.visited.has(SAMPLE_VISIT) && costStraddlesBudget(ctx.record, ctx.provider.predict(ctx.record, ctx.knobs))) {
		return { kind: "sample_offer", nodes: [], textKey: "guide.sample_offer", whyKey: "guide.sample_offer.why" };
	}
	const failing = isFailing(ctx);
	// A run that isn't failing: problems found in the code (a retry loop with no cap) come before
	// savings, since no knob move shows them.
	const found = failing ? undefined : nextFinding(ctx);
	if (found) {
		return {
			kind: "finding",
			finding: found,
			nodes: found.node_id && ctx.record.graph.nodes.some((n) => n.id === found.node_id) ? [found.node_id] : [],
			textKey: "guide.finding",
			whyKey: "guide.finding.why",
		};
	}
	const top = candidates(ctx, ctx.pinned)[0];
	const offer = probeOffer(ctx, top);
	if (offer) {
		return {
			kind: "probe_offer",
			move: offer,
			nodes: nodesForKnob(ctx.record, offer.knobId),
			textKey: "guide.probe_offer",
			whyKey: "guide.probe_offer.why",
		};
	}
	if (top) {
		const fixing = top.clearsFailure || failing;
		// Knob-specific fix lines say the run would stop; they fit only when it likely would.
		const specific = !fixing || likelyFails(ctx);
		return {
			kind: "move",
			move: top,
			nodes: nodesForKnob(ctx.record, top.knobId),
			textKey: specific ? keyFor(top.knobId, "line", fixing) : "guide.rule.generic.fix.line",
			whyKey: specific ? keyFor(top.knobId, "why", fixing) : "guide.rule.generic.fix.why",
		};
	}
	// Nothing movable helps. If a pin is in the way, say which one and what releasing it buys.
	if (ctx.pinned.size) {
		const blocker = candidates(ctx, new Set()).find((m) => ctx.pinned.has(m.knobId));
		if (blocker) {
			return {
				kind: "blocked_by_pin",
				move: blocker,
				pinnedKnob: blocker.knobId,
				nodes: nodesForKnob(ctx.record, blocker.knobId),
				textKey: failing ? "guide.pin_blocks_fix" : "guide.pin_blocks_saving",
				whyKey: "guide.pin_blocks.why",
			};
		}
	}
	if (failing) {
		const nodes = ctx.record.findings.filter((f) => f.severity !== "info" && f.node_id).map((f) => f.node_id as string);
		// The setting most worth measuring: the move that lowers the risk most, even if still tight.
		const toMeasure = ctx.provider
			.moves(ctx.record, ctx.objective, opts(ctx, ctx.pinned, { includeTight: true }))
			.find((m) => m.fit === "tight");
		return {
			kind: "nothing_safe",
			...(toMeasure ? { move: toMeasure } : {}),
			nodes: [...new Set([...nodes, ...(toMeasure ? nodesForKnob(ctx.record, toMeasure.knobId) : [])])],
			textKey: "guide.nothing_safe",
			whyKey: "guide.nothing_safe.why",
		};
	}
	return { kind: "done", nodes: [], textKey: "guide.done", whyKey: "guide.done.why" };
}

export function plannedSteps(ctx: TourContext): number {
	return candidates(ctx, ctx.pinned).length;
}

export const MAX_TOUR_STEPS = 5;
