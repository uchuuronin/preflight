/**
 * Rule-based tour: the guide when Muse Spark is unavailable or the visitor's live turns are
 * spent, labelled "Rule-based tour" on screen. It plans from the provider's ranked moves and
 * re-plans after every Apply, Revert or Pin, so a step never goes stale.
 */
import { hasKey } from "../lib/i18n";
import { type CampaignPlan, finishBand } from "../workspace/campaign/plan";
import { probeCap, rlProbe, spreadCount } from "../workspace/rl";
import type { Move, PredictionProvider } from "../workspace/provider/PredictionProvider";
import { costStraddlesBudget, failureTarget, objectiveTarget } from "../workspace/record/model";
import { FAILS_AT, FITS_BELOW } from "../workspace/record/terms";
import type { Finding, KnobValue, LoadedRecord, Objective } from "../workspace/record/types";
import { template } from "../fix/candidates";
import { changesResults } from "./changesResults";
import { beat, bestCombo, bestSingleRisk, type ComboMove, comboKeys } from "./combos";
import { type Limits, NO_LIMITS, withinLimits } from "./limits";

export { changesResults };

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
	kind: "move" | "combo" | "probe_offer" | "sample_offer" | "finding" | "campaign" | "rl" | "blocked_by_pin" | "nothing_safe" | "done";
	/** combo: a setting that changes several knobs at once, when no single one clears a failure. */
	combo?: ComboMove;
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

/** The knob a finding's own fix moves: its fix template's, else the knob it names. */
function findingFixKnob(f: Finding): string | null {
	return template(f.fix_template)?.knob ?? f.knob_id ?? null;
}

/** Knobs that change the hardware rather than the job. */
const GPU_KNOBS = new Set(["gpu_type", "gpu", "gpus", "gpu_count"]);
export const isGpuMove = (m: Pick<Move, "knobId">): boolean => GPU_KNOBS.has(m.knobId);

/**
 * The fitting moves, ranked for the aim. While the run fails, a move on the current GPU comes before any
 * GPU change, and a finding's own fix first among those (PM, Nov 3: the LoRA sample's first step was
 * L4 → B300 when its own eval fix clears it): a GPU move is offered only when nothing on this GPU clears
 * the failure.
 */
function candidates(ctx: TourContext, pinned: ReadonlySet<string>): Move[] {
	const failing = isFailing(ctx);
	const ranked = ctx.provider
		.moves(ctx.record, ctx.objective, opts(ctx, pinned))
		.filter((m) => m.fit === "fits")
		.filter((m) => !ctx.visited.has(m.knobId))
		.filter((m) => failing || !changesResults(m.knobId, ctx.record));
	if (!failing) return ranked;
	const ownFix = new Set(ctx.record.findings.map((f) => findingFixKnob(f)).filter((k): k is string => !!k));
	const order = (m: Move) => (isGpuMove(m) ? 2 : ownFix.has(m.knobId) ? 0 : 1);
	return [...ranked].sort((a, b) => order(a) - order(b)); // stable: the aim's ranking within each group
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

/**
 * A problem found on a real knob that no move on the dock can show: the knob has no move for the
 * aim (its effect isn't predicted, as for a retrieval run's in-batch negatives or a vision run's
 * CPU augmentation), so the tour points at the finding instead of staying silent about it.
 */
function knobFindings(ctx: TourContext): Finding[] {
	const knobs = new Set(ctx.record.knobs.map((k) => k.id));
	// Knobs the predictions can rank for the aim at all (tight moves and ones that change results
	// included): a finding on one of those is the moves' to show, or deliberately held back.
	const shown = new Set(ctx.provider.moves(ctx.record, ctx.objective, opts(ctx, ctx.pinned, { includeTight: true })).map((m) => m.knobId));
	return ctx.record.findings.filter(
		// A knob the tour already moved, offered or had turned down has been dealt with on the dock.
		(f) =>
			FINDING_SEVERITIES.has(f.severity) &&
			!!f.knob_id &&
			knobs.has(f.knob_id) &&
			!shown.has(f.knob_id) &&
			!ctx.visited.has(f.knob_id) &&
			!ctx.visited.has(`probe:${f.knob_id}`) &&
			!(ctx.limits?.declined ?? []).includes(f.knob_id) &&
			!ctx.visited.has(findingVisit(f)),
	);
}

function nextFinding(ctx: TourContext): Finding | undefined {
	return codeFindings(ctx)[0];
}

/**
 * A finding on a knob comes once the moves are done, so it never pushes a saving aside, and waits
 * only while the run likely fails: a "tight" run with no move to clear it still hears about it.
 */
function lastFinding(ctx: TourContext): Finding | undefined {
	return likelyFails(ctx) ? undefined : knobFindings(ctx)[0];
}

function findingStep(ctx: TourContext, found: Finding): TourStep {
	const onKnob = !!found.knob_id && ctx.record.knobs.some((k) => k.id === found.knob_id);
	const nodes = found.node_id && ctx.record.graph.nodes.some((n) => n.id === found.node_id) ? [found.node_id] : [];
	return {
		kind: "finding",
		finding: found,
		nodes: onKnob ? [...new Set([...nodes, ...nodesForKnob(ctx.record, found.knob_id as string)])] : nodes,
		textKey: onKnob ? "guide.finding.knob" : "guide.finding",
		whyKey: onKnob ? "guide.finding.knob.why" : "guide.finding.why",
	};
}

/** Findings the tour has still to show (each is a step of its own). */
export function pendingFindings(ctx: TourContext): number {
	return (isFailing(ctx) ? 0 : codeFindings(ctx).length) + (likelyFails(ctx) ? 0 : knobFindings(ctx).length);
}

/**
 * When the aim's own number isn't predicted (a run whose length the record doesn't know has no
 * time or cost to rank), the measured memory still is: its moves, so the tour shows what the
 * probes can answer for rather than ending at "done".
 */
function aimUnpredicted(ctx: TourContext): boolean {
	const kind = ctx.objective.kind ?? "min_time";
	if (kind === "fit_memory") return false;
	const goal = objectiveTarget(ctx.record, kind, ctx.objective.target_id);
	if (!goal) return true;
	const e = ctx.provider.predict(ctx.record, ctx.knobs)[goal.id];
	return !e || e.status === "abstain";
}

function memoryFallback(ctx: TourContext): Move | undefined {
	if (!aimUnpredicted(ctx)) return undefined;
	const mem: TourContext = { ...ctx, objective: { ...ctx.objective, kind: "fit_memory" } };
	return candidates(mem, ctx.pinned)[0];
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
	if (found) return findingStep(ctx, found);
	// A failure comes first whatever the aim: when the aim ranks no move that fits, the moves
	// ranked for memory are the ones that can clear it.
	const forAim = candidates(ctx, ctx.pinned)[0];
	const top = forAim ?? (failing ? candidates({ ...ctx, objective: { ...ctx.objective, kind: "fit_memory" } }, ctx.pinned)[0] : undefined);
	// Ranked by memory, not the aim (the aim has no number for the moves that fit): its reason says so,
	// since "costs your aim the least" wouldn't be true (review, Nov 2).
	const byMemory = !forAim && !!top;
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
	const fallback = top || failing ? undefined : memoryFallback(ctx);
	if (fallback) {
		return {
			kind: "move",
			move: fallback,
			nodes: nodesForKnob(ctx.record, fallback.knobId),
			textKey: "guide.rule.aim_unpredicted.line",
			whyKey: "guide.rule.aim_unpredicted.why",
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
			whyKey: byMemory
				? "guide.rule.fix_by_memory.why"
				: fixing && isGpuMove(top)
					? "guide.rule.gpu_type.fix.why"
					: specific
						? keyFor(top.knobId, "why", fixing)
						: "guide.rule.generic.fix.why",
		};
	}
	if (failing) {
		// No single knob clears it: two together may, where B's pair grids cover the combination
		// (a tight one only if it beats the best single move).
		const o = opts(ctx, ctx.pinned);
		const combo = bestCombo(ctx.record, ctx.provider, ctx.knobs, ctx.objective, { ...beat(bestSingleRisk(ctx.record, ctx.provider, ctx.objective, o)), pinned: o.pinned, ...(o.allow ? { allow: o.allow } : {}), visited: ctx.visited });
		if (combo) {
			return {
				kind: "combo",
				combo,
				nodes: [...new Set(combo.changes.flatMap((c) => nodesForKnob(ctx.record, c.knobId)))],
				...comboKeys(combo),
			};
		}
	}
	// Nothing movable helps. A problem found on a knob no move could show comes first.
	const last = lastFinding(ctx);
	if (last) return findingStep(ctx, last);
	// If a pin is in the way, say which one and what releasing it buys.
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
			// "likely to fail" only when it likely would; a small chance is said as one
			textKey: likelyFails(ctx) ? "guide.nothing_safe" : "guide.nothing_safe.tight",
			// the probe line only when there is a setting worth measuring
			whyKey: toMeasure ? "guide.nothing_safe.why" : "guide.nothing_safe.none.why",
		};
	}
	return { kind: "done", nodes: [], textKey: "guide.done", whyKey: "guide.done.why" };
}

export function plannedSteps(ctx: TourContext): number {
	const n = candidates(ctx, ctx.pinned).length;
	if (n || isFailing(ctx) || !aimUnpredicted(ctx)) return n;
	return candidates({ ...ctx, objective: { ...ctx.objective, kind: "fit_memory" } }, ctx.pinned).length;
}

export const MAX_TOUR_STEPS = 5;
