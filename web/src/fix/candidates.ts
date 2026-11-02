/**
 * Which fixes race for one finding: the finding's own template first, then, where verify can
 * judge the winner, up to two others for the same kind of run that aim at the same number (an
 * eval running out of memory: smaller eval batches, eval without gradients, gradient
 * checkpointing). Each knob a fix sets takes a value the
 * engine says fits; a fix whose value isn't known, or whose knob the person pinned, sits out.
 *
 * templates.json is the API's list (api/preflight_api/fix/templates.py), checked equal in
 * api/tests/test_patches.py.
 */
import { bestCombo } from "../guide/combos";
import { type Limits, NO_LIMITS, valueAllowed, withinLimits } from "../guide/limits";
import type { PredictionProvider } from "../workspace/provider/PredictionProvider";
import type { Finding, KnobValue, LoadedRecord, Objective } from "../workspace/record/types";
import data from "./templates.json";

export interface FixTemplate {
	id: string;
	title: string;
	lenses: string[];
	knob: string | null;
	metric: string;
	needs_value: boolean;
	verifiable: boolean;
}

export interface Candidate {
	template_id: string;
	value?: KnobValue;
}

export const TEMPLATES: FixTemplate[] = data.templates;
// Source: derived, the API's limit (fix/race.py MAX_CANDIDATES: the plan's race shows three fixes).
export const MAX_CANDIDATES = 3;
/** What the API reserves for each candidate (fix/run.py USD_PER_CANDIDATE), shown before the click. */
export const USD_PER_CANDIDATE: number = data.usd_per_candidate;

/** Values for knobs a record doesn't carry (the fix is in the code only). */
const CODE_ONLY_VALUES: Record<string, KnobValue> = {
	// Source: guess, a small cap that still lets a flaky tool succeed on a second or third try;
	// the visitor sees the value in the fix's diff.
	"agent.cap_tool_retries": 3,
};

/** Templates that stack others (the API's templates.STACKS): the finding's eval fixes together. */
export const STACKS: Record<string, readonly string[]> = { "eval.no_grad_and_batch": ["eval.wrap_no_grad", "eval.reduce_batch"] };

/**
 * When none of a finding's fixes passes verify alone (Chiya, Nov 2: the LoRA eval OOM needs eval
 * without gradients and a smaller eval batch together), the race stacks them. The eval batch is
 * the one in the engine's combined move that the guide offers (guide/combos.ts bestCombo, with the
 * visitor's pins and limits as the guide passes them), so the value isn't guessed. Only that value
 * is taken: the guide's move may also change another knob (seq_len on the LoRA record), which a
 * fix to the eval code doesn't touch, and the stack adds the no-grad fix the move doesn't model.
 * Null when the finding isn't one of a stack's parts, the guide has no combined move with an eval
 * batch, or the visitor's pins or limits rule it out.
 */
export function stackedCandidate(finding: Pick<Finding, "fix_template">, ctx: Ctx): Candidate | null {
	const id = Object.keys(STACKS).find((k) => STACKS[k]!.includes(finding.fix_template ?? ""));
	const t = template(id);
	if (!t || !t.knob || !t.lenses.includes(ctx.record.job.lens)) return null;
	const limits = ctx.limits ?? NO_LIMITS;
	if (ctx.pinned.has(t.knob) || limits.frozen.includes(t.knob) || ctx.pinned.has("eval_no_grad") || limits.frozen.includes("eval_no_grad")) return null;
	const o = withinLimits(ctx.record, ctx.provider, limits, ctx.knobs, { pinned: ctx.pinned });
	const combo = bestCombo(ctx.record, ctx.provider, ctx.knobs, ctx.objective, { pinned: o.pinned, ...(o.allow ? { allow: o.allow } : {}) });
	const to = combo?.changes.find((c) => c.knobId === t.knob)?.to;
	if (to === undefined || !valueAllowed(limits, t.knob, to)) return null;
	return { template_id: t.id, value: to };
}

export function template(id: string | null | undefined): FixTemplate | undefined {
	return id ? TEMPLATES.find((t) => t.id === id) : undefined;
}

interface Ctx {
	record: LoadedRecord;
	provider: PredictionProvider;
	knobs: Record<string, KnobValue>;
	pinned: ReadonlySet<string>;
	objective: Objective;
	/** The visitor's limits: a fix never sets a value they ruled out or a knob they froze. */
	limits?: Limits;
}

function valueFor(t: FixTemplate, ctx: Ctx): { ok: boolean; value?: KnobValue } {
	const limits = ctx.limits ?? NO_LIMITS;
	if (t.knob && (ctx.pinned.has(t.knob) || limits.frozen.includes(t.knob))) return { ok: false };
	const knob = t.knob ? ctx.record.knobs.find((k) => k.id === t.knob) : undefined;
	// A switch already on has nothing to fix; one the visitor ruled "on" out for isn't written.
	if (knob?.kind === "bool" && (ctx.knobs[knob.id] === true || !valueAllowed(limits, knob.id, true))) return { ok: false };
	if (!t.needs_value) return { ok: true };
	if (!knob) {
		const v = CODE_ONLY_VALUES[t.id];
		return v !== undefined && valueAllowed(limits, t.knob ?? t.id, v) ? { ok: true, value: v } : { ok: false };
	}
	if (knob.kind === "bool") return { ok: true, value: true };
	const move = ctx.provider
		.moves(ctx.record, ctx.objective, withinLimits(ctx.record, ctx.provider, limits, ctx.knobs, { pinned: ctx.pinned }, false))
		.find((m) => m.knobId === knob.id && m.fit === "fits");
	return move ? { ok: true, value: move.to } : { ok: false };
}

export function candidatesFor(finding: Finding, ctx: Ctx): Candidate[] {
	const first = template(finding.fix_template);
	const lens = ctx.record.job.lens;
	if (!first || !first.lenses.includes(lens)) return [];
	// Other fixes race only where verify can pick the winner by measuring: an agent's fixes can't
	// be verified yet, so its finding gets its own fix alone.
	// A stack (two fixes in one) races only after its parts have each failed verify (stackedCandidate).
	const others = first.verifiable ? TEMPLATES.filter((t) => t.id !== first.id && !(t.id in STACKS) && t.verifiable && t.metric === first.metric && t.lenses.includes(lens)) : [];
	const pool = [first, ...others];
	const out: Candidate[] = [];
	for (const t of pool) {
		const v = valueFor(t, ctx);
		if (!v.ok) {
			if (t === first) return []; // the finding's own fix can't be written: nothing races
			continue;
		}
		out.push(v.value === undefined ? { template_id: t.id } : { template_id: t.id, value: v.value });
		if (out.length === MAX_CANDIDATES) break;
	}
	return out;
}

const GPU_KNOBS = new Set(["gpu_type", "gpu", "gpus", "gpu_count"]);

/**
 * What the live fix loop races for a finding (FixRace.tsx). The live site writes fixes but doesn't
 * verify them, so it can't learn that every single fix fails before offering the stack. The
 * engine's own reading stands in: when no single move on the run's own settings clears the failure
 * (a bigger GPU doesn't count: it's a choice of card, not a fix to the script, as in the guide's
 * pair search), the stack races beside the single fixes. On
 * the LoRA sample this is what verify found on Nov 2: eval without gradients alone still ran out of
 * memory, the stack passed (docs/status/c.md D8).
 */
export function liveCandidates(finding: Finding, ctx: Ctx): Candidate[] {
	const round1 = candidatesFor(finding, ctx);
	if (!round1.length || round1.length >= MAX_CANDIDATES) return round1;
	const o = withinLimits(ctx.record, ctx.provider, ctx.limits ?? NO_LIMITS, ctx.knobs, { pinned: ctx.pinned });
	const clears = [ctx.objective.kind ?? "min_time", "fit_memory" as const].some((kind) =>
		ctx.provider.moves(ctx.record, { ...ctx.objective, kind }, o).some((m) => m.fit === "fits" && !GPU_KNOBS.has(m.knobId)),
	);
	if (clears) return round1;
	const stack = stackedCandidate(finding, ctx);
	return stack ? [...round1, stack] : round1;
}
