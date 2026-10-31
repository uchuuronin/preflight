import { describe, expect, it } from "vitest";
import { hasKey } from "../../lib/i18n";
import { applyKnobViews, type KnobViewContext } from "../../registry";
import { GridProvider } from "../../workspace/provider/GridProvider";
import { currentKnobs } from "../../workspace/record/model";
import { type FlightRecord, type LoadedRecord, type ObjectiveKind, normaliseRecord } from "../../workspace/record/types";
import { type Proportion, beatsClearly, wilson } from "../_stats/wilson";
import { createKnobView } from "./knobView";
import { resetTierForTests, setTier } from "./store";
import { registerExpertiseStrings } from "./strings";
import { type Claim, MAX_NEW_KNOBS, POLICY, type Policy } from "./tiers";
import { worthChecking } from "./worthChecking";

/**
 * Does "New to this" show what the zoo planted? Ground truth comes from A's zoo scripts, written
 * independently of the dial: each docstring lists its planted problems, and which make the run
 * fail. Records are matched to a script by `job.source.entrypoint`, so this reruns on whatever
 * predictions the records carry: A's fixtures today, B's engine output once make_golden.py runs,
 * and the unseen scripts on Nov 2 (held out: the policy is frozen before then and not retuned).
 *
 * A newcomer arrives at the record's own setting with the record's own goal; the dock shows what
 * C's applyKnobViews returns for D's view, ranked by that goal. Counts are per record: one per
 * planted problem, under the record's goal. Safety checks (failure knobs in the dock, nothing
 * lost) run for every goal. Two dock states:
 * - on screen today: C's dock doesn't list the warnings yet, so their knobs stay in the dock;
 * - with the list: C lists the warnings under the dock ("things worth checking"). That list is
 *   not on screen yet, so this is a design check of the rule, not a result.
 */
const ZOO: Record<string, { failure: string[]; waste: string[] }> = {
	// zoo/finetune_lora_hf/train.py: 1 eval batch 32, 2 eval without no_grad -> OOM at the first eval; 3 no packing
	"zoo/finetune_lora_hf/train.py": { failure: ["eval_batch_size", "eval_no_grad"], waste: ["packing"] },
	// zoo/pretrain_nanogpt/train.py: 1 num_workers 0, 2 fp32, 3 checkpoints every 20,000 steps; none fails the run
	"zoo/pretrain_nanogpt/train.py": { failure: [], waste: ["num_workers", "precision", "checkpoint_every"] },
};

const GOALS: ObjectiveKind[] = ["min_cost", "min_time", "fit_memory"];

const RECORDS = import.meta.glob<FlightRecord>("../../../../schema/examples/*.json", { eager: true, import: "default" });

interface Count {
	k: number;
	n: number;
}

interface Row {
	record: string;
	/** Where the predictions came from, e.g. golden-fixture@0. */
	models: string;
	/** The record's own goal, under which it is counted. */
	goal: string;
	/** Failure knobs in the dock, over every goal (a safety check, not a count). */
	failure: Count;
	/** Planted knob in the dock, under the record's own goal: one count per planted problem. */
	dock: Count;
	/** Planted knob in the dock, or its warning in the list (list state only), own goal. */
	visible: Count;
	/** Planted problems whose knob was missing from the dock, over every goal. */
	missedDock: string[];
	/** Planted problems neither in the dock nor listed, over every goal. */
	missed: string[];
	/** Most knobs the view picked by itself, over the goals. */
	most: number;
	/** Most knobs that must stay (failures, and warnings no list shows). */
	must: number;
	/** Listed warnings whose knob stayed in the dock after the visitor opened and closed it. */
	revealed: Count;
}

const zero = (): Count => ({ k: 0, n: 0 });

function zooRecords(): { name: string; record: LoadedRecord; truth: (typeof ZOO)[string] }[] {
	return Object.entries(RECORDS).flatMap(([path, raw]) => {
		const record = normaliseRecord(raw);
		const truth = ZOO[record.job.source?.entrypoint ?? ""];
		return truth ? [{ name: path.split("/").pop() ?? path, record, truth }] : [];
	});
}

function evaluate(listShown: boolean, policy: Policy = POLICY): Row[] {
	resetTierForTests();
	registerExpertiseStrings();
	setTier("new");
	const provider = new GridProvider();
	return zooRecords().map(({ name, record, truth }) => {
		// The app's default goal when a record states none (GridProvider, GuidePanel).
		const own = record.job.objective?.kind ?? "min_time";
		const row: Row = {
			record: name,
			models: [...new Set(record.models.map((m) => m.id))].join(", ") || "none",
			goal: own,
			failure: zero(),
			dock: zero(),
			visible: zero(),
			missedDock: [],
			missed: [],
			most: 0,
			must: 0,
			revealed: zero(),
		};
		const listed = listShown ? worthChecking(record) : [];
		const listedKnobs = new Set(listed.flatMap((f) => (f.knob_id ? [f.knob_id] : [])));
		const failing = record.findings.filter((f) => f.severity === "error" || f.severity === "critical").map((f) => f.knob_id);
		const warned = worthChecking(record).map((f) => f.knob_id);
		row.must = new Set([...failing, ...(listShown ? [] : warned)].filter(Boolean)).size;
		for (const goal of new Set<ObjectiveKind>([own as ObjectiveKind, ...GOALS])) {
			const knobs = currentKnobs(record);
			const ctx = {
				record,
				lens: record.job.lens,
				knobs,
				pinned: new Set<string>(),
				highlight: new Set<string>(),
				ranked: provider.sensitivity(record, { kind: goal }, { knobs }).map((x) => x.knob_id),
				active: null,
				findingsShown: listShown,
			} as KnobViewContext;
			const view = createKnobView(policy);
			const shown = new Set(applyKnobViews(record.knobs, ctx, [view], hasKey).shown.map((k) => k.id));
			row.most = Math.max(row.most, shown.size);
			for (const id of [...truth.failure, ...truth.waste]) {
				const inDock = shown.has(id);
				const seen = inDock || listedKnobs.has(id);
				if (goal === own) {
					row.dock.n++;
					row.visible.n++;
					if (inDock) row.dock.k++;
					if (seen) row.visible.k++;
				}
				if (!inDock) row.missedDock.push(`${id} (${goal})`);
				if (!seen) row.missed.push(`${id} (${goal})`);
				if (truth.failure.includes(id)) {
					row.failure.n++;
					if (inDock) row.failure.k++;
				}
			}
			for (const id of listedKnobs) {
				applyKnobViews(record.knobs, { ...ctx, active: id }, [view], hasKey);
				const after = applyKnobViews(record.knobs, ctx, [view], hasKey).shown.map((k) => k.id);
				row.revealed.n++;
				if (after.includes(id)) row.revealed.k++;
			}
		}
		return row;
	});
}

function pct(c: Count): string {
	const w: Proportion = wilson(c.k, c.n);
	return `${c.k}/${c.n} = ${(w.p * 100).toFixed(0)}% [${(w.lo * 100).toFixed(0)}, ${(w.hi * 100).toFixed(0)}]`;
}

function sum(rows: Row[], pick: (r: Row) => Count): Count {
	return rows.reduce((acc, r) => ({ k: acc.k + pick(r).k, n: acc.n + pick(r).n }), zero());
}

describe("New to this on the zoo, as on screen today (C's list not shown)", () => {
	const rows = evaluate(false);

	it("finds the zoo's records", () => {
		expect(rows.map((r) => r.record).sort()).toEqual(["finetune.json", "pretrain.json"]);
		expect(sum(rows, (r) => r.failure).n).toBeGreaterThan(0);
	});

	it("shows every planted problem's knob in the dock, warnings included, for every goal", () => {
		for (const r of rows) expect(r.missedDock, r.record).toEqual([]);
	});

	it("picks at most five knobs, unless more must stay", () => {
		for (const r of rows) expect(r.most, r.record).toBeLessThanOrEqual(Math.max(MAX_NEW_KNOBS, r.must));
	});

	it("reports per record: in-sample, 2 records, on screen (Wilson 95%)", () => {
		const report = rows.map((r) => `${r.record} [${r.models}, ${r.goal}]: in the dock ${pct(r.dock)}; at most ${r.most} knobs`);
		console.info(`[dial vs zoo: on screen today; in-sample, 2 records, one count per planted problem]\n${report.join("\n")}`);
		expect(report).toHaveLength(2);
	});
});

describe("New to this on the zoo, design check with C's list (list not yet on screen)", () => {
	const rows = evaluate(true);

	it("always shows the knobs behind a planted failure in the dock, for every goal", () => {
		for (const r of rows) expect(r.failure.k, `${r.record}: ${r.missedDock.join(", ")}`).toBe(r.failure.n);
	});

	it("picks at most five knobs when the visitor has done nothing", () => {
		for (const r of rows) expect(r.most, r.record).toBeLessThanOrEqual(Math.max(MAX_NEW_KNOBS, r.must));
	});

	it("keeps every planted problem visible, in the dock or listed as worth checking, for every goal", () => {
		for (const r of rows) expect(r.missed, r.record).toEqual([]);
	});

	it("shows a listed warning's knob once the visitor opens it, and keeps it", () => {
		for (const r of rows) expect(r.revealed.k, r.record).toBe(r.revealed.n);
	});

	it("reports per record: design check, 2 records, in-sample, counts a list not yet on screen (Wilson 95%)", () => {
		const report = rows.map(
			(r) => `${r.record} [${r.models}, ${r.goal}]: visible ${pct(r.visible)}; in the dock ${pct(r.dock)}; at most ${r.most} knobs`,
		);
		console.info(
			`[dial vs zoo: design check, 2 records, in-sample, counts a list not yet on screen, one count per planted problem]\n${report.join("\n")}`,
		);
		expect(report).toHaveLength(2);
	});
});

/**
 * Which order and how many of the goal's top knobs? With the list on screen, each version is
 * scored by how often the planted problems' knobs are in the dock under each record's own goal,
 * one count per planted problem, pooled over records (Chiya, Oct 28: per record, not per goal).
 * These are the records the rule is tuned on, so picking the best of nine would test on its own
 * training data: the configured POLICY stays unless another version beats it by a clear margin,
 * its Wilson 95% interval wholly above POLICY's (Chiya, Oct 27). The comparison is reported either
 * way. Planted problems in the same record still share a script, so the counts are not fully
 * independent; with 6 problems the intervals are wide.
 */
const ORDERS: Record<string, readonly Claim[]> = {
	"failures, top, baseline": POLICY.order,
	"failures, baseline, top": ["must", "baseline", "top", "rest"],
	"failures, top only": ["must", "top", "rest"],
};

describe("New to this: order of the five and the top count, compared on the zoo", () => {
	const results = Object.entries(ORDERS).flatMap(([name, order]) =>
		[1, 2, 3].map((top) => {
			const rows = evaluate(true, { top, order });
			return { name, order, top, dock: sum(rows, (r) => r.dock), models: rows[0]?.models ?? "none" };
		}),
	);

	it("keeps the configured policy unless another beats it by a clear margin", () => {
		const mine = results.find((r) => r.top === POLICY.top && r.order === POLICY.order);
		expect(mine, "POLICY is one of the compared versions").toBeDefined();
		const clear = results.filter((r) => mine && beatsClearly(r.dock, mine.dock));
		// With few planted problems even a perfect challenger can't clear today's interval; say so.
		const n = mine?.dock.n ?? 0;
		const decidable = !!mine && n > 0 && beatsClearly({ k: n, n }, mine.dock);
		console.info(
			`[dial order vs zoo, list on screen; predictions ${results[0]?.models}; in-sample, so the incumbent stays without a clear margin; ${decidable ? "a clear win is possible" : `no challenger can win on ${n} problems, so this can't decide`}]\n` +
				results.map((r) => `${r.name}, top ${r.top}: in the dock ${pct(r.dock)}${r === mine ? " (configured)" : ""}`).join("\n"),
		);
		expect(clear.map((r) => `${r.name}, top ${r.top}`)).toEqual([]);
	});
});
