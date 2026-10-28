import { describe, expect, it } from "vitest";
import { hasKey } from "../../lib/i18n";
import { applyKnobViews, type KnobViewContext } from "../../registry";
import { GridProvider } from "../../workspace/provider/GridProvider";
import { currentKnobs } from "../../workspace/record/model";
import { type FlightRecord, type ObjectiveKind, normaliseRecord } from "../../workspace/record/types";
import { type Proportion, wilson } from "../_stats/wilson";
import { createKnobView } from "./knobView";
import { resetTierForTests, setTier } from "./store";
import { registerExpertiseStrings } from "./strings";
import { worthChecking } from "./worthChecking";

/**
 * Does "New to this" show what the zoo planted? Ground truth comes from A's zoo scripts, written
 * independently of the dial: each docstring lists its planted problems, and which make the run
 * fail. Records are matched to a script by `job.source.entrypoint`, so this reruns on whatever
 * predictions the records carry: A's fixtures today, B's engine output once make_golden.py runs.
 *
 * A newcomer arrives at the record's own setting with one goal; for each goal, the dock shows
 * what C's applyKnobViews returns for D's view, ranked by that same goal. A planted problem is
 * visible when its knob is in the dock or its finding is in the "things worth checking" list
 * (worthChecking.ts), which C's findings display renders under the dock once that spot exists.
 */
const ZOO: Record<string, { failure: string[]; waste: string[] }> = {
	// zoo/finetune_lora_hf/train.py: 1 eval batch 32, 2 eval without no_grad -> OOM at the first eval; 3 no packing
	"zoo/finetune_lora_hf/train.py": { failure: ["eval_batch_size", "eval_no_grad"], waste: ["packing"] },
	// zoo/pretrain_nanogpt/train.py: 1 num_workers 0, 2 fp32, 3 checkpoints every 20,000 steps; none fails the run
	"zoo/pretrain_nanogpt/train.py": { failure: [], waste: ["num_workers", "precision", "checkpoint_every"] },
};

const GOALS: ObjectiveKind[] = ["min_cost", "min_time", "fit_memory"];

const RECORDS = import.meta.glob<FlightRecord>("../../../../schema/examples/*.json", { eager: true, import: "default" });

interface Row {
	record: string;
	failure: { k: number; n: number };
	/** Knob in the dock. */
	dock: { k: number; n: number };
	/** Knob in the dock, or its warning in the list. */
	visible: { k: number; n: number };
	missed: string[];
	/** Most knobs the view picked by itself, over the goals. */
	most: number;
	/** Listed warnings whose knob stayed in the dock after the visitor opened and closed it. */
	revealed: { k: number; n: number };
}

function evaluate(): Row[] {
	resetTierForTests();
	registerExpertiseStrings();
	setTier("new");
	const provider = new GridProvider();
	const rows: Row[] = [];
	for (const [path, raw] of Object.entries(RECORDS)) {
		const record = normaliseRecord(raw);
		const truth = ZOO[record.job.source?.entrypoint ?? ""];
		if (!truth) continue;
		const row: Row = {
			record: path.split("/").pop() ?? path,
			failure: { k: 0, n: 0 },
			dock: { k: 0, n: 0 },
			visible: { k: 0, n: 0 },
			missed: [],
			most: 0,
			revealed: { k: 0, n: 0 },
		};
		const listed = worthChecking(record);
		const listedKnobs = new Set(listed.flatMap((f) => (f.knob_id ? [f.knob_id] : [])));
		for (const goal of GOALS) {
			const knobs = currentKnobs(record);
			const ctx: KnobViewContext = {
				record,
				lens: record.job.lens,
				knobs,
				pinned: new Set(),
				highlight: new Set(),
				ranked: provider.sensitivity(record, { kind: goal }, { knobs }).map((x) => x.knob_id),
				active: null,
			};
			const view = createKnobView();
			const shown = new Set(applyKnobViews(record.knobs, ctx, [view], hasKey).shown.map((k) => k.id));
			row.most = Math.max(row.most, shown.size);
			for (const id of [...truth.failure, ...truth.waste]) {
				const inDock = shown.has(id);
				const seen = inDock || listedKnobs.has(id);
				row.dock.n++;
				row.visible.n++;
				if (inDock) row.dock.k++;
				if (seen) row.visible.k++;
				else row.missed.push(`${id} (${goal})`);
				if (truth.failure.includes(id)) {
					row.failure.n++;
					if (inDock) row.failure.k++;
				}
			}
			// The visitor opens each listed warning's knob, then closes it.
			for (const id of listedKnobs) {
				applyKnobViews(record.knobs, { ...ctx, active: id }, [view], hasKey);
				const after = applyKnobViews(record.knobs, ctx, [view], hasKey).shown.map((k) => k.id);
				row.revealed.n++;
				if (after.includes(id)) row.revealed.k++;
			}
		}
		rows.push(row);
	}
	return rows;
}

function pct(w: Proportion): string {
	return `${(w.p * 100).toFixed(0)}% [${(w.lo * 100).toFixed(0)}, ${(w.hi * 100).toFixed(0)}]`;
}

describe("New to this on the zoo's planted problems", () => {
	const rows = evaluate();

	it("finds the zoo's records", () => {
		expect(rows.map((r) => r.record).sort()).toEqual(["finetune.json", "pretrain.json"]);
		expect(rows.reduce((s, r) => s + r.failure.n, 0)).toBeGreaterThan(0);
	});

	it("always shows the knobs behind a planted failure in the dock, for every goal", () => {
		for (const r of rows) expect(r.failure.k, `${r.record}: ${r.missed.join(", ")}`).toBe(r.failure.n);
	});

	it("picks at most five knobs when the visitor has done nothing", () => {
		for (const r of rows) expect(r.most, r.record).toBeLessThanOrEqual(5);
	});

	it("keeps every planted problem visible, in the dock or listed as worth checking", () => {
		for (const r of rows) expect(r.visible.k, `${r.record}: hidden ${r.missed.join(", ")}`).toBe(r.visible.n);
	});

	it("shows a listed warning's knob once the visitor opens it, and keeps it", () => {
		for (const r of rows) expect(r.revealed.k, r.record).toBe(r.revealed.n);
	});

	it("reports, per record, planted problems visible and in the dock (Wilson 95%)", () => {
		const report = rows.map(
			(r) =>
				`${r.record}: visible ${r.visible.k}/${r.visible.n} = ${pct(wilson(r.visible.k, r.visible.n))}; ` +
				`in the dock ${r.dock.k}/${r.dock.n} = ${pct(wilson(r.dock.k, r.dock.n))}; at most ${r.most} knobs`,
		);
		console.info(`[dial vs zoo, A's fixture predictions]\n${report.join("\n")}`);
		expect(report).toHaveLength(rows.length);
	});
});
