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

/**
 * Does "New to this" show what the zoo planted? Ground truth comes from A's zoo scripts, written
 * independently of the dial: each docstring lists its planted problems, and which make the run
 * fail. Records are matched to a script by `job.source.entrypoint`, so this reruns on whatever
 * predictions the records carry: A's fixtures today, B's engine output once make_golden.py runs.
 *
 * A newcomer arrives at the record's own setting with one goal; for each goal, the dock shows
 * what C's applyKnobViews returns for D's view, ranked by that same goal.
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
	all: { k: number; n: number };
	missed: string[];
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
		const row: Row = { record: path.split("/").pop() ?? path, failure: { k: 0, n: 0 }, all: { k: 0, n: 0 }, missed: [] };
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
			const view = { ...createKnobView() };
			const shown = new Set(applyKnobViews(record.knobs, ctx, [view], hasKey).shown.map((k) => k.id));
			for (const id of [...truth.failure, ...truth.waste]) {
				const hit = shown.has(id);
				row.all.n++;
				if (hit) row.all.k++;
				else row.missed.push(`${id} (${goal})`);
				if (truth.failure.includes(id)) {
					row.failure.n++;
					if (hit) row.failure.k++;
				}
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

	it("always shows the knobs behind a planted failure, for every goal", () => {
		for (const r of rows) expect(r.failure.k, `${r.record}: ${r.missed.join(", ")}`).toBe(r.failure.n);
	});

	it("reports how many planted problems a newcomer can see, per record (Wilson 95%)", () => {
		const report = rows.map((r) => `${r.record}: ${r.all.k}/${r.all.n} = ${pct(wilson(r.all.k, r.all.n))}; hidden: ${r.missed.join(", ") || "none"}`);
		console.info(`[dial vs zoo]\n${report.join("\n")}`);
		expect(report).toHaveLength(rows.length);
	});
});
