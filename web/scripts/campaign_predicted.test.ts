/**
 * What the cockpit predicts for each cell of the campaign probe grid (api/campaign_probe_app.py),
 * from the Nov 2 live record (PF102, measured on an L4), on the L4 and the A10, before the cells
 * were probed. It writes JSON and checks nothing, so it runs only when asked for:
 *
 *     PREDICTED_OUT=out.json npx vitest run --dir scripts campaign_predicted
 */
import { writeFileSync } from "node:fs";
import { it } from "vitest";
import raw from "../src/workspace/fixtures/lora-live-nov1.record.json";
import { GridProvider } from "../src/workspace/provider/GridProvider";
import { currentKnobs } from "../src/workspace/record/model";
import {
	type FlightRecord,
	normaliseRecord,
} from "../src/workspace/record/types";

it.skipIf(!process.env.PREDICTED_OUT)("writes the grid's predictions", () => {
	const record = normaliseRecord(raw as unknown as FlightRecord);
	const provider = new GridProvider();
	const base = currentKnobs(record);
	const out: Record<string, unknown> = { record: record.record_id };
	for (const batch_size of [2, 4])
		for (const lr of [0.0001, 0.0002])
			for (const gpu_type of ["A10", "L4"]) {
				const knobs = { ...base, batch_size, lr, gpu_type };
				const est = provider.predict(record, knobs);
				const pick = (id: string) => {
					const e = est[id];
					return e
						? {
								status: e.status,
								point: e.point,
								lo: e.lo,
								hi: e.hi,
								reason: e.reason ?? null,
							}
						: null;
				};
				out[`bs${batch_size}-lr${lr}-${gpu_type}`] = {
					"time.step": pick("time.step"),
					"memory.peak": pick("memory.peak"),
					"risk.oom": pick("risk.oom"),
					"time.total": pick("time.total"),
					"cost.total": pick("cost.total"),
				};
			}
	writeFileSync(process.env.PREDICTED_OUT!, JSON.stringify(out, null, 1));
});
