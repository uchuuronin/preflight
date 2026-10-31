import { describe, expect, it } from "vitest";
import golden from "../../../../schema/examples/finetune.json";
import type { Estimate, FlightRecord } from "../record/types";
import { normaliseRecord } from "../record/types";
import { exportPlan } from "./plan";

const record = normaliseRecord(golden as unknown as FlightRecord);
const at = (over: Record<string, number | string | boolean>) => ({ ...Object.fromEntries(record.knobs.map((k) => [k.id, k.current])), ...over });
const est = (status: Estimate["status"], point = 100, lo = 80, hi = 130) => ({ status, point, lo, hi }) as Estimate;

describe("the plan an export carries", () => {
	it("lists only the moved knobs, with where the script sets them", () => {
		const plan = exportPlan(record, at({ eval_batch_size: 8 }), {}, "tok", "https://x.test/");
		expect(plan.changes.map((c) => c.knobId)).toEqual(["eval_batch_size"]);
		const c = plan.changes[0];
		expect(c?.from).toBe(32);
		expect(c?.to).toBe(8);
		expect(c?.location?.expr).toBe("per_device_eval_batch_size=32");
		expect(plan.config.eval_batch_size).toBe(8);
		expect(Object.keys(plan.config)).toHaveLength(record.knobs.length);
	});

	it("never carries a number for a readout that abstains", () => {
		const plan = exportPlan(record, at({}), { "time.total": est("abstain"), "memory.peak": est("predicted") }, null, null);
		const time = plan.readouts.find((r) => r.targetId === "time.total");
		expect(time?.status).toBe("abstain");
		expect([time?.point, time?.lo, time?.hi]).toEqual([null, null, null]);
		expect(plan.wallTimeHiS).toBeNull();
		expect(plan.readouts.find((r) => r.targetId === "memory.peak")?.hi).toBe(130);
		// A target with no estimate at all reads as abstaining, not as zero.
		expect(plan.readouts.find((r) => r.targetId === "cost.total")?.status).toBe("abstain");
	});

	it("gives a scheduler the upper end of the trusted wall time", () => {
		const plan = exportPlan(record, at({}), { "time.total": est("predicted", 3600, 3000, 4500) }, null, null);
		expect(plan.wallTimeHiS).toBe(4500);
	});

	it("calls code changes verified only when a passed patch matches each one", () => {
		const knob = record.knobs.find((k) => k.id === "eval_no_grad");
		expect(knob?.needs_code).toBe(true);
		const moved = at({ eval_no_grad: !knob?.current });
		expect(exportPlan(record, moved, {}, null, null).codeChangesVerified).toBe(false);
		const passed = { ...record, patches: [{ id: "p", kind: "template" as const, template_id: knob?.patch_template ?? null, title: "", diff: "", verify: { status: "passed" as const } }] };
		expect(exportPlan(passed, moved, {}, null, null).codeChangesVerified).toBe(true);
		const failed = { ...passed, patches: [{ ...passed.patches[0]!, verify: { status: "failed" as const } }] };
		expect(exportPlan(failed, moved, {}, null, null).codeChangesVerified).toBe(false);
		// Nothing that needs code moved: nothing to verify.
		expect(exportPlan(record, at({ batch_size: 2 }), {}, null, null).codeChangesVerified).toBe(true);
	});

	it("links the record read-only by the ref the cockpit opened it with", () => {
		expect(exportPlan(record, at({}), {}, "abc def", "https://x.test/").shareUrl).toBe("https://x.test/#/cockpit/abc%20def");
		expect(exportPlan(record, at({}), {}, null, "https://x.test/").shareUrl).toBeNull();
	});

	it("takes the GPU from the knob when the visitor moved it", () => {
		expect(exportPlan(record, at({}), {}, null, null).gpu.name).toBe(String(record.knobs.find((k) => k.id === "gpu_type")?.current));
		expect(exportPlan(record, at({ gpu_type: "A100-SXM-80GB" }), {}, null, null).gpu.name).toBe("A100-SXM-80GB");
	});
});
