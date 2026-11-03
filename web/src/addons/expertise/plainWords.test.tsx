import { act, render, renderHook, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useRegistry } from "../../registry";
import devRaw from "../../workspace/fixtures/dev-finetune.record.json";
import agentRaw from "../../workspace/fixtures/agent-eval-oct31.record.json";
import { GridProvider } from "../../workspace/provider/GridProvider";
import { capacityBytes, currentKnobs, failureTarget, targetFor } from "../../workspace/record/model";
import { type FlightRecord, normaliseRecord } from "../../workspace/record/types";
import { FLAG } from "./index";
import { PlainLine } from "./PlainLine";
import { plainLine, plainParts } from "./plainWords";
import { setTier } from "./store";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const agent = normaliseRecord(agentRaw as unknown as FlightRecord);
const provider = new GridProvider();
type K = Record<string, string | number | boolean>;
const at = (knobs: Record<string, unknown>) => {
	const k = { ...currentKnobs(dev), ...knobs } as K;
	return { k, est: provider.predict(dev, k) };
};

describe("the beginner's line", () => {
	afterEach(() => act(() => setTier("practitioner")));

	it("says how much memory is left, then the time with its range, then the cost", () => {
		const { k, est } = at({ eval_batch_size: 8 });
		const parts = plainParts(dev, est, k);
		expect(parts[0]).toMatch(/^Fits on A100-SXM-80GB with .+ GiB to spare$/);
		expect(parts[1]).toMatch(/^about .+ \(.+ to .+\)$/);
		expect(parts[2]).toMatch(/^about US\$[\d.,]+$/);
		expect(plainLine(dev, est, k)).toBe(parts.join(" · "));
	});

	it("uses only numbers the readouts show: the spare memory is the card less the peak", () => {
		const { k, est } = at({ eval_batch_size: 8 });
		const memId = targetFor(dev, "memory")!.id;
		const cap = capacityBytes(dev, k)!;
		const spareGiB = (cap - est[memId]!.point!) / 2 ** 30;
		const shown = Number(/with ([\d.]+) GiB/.exec(plainParts(dev, est, k)[0]!)![1]);
		expect(Math.abs(shown - spareGiB)).toBeLessThan(0.1 * spareGiB + 0.05);
	});

	it("says a likely failure in the dials' words, and the time as what it would take to the end", () => {
		const { k, est } = at({});
		const parts = plainParts(dev, est, k);
		expect(parts[0]).toMatch(/^Likely to run out of memory on A100-SXM-80GB: about .+ GiB needed, 79.2 GiB on the card$/);
		// Time and cost are both what it would take to the end, as the dials say under both.
		expect(parts[1]).toMatch(/^if it ran to the end, about .+ \(.+ to .+\) and about US\$[\d.,]+$/);
		expect(parts).toHaveLength(2);
	});

	it("says an abstaining readout plainly instead of a number", () => {
		const { k, est } = at({ eval_batch_size: 8 });
		const timeId = targetFor(dev, "time")!.id;
		const costId = targetFor(dev, "cost")!.id;
		const memId = targetFor(dev, "memory")!.id;
		const out = { ...est, [timeId]: { ...est[timeId]!, status: "abstain" as const, point: null }, [costId]: { ...est[costId]!, status: "abstain" as const, point: null }, [memId]: { ...est[memId]!, status: "abstain" as const, point: null } };
		expect(plainLine(dev, out, k)).toBe("Memory on A100-SXM-80GB: not measured here · time: not measured here · cost: not measured here");
		const why = "the run's length isn't known: no max_steps";
		const unknown = { ...est, [timeId]: { ...est[timeId]!, status: "abstain" as const, point: null, reason: why }, [costId]: { ...est[costId]!, status: "abstain" as const, point: null, reason: why } };
		expect(plainParts(dev, unknown, k).slice(1)).toEqual(["time: run length unknown", "cost: run length unknown"]);
	});

	it("says a tight fit with what's left, and a peak past the card without spare memory", () => {
		const { k, est } = at({ eval_batch_size: 8 });
		const memId = targetFor(dev, "memory")!.id;
		const failId = failureTarget(dev)!;
		const cap = capacityBytes(dev, k)!;
		const tight = { ...est, [memId]: { ...est[memId]!, point: cap - 2 ** 30 }, [failId]: { ...est[failId]!, point: 0.3 } };
		expect(plainParts(dev, tight, k)[0]).toBe("Tight on A100-SXM-80GB: 1 GiB to spare");
		const over = { ...est, [memId]: { ...est[memId]!, point: cap + 2 ** 30 }, [failId]: { ...est[failId]!, point: 0.3 } };
		expect(plainParts(dev, over, k)[0]).toMatch(/^Tight on A100-SXM-80GB: a peak of about .+ GiB$/);
	});

	it("says where a probe ran out of memory at this setting", () => {
		const { k, est } = at({});
		const memId = targetFor(dev, "memory")!.id;
		const out = { ...est, [memId]: { ...est[memId]!, status: "abstain" as const, point: null, reason: "a probe ran out of memory at this setting" } };
		expect(plainParts(dev, out, k)[0]).toBe("Ran out of memory here on A100-SXM-80GB");
	});

	it("gives an agent run its risk dial in words, with no memory part", () => {
		const k = currentKnobs(agent) as K;
		const parts = plainParts(agent, provider.predict(agent, k), k);
		expect(parts.some((p) => /memory/i.test(p))).toBe(false);
		expect(parts[0]).toMatch(/^Context window filled: about \d+%$/);
		expect(parts.at(-1)).toMatch(/^chance of a runaway agent: \d+%$/);
		expect(parts.length).toBeGreaterThanOrEqual(2);
		expect(parts.join(" · ")).not.toMatch(/NaN|undefined|null|\{/);
	});

	it("gives no advice and no verdict of its own", () => {
		for (const knobs of [{}, { eval_batch_size: 8 }, { eval_batch_size: 4 }]) {
			const { k, est } = at(knobs);
			expect(plainLine(dev, est, k)).not.toMatch(/\b(should|try|recommend|better|worse|good|bad|safe|risky|need to)\b/i);
		}
	});

	it("shows at the New to this level only", () => {
		const { k, est } = at({ eval_batch_size: 8 });
		const { container, rerender } = render(<PlainLine record={dev} knobs={k} estimates={est} />);
		expect(container.textContent).toBe("");
		act(() => setTier("new"));
		rerender(<PlainLine record={dev} knobs={k} estimates={est} />);
		const line = container.querySelector("p.pf-addon-plainline")!;
		expect(line.textContent).toMatch(/^The readouts in plain words: Fits on A100-SXM-80GB/);
		expect(line.querySelector(".pf-visually-hidden")?.textContent).toBe("The readouts in plain words: ");
		expect(line.getAttribute("aria-live")).toBeNull();
		expect(screen.getByText(/^Fits on A100-SXM-80GB/)).toBe(line);
		act(() => setTier("expert"));
		rerender(<PlainLine record={dev} knobs={k} estimates={est} />);
		expect(container.textContent).toBe("");
	});

	it("registers in the cockpit head after the detail level, behind the expertise flag", () => {
		const { result } = renderHook(() => useRegistry());
		const level = result.current.panels.find((p) => p.id === "expertise.detail-level")!;
		const line = result.current.panels.find((p) => p.id === "expertise.plain-line")!;
		expect(line).toMatchObject({ slot: "cockpit-head", flag: FLAG, component: PlainLine });
		expect(line.order!).toBeGreaterThan(level.order!);
		expect(failureTarget(dev)).toBe("risk.oom");
	});
});
