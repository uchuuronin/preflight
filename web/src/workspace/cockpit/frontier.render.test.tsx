// @vitest-environment jsdom
import { cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import agentRaw from "../fixtures/agent-eval-oct31.record.json";
import { frontierPoints } from "../frontier";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { CostFrontier } from "./CostFrontier";

const measured = normaliseRecord(agentRaw as unknown as FlightRecord);

/**
 * A stand-in sweep for the test only: A's 30 real tasks, re-labelled across three models, with
 * success verdicts set by rule (the real sample has none scored). Not shown anywhere.
 */
function sweep(): ReturnType<typeof normaliseRecord> {
	const tasks = measured.agent_sample?.tasks ?? [];
	const models = ["muse-spark-1.3", "muse-spark-open", "muse-glimmer"];
	// Spark: dear and right most often; Open: cheaper, right less often; Glimmer: dearer than Open and wrong more (beaten).
	const rule: Record<string, { costX: number; rightEvery: number }> = {
		"muse-spark-1.3": { costX: 1, rightEvery: 2 },
		"muse-spark-open": { costX: 0.3, rightEvery: 3 },
		"muse-glimmer": { costX: 0.5, rightEvery: 5 },
	};
	const all = models.flatMap((m) =>
		tasks.map((task, i) => ({ ...task, task_id: `${m}-${task.task_id}`, model: m, cost_usd: (task.cost_usd ?? 0.01) * rule[m]!.costX, success: i % rule[m]!.rightEvery === 0 })),
	);
	return { ...measured, agent_sample: { ...measured.agent_sample!, tasks: all } };
}

const open = (record: ReturnType<typeof normaliseRecord>) =>
	render(
		<WorkspaceProvider record={record} provider={new GridProvider()}>
			<CostFrontier />
		</WorkspaceProvider>,
	);

afterEach(cleanup);

describe("the cost frontier", () => {
	it("draws nothing for A's one-model run, whose tasks carry no success verdict", () => {
		expect(frontierPoints(measured)).toHaveLength(0);
		expect(open(measured).container.querySelector(".pf-frontier")).toBeNull();
	});

	it("puts each model at its mean cost and Wilson success rate, and marks the one beaten on both", () => {
		const pts = frontierPoints(sweep());
		expect(pts.map((p) => p.model)).toEqual(["muse-spark-open", "muse-glimmer", "muse-spark-1.3"]);
		const g = pts.find((p) => p.model === "muse-glimmer")!;
		const spark = pts.find((p) => p.model === "muse-spark-1.3")!;
		expect(spark.success.k).toBe(15);
		expect(spark.success.n).toBe(30);
		expect(spark.success.lo).toBeLessThan(0.5);
		expect(spark.success.hi).toBeGreaterThan(0.5);
		expect(spark.costLo).toBeLessThan(spark.costMean);
		// Glimmer costs more than Open and solves fewer: beaten on both.
		expect(g.success.p).toBeLessThan(pts.find((p) => p.model === "muse-spark-open")!.success.p);
		expect(g.dominated).toBe(true);
	});

	it("names every point, gives each a spoken summary, and has a table", () => {
		const { container } = open(sweep());
		expect(container.querySelectorAll(".pf-frontier__pt")).toHaveLength(3);
		expect(container.querySelectorAll(".pf-frontier__pt.is-dominated")).toHaveLength(1);
		expect(container.querySelector(".pf-frontier__label")?.textContent).toBe("Muse Spark Open");
		const pt = container.querySelector(".pf-frontier__pt")!;
		expect(pt.getAttribute("aria-label")).toMatch(/Muse Spark Open: .* a task .*solved \d+ of 30/);
		fireEvent.focus(pt);
		expect(container.querySelector(".pf-frontier__tip")?.textContent).toContain("Muse Spark Open");
		expect(container.querySelectorAll(".pf-frontier__table tbody tr")).toHaveLength(3);
	});
});
