// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { FindingsBand, FindingsChip, looseFindings, splitQuote } from "./Findings";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);

// As A's harness/triage.py writes one for a probe that didn't finish: info, no node, the log line
// quoted inside the detail.
const triage = {
	id: "triage.p3",
	severity: "info" as const,
	code: "triage.out_of_memory",
	title: "Spark Open on probe p3: out of memory",
	detail: 'The eval pass ran out of GPU memory at batch 32. (log: "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")',
	node_id: null,
	evidence: [{ metric: "triage", value: 1.0, probe_id: "p3" }],
	source: "spark_open_triage" as const,
};

const withTriage = { ...dev, findings: [...dev.findings, triage] };

const open = (record = withTriage) =>
	render(
		<WorkspaceProvider record={record} provider={new GridProvider()}>
			<FindingsChip />
			<FindingsBand />
		</WorkspaceProvider>,
	);

afterEach(cleanup);

describe("findings with no place on the map", () => {
	it("splits triage's detail into what it says and the log line it quotes", () => {
		expect(splitQuote(triage.detail)).toEqual({
			text: "The eval pass ran out of GPU memory at batch 32.",
			quote: "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
			unquoted: false,
		});
		expect(splitQuote("The probe stalled. (no log line quoted)")).toEqual({ text: "The probe stalled.", quote: null, unquoted: true });
		expect(splitQuote("Plain detail.")).toEqual({ text: "Plain detail.", quote: null, unquoted: false });
	});

	it("lists a triage finding with its severity, text, quoted line and who found it", () => {
		const { container } = open();
		const row = [...container.querySelectorAll(".pf-findings .pf-finding")].find((r) => r.textContent?.includes("Spark Open on probe p3"))!;
		expect(row).toBeTruthy();
		expect(row.querySelector(".pf-findings__sev")!.textContent).toBe("Info");
		expect(row.querySelector(".pf-findings__quote")!.textContent).toBe("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB");
		expect(row.textContent).toContain("Found by Muse Spark Open triage");
		// The triage marker isn't evidence a reader needs.
		expect(row.querySelector(".pf-findings__evidence")).toBeNull();
		expect(container.querySelector(".pf-findings-chip")!.textContent).toMatch(/finding/);
	});

	it("leaves a finding on a drawn node to that node's sheet", () => {
		const drawn = new Set(dev.graph.nodes.map((n) => n.id));
		const onNode = dev.findings.filter((f) => f.node_id && drawn.has(f.node_id));
		const loose = looseFindings(withTriage, drawn);
		for (const f of onNode) expect(loose).not.toContain(f);
		expect(loose.map((f) => f.id)).toContain("triage.p3");
	});

	it("draws nothing when every finding has its node", () => {
		const none = { ...dev, findings: [] };
		const { container } = open(none);
		expect(container.querySelector(".pf-findings")).toBeNull();
		expect(container.querySelector(".pf-findings-chip")).toBeNull();
	});
});
