import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { t } from "../../lib/i18n";
import { FindingCitations, findingSourceKey } from "./findingSource";

const line = (f: Parameters<typeof findingSourceKey>[0]) => t(findingSourceKey(f));

describe("who found a finding", () => {
	it("names Muse Glimmer for a private check-in's triage, as the live record titles it (Nov 2)", () => {
		expect(line({ source: "spark_open_triage", code: "triage.oom", title: "Muse Glimmer on probe p1-eval: oom" })).toBe("Found by Muse Glimmer");
	});
	it("keeps Spark Open's line for its own triage, and other sources as they were", () => {
		expect(line({ source: "spark_open_triage", code: "triage.oom", title: "Spark Open on probe p3: oom" })).toBe("Found by Muse Spark Open triage");
		expect(line({ source: "rule", code: "eval.no_grad", title: "Muse Glimmer on probe p1: oom" })).toBe("Found by a rule check");
	});
	it("labels a diagnosis by what checked it, as A's claim check writes it", () => {
		expect(line({ source: "spark_diagnosis", code: "diagnosis.mismatch", title: "Reserved climbs", evidence: [{ metric: "memory.reserved", value: 9.1e9 }], detail: "Muse Spark wrote: Spark Open checked it against nothing" })).toBe("Found by Muse Spark diagnosis");
		expect(line({ source: "spark_diagnosis", code: "diagnosis.mismatch", title: "Reserved climbs", evidence: [{ metric: "memory.reserved", value: 9.1e9 }, { metric: "spark_open_check", value: "p1.step_log[12].reserved" }] })).toBe(
			"Found by Muse Spark diagnosis, checked by Muse Spark Open",
		);
		expect(line({ source: "spark_diagnosis", code: "diagnosis.abstain", title: "Not backed by a probe number: abstain, or run a probe (Reserved climbs)" })).toBe("Unsupported: run a probe");
	});
});

describe("a finding's sources", () => {
	it("are links that open apart from the cockpit, and only web addresses link", () => {
		const { container } = render(
			<FindingCitations
				finding={{
					citations: [
						{ title: "PyTorch: CUDA memory", url: "https://pytorch.org/docs/stable/notes/cuda.html" },
						{ title: "Not a link", url: "javascript:alert(1)" },
						{ title: "No address" },
					],
				}}
			/>,
		);
		const links = container.querySelectorAll("a");
		expect(links).toHaveLength(1);
		expect(links[0]!.getAttribute("href")).toBe("https://pytorch.org/docs/stable/notes/cuda.html");
		expect(links[0]!.getAttribute("rel")).toBe("noopener noreferrer");
		expect(container.textContent).toContain("Not a link");
		expect(container.textContent).toContain("No address");
	});
	it("draw nothing when there are none", () => {
		expect(render(<FindingCitations finding={{}} />).container.innerHTML).toBe("");
	});
});
