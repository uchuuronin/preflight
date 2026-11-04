import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { t } from "../../lib/i18n";
import { FindingCitations, findingSourceKey, plainDetail } from "./findingSource";

const line = (f: Parameters<typeof findingSourceKey>[0]) => t(findingSourceKey(f));

describe("who found a finding", () => {
	it("names Muse Glimmer for a private check-in's triage, as the live record titles it (Nov 2)", () => {
		expect(line({ source: "spark_open_triage", code: "triage.oom", title: "Muse Glimmer on probe p1-eval: oom" })).toBe("Found by Muse Glimmer, on Preflight's own GPU");
	});
	it("keeps Spark Open's line for its own triage, and other sources as they were", () => {
		expect(line({ source: "spark_open_triage", code: "triage.oom", title: "Spark Open on probe p3: oom" })).toBe("Found by Muse Spark Open triage, on Baseten");
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

describe("Muse Spark's web search", () => {
	it("says the search found it, for a known issue and for none found, and leaves prediction models as they were", () => {
		expect(line({ source: "model", code: "search.known_issue", title: "Known issue: bitsandbytes on this CUDA" })).toBe("Muse Spark searched the web");
		expect(line({ source: "model", code: "search.none_found", title: "No known issue matched" })).toBe("Muse Spark searched the web");
		expect(line({ source: "model", code: "divergence.risk", title: "Loss may diverge" })).toBe("Flagged by a prediction model");
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

describe("plainDetail (PM's audit, Nov 3)", () => {
	it("keeps the diagnosis's sentence and drops the claim check's internals", () => {
		const d =
			"The loss plot shows a pronounced peak on the p1 line in this region that lines up with the logged series entry for that step. (plot: loss, steps 8-10, the p1 line peak; number: probes[p1].stability.series.loss[9] = 1.5649110525846481) Spark Open checked it against probes[p1].stability.series.loss[9] = 1.5649110525846481. (10: unsupported: run a probe)";
		expect(plainDetail(d)).toEqual({
			text: "The loss plot shows a pronounced peak on the p1 line in this region that lines up with the logged series entry for that step.",
			unsupported: true,
		});
	});
	it("leaves an ordinary detail alone", () => {
		expect(plainDetail("Eval runs with gradients kept, so it holds every layer's activations.")).toEqual({ text: "Eval runs with gradients kept, so it holds every layer's activations.", unsupported: false });
	});
});

describe("a finding's detail in words, more shapes (review, Nov 3)", () => {
	it("drops a plot region that holds parentheses, and an abstained check's reason", () => {
		expect(plainDetail("Loss spikes after warmup. (plot: loss (smoothed), steps 8-10; number: probes[p1].loss[9] = 1.56)").text).toBe("Loss spikes after warmup.");
		expect(plainDetail("Reserved climbs. (plot: memory, steps 1-4; number: p1.reserved = 9.1e9) Spark Open's check: no number in the probes matches.").text).toBe("Reserved climbs.");
	});
});
