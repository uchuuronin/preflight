import { cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import realRaw from "../../../../api/tests/fixtures/a-smoke-l4-main-4.record.json";
import goldRaw from "../../../../schema/examples/finetune.json";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useRegistry } from "../../registry";
import { type FlightRecord, type KnobValue, normaliseRecord } from "../../workspace/record/types";
import { slurmTime } from "./build";
import { ExportsPanel } from "./ExportsPanel";
import { FLAG } from "./index";
import { registerExportsStrings } from "./strings";

const real = normaliseRecord(realRaw as unknown as FlightRecord);
const gold = normaliseRecord(goldRaw as unknown as FlightRecord);

function knobsOf(r: typeof real, set: Record<string, KnobValue> = {}) {
	return { ...Object.fromEntries(r.knobs.map((k) => [k.id, k.current as KnobValue])), ...set };
}

describe("exports registration", () => {
	it("adds a band under the cockpit's whole run, behind the existing c.exports flag", () => {
		const { result } = renderHook(() => useRegistry());
		const panel = result.current.panels.find((p) => p.id === "exports.plan");
		expect(panel).toMatchObject({ slot: "below", flag: "c.exports", component: ExportsPanel });
		expect(panel?.lenses).toBeUndefined();
		expect(FLAG).toBe("c.exports");
		expect(t(panel?.titleKey ?? "")).toBe("Take this plan with you");
	});
});

describe("ExportsPanel", () => {
	beforeAll(registerExportsStrings);
	afterEach(() => {
		cleanup();
		window.location.hash = "";
	});

	it("offers the five exports, each with copy, and downloads for the three files", () => {
		window.location.hash = "#/cockpit/ICDYrfHQ2nBNbv3mlRHHMg";
		render(<ExportsPanel record={real} knobs={knobsOf(real)} estimates={real.predictions.estimates} />);
		const names = ["Plan", "Edits to your script", "Slurm batch file", "Command", "Link to this record"];
		for (const n of names) expect(screen.getByRole("button", { name: `Copy: ${n}` })).toBeTruthy();
		const downloads = screen.getAllByRole("button", { name: /^Download:/ }).map((b) => b.getAttribute("aria-label"));
		expect(downloads).toEqual(["Download: preflight-plan.json", "Download: preflight-edits.sh", "Download: preflight.sbatch"]);
		// The CLI item stays hidden while d.exports_cli is off (the repo is private until Nov 3 to 4).
		expect(screen.queryByRole("button", { name: "Copy: Check it again on your machine" })).toBeNull();
		expect(document.body.textContent).not.toContain("pip install");
		expect(screen.getByLabelText("Link to this record").textContent).toBe(`${window.location.origin}/#/cockpit/ICDYrfHQ2nBNbv3mlRHHMg`);
	});

	it("moves with the knobs: a changed batch size shows in the edits", () => {
		const { rerender } = render(<ExportsPanel record={real} knobs={knobsOf(real)} estimates={{}} />);
		expect(screen.getByText(/Nothing changed from the record's settings yet/)).toBeTruthy();
		rerender(<ExportsPanel record={real} knobs={knobsOf(real, { batch_size: 8, precision: "fp16", packing: true })} estimates={{}} />);
		expect(
			screen.getByText(
				"3 settings changed: 1 made by the script, 1 to make by hand, 1 needs a code change from Muse Code. Run it with sh preflight-edits.sh from your project's top folder. It checks every line first and changes nothing if one has moved since Preflight read it.",
			),
		).toBeTruthy();
		expect(screen.getByLabelText("Edits to your script").textContent).toContain("per_device_train_batch_size=8");
	});

	it("sizes the batch file from the time estimate, and says so when there is none", () => {
		const { rerender } = render(<ExportsPanel record={gold} knobs={knobsOf(gold)} estimates={gold.predictions.estimates} />);
		const est = gold.predictions.estimates["time.total"];
		const asked = Math.ceil(((est?.hi ?? 0) + 15 * 60) / 300) * 300;
		expect(screen.getByText(`Asks for 1 x L4 and ${fmt(asked, "s")}: the slow end of the predicted range plus 15 minutes to set up.`)).toBeTruthy();
		expect(screen.getByLabelText("Slurm batch file").textContent).toContain(`#SBATCH --time=${slurmTime(asked)}`);
		rerender(<ExportsPanel record={gold} knobs={knobsOf(gold)} estimates={{}} />);
		expect(screen.getByText("Asks for 1 x L4. Preflight has no time estimate at these settings, so set the time limit yourself.")).toBeTruthy();
	});

	it("asks for no GPU on an API run, with or without a time estimate", () => {
		const api = { ...gold, env: { ...gold.env, backend: "api" } } as typeof gold;
		const { rerender } = render(<ExportsPanel record={api} knobs={knobsOf(gold)} estimates={{}} />);
		expect(screen.getByText("Asks for no GPU. Preflight has no time estimate at these settings, so set the time limit yourself.")).toBeTruthy();
		rerender(<ExportsPanel record={api} knobs={knobsOf(gold)} estimates={gold.predictions.estimates} />);
		expect(screen.getByText(/^Asks for no GPU and /)).toBeTruthy();
	});

	it("says when the plan is longer than the time limit, as the file does", () => {
		const capped = { ...gold, job: { ...gold.job, constraints: { ...gold.job.constraints, wall_time_cap_s: 3600 } } } as typeof gold;
		render(<ExportsPanel record={capped} knobs={knobsOf(gold)} estimates={gold.predictions.estimates} />);
		expect(screen.getByText(/^Asks for 1 x L4 and 1 h, your time limit\. The plan needs about .+, so save checkpoints and resubmit to finish\.$/)).toBeTruthy();
		expect(screen.getByLabelText("Slurm batch file").textContent).toContain("#SBATCH --time=01:00:00");
	});

	it("says a sample's numbers are illustrative", () => {
		render(<ExportsPanel record={gold} knobs={knobsOf(gold)} estimates={gold.predictions.estimates} />);
		expect(screen.getByText(/This is a sample: its numbers are illustrative/)).toBeTruthy();
	});

	it("copies, and says so to screen readers; says what to do if copying fails", async () => {
		const writeText = vi.fn().mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error("denied"));
		Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
		render(<ExportsPanel record={real} knobs={knobsOf(real)} estimates={{}} />);
		fireEvent.click(screen.getByRole("button", { name: "Copy: Command" }));
		await waitFor(() => expect(screen.getByRole("status").textContent).toBe("Copied: Command."));
		expect(writeText).toHaveBeenCalledWith("python train.py");
		fireEvent.click(screen.getByRole("button", { name: "Copy: Plan" }));
		await waitFor(() => expect(screen.getByRole("status").textContent).toBe("Couldn't copy. Select the text and copy it instead."));
	});

	it("lists Muse Code patches with their verify status", () => {
		const withPatches = {
			...real,
			patches: [
				{ id: "p1", kind: "template" as const, title: "Pack sequences", diff: "blob://p1", verify: { status: "passed" as const } },
				{ id: "p2", kind: "open_ended" as const, title: "Wrap eval in no_grad", diff: "blob://p2" },
			],
		};
		render(<ExportsPanel record={withPatches} knobs={knobsOf(real)} estimates={{}} />);
		expect(screen.getByText("Pack sequences · Verified")).toBeTruthy();
		expect(screen.getByText("Wrap eval in no_grad · Not verified yet")).toBeTruthy();
	});
});
