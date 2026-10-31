import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import realRaw from "../../../../api/tests/fixtures/a-smoke-l4-main-4.record.json";
import flags from "../../../../flags.json";
import { type FlightRecord, type KnobValue, normaliseRecord } from "../../workspace/record/types";
import { ExportsPanel } from "./ExportsPanel";
import { CLI_FLAG } from "./flags";
import { registerExportsStrings } from "./strings";

vi.mock("../../lib/flags", () => ({ flag: (name: string) => name === "c.exports" || name === "d.exports_cli" }));

const sample = normaliseRecord(realRaw as unknown as FlightRecord);
// The same job checked in from a repo: the case a visitor can re-check on their own machine.
const real = { ...sample, job: { ...sample.job, source: { ...sample.job.source, kind: "repo" as const } } };
const knobs = Object.fromEntries(real.knobs.map((k) => [k.id, k.current as KnobValue]));

describe("ExportsPanel with d.exports_cli on (the repo is public)", () => {
	beforeAll(registerExportsStrings);
	afterEach(cleanup);

	it("is off in flags.json until Chiya turns it on", () => {
		expect((flags as Record<string, unknown>)[CLI_FLAG]).toBe(false);
	});

	it("shows the install line and the re-check command, before the link", () => {
		render(<ExportsPanel record={real} knobs={knobs} estimates={{}} />);
		const text = screen.getByLabelText("Check it again on your machine").textContent ?? "";
		expect(text).toContain("pip install git+https://github.com/uchuuronin/preflight\n");
		expect(text).toContain("preflight run . --entrypoint=train.py --out .preflight/records\n");
		const copies = screen.getAllByRole("button", { name: /^Copy:/ }).map((b) => b.getAttribute("aria-label"));
		expect(copies.slice(-2)).toEqual(["Copy: Check it again on your machine", "Copy: Link to this record"]);
	});

	it("isn't offered for a sample, which has no folder on the visitor's machine", () => {
		render(<ExportsPanel record={sample} knobs={knobs} estimates={{}} />);
		expect(screen.queryByRole("button", { name: "Copy: Check it again on your machine" })).toBeNull();
	});
});
