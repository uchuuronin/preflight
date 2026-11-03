// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import en from "../../i18n/en.json";
import devRaw from "../fixtures/dev-finetune.record.json";
import { Provenance, provenanceKey } from "../components/StatusMark";
import { GridProvider } from "../provider/GridProvider";
import { isFixture } from "../record/model";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { Measured } from "./Measured";

const sample = normaliseRecord(devRaw as unknown as FlightRecord);
const raw = devRaw as unknown as FlightRecord;
// The same record as if B's engine had predicted it: no fixture model ids anywhere.
const real = (e: { model_id: string } | undefined) => (e ? { ...e, model_id: "memory-analytical@0.2.0" } : e);
const live = normaliseRecord({
	...raw,
	models: [],
	predictions: {
		...raw.predictions,
		estimates: Object.fromEntries(Object.entries(raw.predictions?.estimates ?? {}).map(([k, e]) => [k, real(e)])),
		grids: (raw.predictions?.grids ?? []).map((g) => ({ ...g, model_id: "memory-analytical@0.2.0" })),
	},
} as unknown as FlightRecord);

const inCockpit = (record: typeof sample, node: React.ReactNode) =>
	render(
		<WorkspaceProvider record={record} provider={new GridProvider()}>
			{node}
		</WorkspaceProvider>,
	);

afterEach(cleanup);

describe("samples never read as measured", () => {
	it("uses the illustrative word for every status but an abstention", () => {
		expect(provenanceKey("measured", true)).toBe("provenance.illustrative");
		expect(provenanceKey("predicted", true)).toBe("provenance.illustrative");
		expect(provenanceKey("abstain", true)).toBe("provenance.abstain");
		expect(provenanceKey("measured", false)).toBe("provenance.measured");
	});

	it("says Example numbers under a reading on a sample, Measured on a live record", () => {
		expect(isFixture(sample)).toBe(true);
		expect(isFixture(live)).toBe(false);
		inCockpit(sample, <Provenance status="measured" />);
		expect(screen.getByText(en["provenance.illustrative"])).toBeTruthy();
		cleanup();
		inCockpit(live, <Provenance status="measured" />);
		expect(screen.getByText(en["provenance.measured"])).toBeTruthy();
	});

	it("labels a sample's probes as examples and shows none of their made-up costs", () => {
		inCockpit(sample, <Measured />);
		const capsule = screen.getByRole("button", { expanded: false });
		expect(capsule.textContent).toMatch(/example probes/);
		expect(capsule.textContent).not.toMatch(/\$/);
		fireEvent.click(capsule);
		const drawer = screen.getByRole("complementary");
		expect(drawer.textContent).not.toMatch(/\$/);
	});
});
