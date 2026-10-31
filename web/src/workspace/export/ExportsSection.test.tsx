// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

let exportsOn = true;
vi.mock("../../lib/flags", () => ({ flag: (name: string) => name === "c.exports" && exportsOn }));

import devRaw from "../fixtures/dev-finetune.record.json";
import { type ExportPlan, registerExportsPanel, unregister } from "../../registry";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { ExportsSection } from "./ExportsSection";

const record = normaliseRecord(devRaw as unknown as FlightRecord);
let seen: ExportPlan | null = null;
function Panel({ plan }: { plan: ExportPlan }) {
	seen = plan;
	return <p>config for {plan.lens}</p>;
}

const open = () =>
	render(
		<WorkspaceProvider record={record} recordRef="sample:dev" provider={new GridProvider()}>
			<ExportsSection />
		</WorkspaceProvider>,
	);

afterEach(() => {
	cleanup();
	unregister("d.exports");
	exportsOn = true;
	seen = null;
});

describe("the exports hook", () => {
	it("hands a registered panel the plan at the current setting", () => {
		registerExportsPanel({ id: "d.exports", titleKey: "exports.title", component: Panel });
		open();
		expect(screen.getByRole("region", { name: "Exports" })).toBeTruthy();
		expect(screen.getByText("config for finetune")).toBeTruthy();
		expect(seen?.changes).toEqual([]);
		expect(seen?.shareUrl).toMatch(/#\/cockpit\/sample%3Adev$/);
	});

	it("renders nothing while the flag is off or no panel is registered", () => {
		open();
		expect(screen.queryByRole("region", { name: "Exports" })).toBeNull();
		cleanup();
		registerExportsPanel({ id: "d.exports", titleKey: "exports.title", component: Panel });
		exportsOn = false;
		open();
		expect(screen.queryByRole("region", { name: "Exports" })).toBeNull();
	});
});
