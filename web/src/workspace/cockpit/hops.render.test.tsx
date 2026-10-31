// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/flags", () => ({ flag: (name: string) => name === "c.hops" }));

import en from "../../i18n/en.json";
import devRaw from "../fixtures/dev-finetune.record.json";
import { FRAME_MS, setHops } from "../hops";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { MemoryBand } from "./Bands";
import { Dials } from "./Dials";

const record = normaliseRecord(devRaw as unknown as FlightRecord);
const open = () =>
	render(
		<WorkspaceProvider record={record} provider={new GridProvider()}>
			<Dials />
			<MemoryBand />
		</WorkspaceProvider>,
	);

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
	act(() => setHops(false));
	cleanup();
	vi.useRealTimers();
});

describe("the possible-outcomes toggle", () => {
	it("is off until pressed, then moves one outcome per frame on the dials", () => {
		const { container } = open();
		const toggle = screen.getByRole("button", { name: en["hops.toggle"] });
		expect(toggle.getAttribute("aria-pressed")).toBe("false");
		expect(container.querySelectorAll(".pf-dial__hop")).toHaveLength(0);
		fireEvent.click(toggle);
		expect(toggle.getAttribute("aria-pressed")).toBe("true");
		const first = container.querySelector(".pf-dial__hop")?.getAttribute("stroke-dasharray");
		expect(first).toBeTruthy();
		act(() => {
			vi.advanceTimersByTime(FRAME_MS);
		});
		expect(container.querySelector(".pf-dial__hop")?.getAttribute("stroke-dasharray")).not.toBe(first);
		expect(screen.getByText(en["hops.help"])).toBeTruthy();
	});

	it("never draws outcomes on the risk dial, whose number is the engine's own", () => {
		const { container } = open();
		fireEvent.click(screen.getByRole("button", { name: en["hops.toggle"] }));
		const risk = record.predictions.targets.find((t) => t.readout === "risk");
		const label = (en as Record<string, string>)[risk?.label_key ?? ""];
		expect(label).toBeTruthy();
		const dial = [...container.querySelectorAll(".pf-dial-wrap")].find((d) => d.querySelector(".pf-dial__label")?.textContent === label);
		expect(dial).toBeTruthy();
		expect(dial?.querySelector(".pf-dial__hop")).toBeNull();
		expect(container.querySelectorAll(".pf-dial__hop").length).toBeGreaterThan(0);
	});

	it("stops when the toggle leaves the screen", () => {
		const view = open();
		fireEvent.click(screen.getByRole("button", { name: en["hops.toggle"] }));
		view.unmount();
		const again = open();
		expect(screen.getByRole("button", { name: en["hops.toggle"] }).getAttribute("aria-pressed")).toBe("false");
		expect(again.container.querySelectorAll(".pf-dial__hop")).toHaveLength(0);
	});
});
