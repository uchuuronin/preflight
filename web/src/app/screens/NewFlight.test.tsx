// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/flags", () => ({ flag: (name: string) => name === "runner.queue" }));

import { type IntakePanelProps, registerAppPanel, unregister } from "../../registry";
import { NewFlight } from "./NewFlight";

const SOURCE = { kind: "notebook", files: { "train.ipynb": '{"cells":[]}' } };

function Panel({ setSource }: IntakePanelProps) {
	return (
		<button type="button" onClick={() => setSource?.(SOURCE)}>
			hand over
		</button>
	);
}

afterEach(() => {
	cleanup();
	unregister("notebook.intake");
	vi.unstubAllGlobals();
	try {
		localStorage.clear();
	} catch {
		// no storage in this environment
	}
});

describe("the check-in desk and intake panels", () => {
	it("checks in the source an intake panel hands over, with the goal and GPU", async () => {
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: Panel });
		const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "01JOB" }), { status: 202 }));
		vi.stubGlobal("fetch", fetch);
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		fireEvent.click(screen.getByRole("checkbox"));
		const checkIn = screen.getByRole("button", { name: "Check in" }) as HTMLButtonElement;
		expect(checkIn.disabled).toBe(true); // nothing handed over yet
		fireEvent.click(screen.getByRole("button", { name: "hand over" }));
		expect(checkIn.disabled).toBe(false);
		await act(async () => {
			fireEvent.click(checkIn);
		});
		const body = JSON.parse(fetch.mock.calls[0][1].body as string);
		expect(body.source).toEqual(SOURCE);
		expect(body.objective.kind).toBe("min_cost");
		expect(body.gpu).toBe("A100-SXM-80GB");
	});

	it("drops a tab's source when the visitor leaves it, since the panel forgets it too", () => {
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: Panel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		fireEvent.click(screen.getByRole("checkbox"));
		fireEvent.click(screen.getByRole("button", { name: "hand over" }));
		const checkIn = () => screen.getByRole("button", { name: "Check in" }) as HTMLButtonElement;
		expect(checkIn().disabled).toBe(false);
		fireEvent.click(screen.getAllByRole("tab")[0]);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		expect(checkIn().disabled).toBe(true);
	});

	it("gives a panel the same setSource on every render, so calling it from an effect can't loop", () => {
		const seen = new Set<unknown>();
		function EffectPanel({ setSource }: IntakePanelProps) {
			seen.add(setSource);
			useEffect(() => {
				setSource?.(SOURCE);
			}, [setSource]);
			return <span>effect panel</span>;
		}
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: EffectPanel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		expect(screen.getByText("effect panel")).toBeTruthy();
		expect(seen.size).toBe(1);
	});

	it("refuses a handed-over source over the size limit, before anything is sent", () => {
		const big = { kind: "notebook", files: { "train.ipynb": "x".repeat(5_000_001) } };
		function BigPanel({ setSource }: IntakePanelProps) {
			return (
				<button type="button" onClick={() => setSource?.(big)}>
					hand over big
				</button>
			);
		}
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: BigPanel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		fireEvent.click(screen.getByRole("checkbox"));
		fireEvent.click(screen.getByRole("button", { name: "hand over big" }));
		expect((screen.getByRole("button", { name: "Check in" }) as HTMLButtonElement).disabled).toBe(true);
		expect(screen.getByRole("alert").textContent).toMatch(/more than 5 MB/);
	});

	it("sends notebooks to that tab instead of the Upload tab", () => {
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: Panel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getByRole("tab", { name: "Upload" }));
		const input = document.querySelector("input[type=file]") as HTMLInputElement;
		expect(input.accept).not.toContain(".ipynb");
		expect(input.accept).toContain(".py");
	});
});
