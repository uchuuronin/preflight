// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
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

	it("sends notebooks to that tab instead of the Upload tab", () => {
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: Panel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getByRole("tab", { name: "Upload" }));
		const input = document.querySelector("input[type=file]") as HTMLInputElement;
		expect(input.accept).not.toContain(".ipynb");
		expect(input.accept).toContain(".py");
	});
});
