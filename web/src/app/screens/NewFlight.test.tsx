// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/flags", () => ({ flag: (name: string) => name === "runner.queue" }));

import { type IntakePanelProps, registerAppPanel, unregister } from "../../registry";
import en from "../../i18n/en.json";
import { DATA_USE_OK_ID } from "../DataUse";
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

	it("says why Check in is grey when the link isn't a GitHub repo, and marks the field", () => {
		render(<NewFlight onSample={() => {}} />);
		const field = screen.getByLabelText("GitHub repository") as HTMLInputElement;
		expect(screen.queryByText(en["newflight.repo.invalid"])).toBeNull(); // nothing typed: no complaint
		for (const bad of ["not a link", "https://gitlab.com/a/b", "github.com/owner"]) {
			fireEvent.change(field, { target: { value: bad } });
			expect(screen.queryByText(en["newflight.repo.invalid"])).toBeNull(); // still typing
			fireEvent.blur(field);
			const msg = screen.getByText(en["newflight.repo.invalid"]);
			expect(msg.id).not.toBe("");
			expect(field.getAttribute("aria-describedby")?.split(" ")).toContain(msg.id);
			expect(field.getAttribute("aria-invalid")).toBe("true");
			expect(msg.closest("[role=status]")).not.toBeNull(); // spoken when it appears
		}
		fireEvent.change(field, { target: { value: "github.com/karpathy/nanoGPT" } });
		fireEvent.blur(field);
		expect(screen.queryByText(en["newflight.repo.invalid"])).toBeNull();
		expect(field.getAttribute("aria-invalid")).toBeNull();
	});

	it("takes the visitor to the Got it box, then back to Check in once it's ticked", async () => {
		render(<NewFlight onSample={() => {}} />);
		Element.prototype.scrollIntoView = vi.fn();
		fireEvent.change(screen.getByLabelText("GitHub repository"), { target: { value: "github.com/karpathy/nanoGPT" } });
		const box = document.getElementById(DATA_USE_OK_ID) as HTMLInputElement;
		fireEvent.click(screen.getByRole("button", { name: en["datause.needed"] }));
		expect(document.activeElement).toBe(box);
		fireEvent.click(box);
		await act(async () => {
			await new Promise((r) => requestAnimationFrame(() => r(null)));
		});
		expect(document.activeElement).toBe(screen.getByRole("button", { name: "Check in" }));
	});

	it("states the measured run, with no dollar promise, once the notice is ticked", () => {
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getByRole("checkbox"));
		const hint = screen.getByText(en["newflight.cost_hint"]);
		expect(hint.textContent).toMatch(/measured/i);
		expect(hint.textContent).not.toMatch(/\$|US\$|£/);
	});
});
