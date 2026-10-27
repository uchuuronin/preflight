import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Knob } from "../../workspace/record/types";
import { GlossaryCard } from "./GlossaryCard";
import { resetTierForTests, setTier } from "./store";
import { registerExpertiseStrings } from "./strings";

function knob(id: string): Knob {
	return { id, label_key: `knob.${id}`, kind: "int", current: 1 };
}

describe("GlossaryCard", () => {
	beforeEach(() => {
		localStorage.clear();
		resetTierForTests();
		registerExpertiseStrings();
	});
	afterEach(cleanup);

	it("renders nothing for a knob the glossary does not know", () => {
		const { container } = render(<GlossaryCard knob={knob("mystery")} />);
		expect(container.textContent).toBe("");
	});

	it("explains the knob at the chosen level behind a disclosure", () => {
		render(<GlossaryCard knob={knob("eval_batch_size")} />);
		const summary = screen.getByText("What this setting does");
		expect(summary.closest("details")).toBeTruthy();
		expect(screen.getByText(/Eval often skips no_grad/)).toBeTruthy();
	});

	it("follows the detail level as it changes", () => {
		render(<GlossaryCard knob={knob("eval_batch_size")} />);
		act(() => setTier("new"));
		expect(screen.getByText(/crash a run that trained fine/)).toBeTruthy();
		act(() => setTier("expert"));
		expect(screen.getByText(/the usual first-eval OOM/)).toBeTruthy();
	});

	it("says what to change when there is a clear fix", () => {
		render(<GlossaryCard knob={knob("eval_batch_size")} />);
		expect(screen.getByText("What to change")).toBeTruthy();
		expect(screen.getByText(/Lower it, or wrap eval in no_grad/)).toBeTruthy();
	});

	it("leaves out the fix line when there is none", () => {
		render(<GlossaryCard knob={knob("batch_size")} />);
		expect(screen.queryByText("What to change")).toBeNull();
	});

	it("shows one level deeper on request, without changing the setting", () => {
		act(() => setTier("new"));
		render(<GlossaryCard knob={knob("seq_len")} />);
		fireEvent.click(screen.getByRole("button", { name: "More detail" }));
		expect(screen.getByText(/attention cost grows faster/)).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "More detail" }));
		expect(screen.getByText(/quadratic without/)).toBeTruthy();
		expect(screen.queryByRole("button", { name: "More detail" })).toBeNull();
	});

	it("offers no deeper step at expert", () => {
		act(() => setTier("expert"));
		render(<GlossaryCard knob={knob("seq_len")} />);
		expect(screen.queryByRole("button", { name: "More detail" })).toBeNull();
	});
});
