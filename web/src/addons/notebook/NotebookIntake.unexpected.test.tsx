import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { registerNotebookStrings } from "./strings";

vi.mock("./convert", async (importOriginal) => ({
	...(await importOriginal<typeof import("./convert")>()),
	prepareNotebook: () => {
		throw new TypeError("boom");
	},
}));

const { NotebookIntake } = await import("./NotebookIntake");

describe("NotebookIntake when conversion fails unexpectedly", () => {
	beforeAll(registerNotebookStrings);
	afterEach(cleanup);

	it("shows an error and clears the desk's source instead of staying on Reading", async () => {
		const setSource = vi.fn();
		render(<NotebookIntake setSource={setSource} />);
		await act(async () => {
			fireEvent.change(screen.getByLabelText(/Choose a notebook/), { target: { files: [new File(["{}"], "a.ipynb")] } });
		});
		expect((await screen.findByRole("alert")).textContent).toMatch(/couldn't read this notebook/);
		expect(setSource).toHaveBeenLastCalledWith(null);
		expect(screen.getByRole("status").textContent).toBe("");
	});
});
