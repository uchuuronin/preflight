import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { NotebookIntake } from "./NotebookIntake";
import colab from "./fixtures/colab_lora.ipynb?raw";
import markdownOnly from "./fixtures/markdown_only.ipynb?raw";
import { registerNotebookStrings } from "./strings";

function choose(file: File) {
	const input = screen.getByLabelText(/Choose a notebook/);
	return act(async () => {
		fireEvent.change(input, { target: { files: [file] } });
	});
}

const nb = (name: string, text: string) => new File([text], name, { type: "application/x-ipynb+json" });

describe("NotebookIntake", () => {
	beforeAll(registerNotebookStrings);
	afterEach(cleanup);

	it("starts with a labelled file chooser that takes .ipynb files", () => {
		render(<NotebookIntake setSource={() => {}} />);
		const input = screen.getByLabelText(/Choose a notebook/) as HTMLInputElement;
		expect(input.type).toBe("file");
		expect(input.accept).toBe(".ipynb");
		expect(screen.getByText(/Only the code is sent/)).toBeTruthy();
	});

	it("previews what will run and hands the desk a notebook source", async () => {
		const setSource = vi.fn();
		render(<NotebookIntake setSource={setSource} />);
		await choose(nb("colab_lora.ipynb", colab));
		await waitFor(() => expect(screen.getByText("colab_lora.ipynb runs as colab_lora.py")).toBeTruthy());
		expect(screen.getByText(/3 code cells/)).toBeTruthy();
		expect(screen.getByText(/2 other cells skipped/)).toBeTruthy();
		expect(screen.getByText(/4 shell or magic lines become comments/)).toBeTruthy();
		expect(screen.getByText(/4 packages to install/)).toBeTruthy();
		expect(screen.getByText(/saved outputs left out/)).toBeTruthy();
		expect(screen.getByLabelText("colab_lora.py").textContent).toContain("import torch");
		expect(screen.getByLabelText("requirements.txt").textContent).toContain("peft==0.13.2");
		expect(screen.getByText("requirements.txt", { selector: "figcaption" })).toBeTruthy();
		const source = setSource.mock.calls.at(-1)?.[0];
		expect(source.kind).toBe("notebook");
		expect(Object.keys(source.files)).toEqual(["colab_lora.ipynb"]);
	});

	it("says when it cleaned the file name so the notebook can run", async () => {
		render(<NotebookIntake setSource={() => {}} />);
		await choose(nb("train (1).ipynb", colab));
		await waitFor(() => expect(screen.getByText("train_1.ipynb runs as train_1.py")).toBeTruthy());
		expect(screen.getByText(/Renamed from train \(1\)\.ipynb/)).toBeTruthy();
	});

	it("explains a file that isn't a notebook, and clears the desk's source", async () => {
		const setSource = vi.fn();
		render(<NotebookIntake setSource={setSource} />);
		await choose(nb("broken.ipynb", "{ nope"));
		const alert = await screen.findByRole("alert");
		expect(alert.textContent).toMatch(/can't be read as a notebook/);
		expect(alert.textContent).toMatch(/Download it again/);
		expect(setSource).toHaveBeenLastCalledWith(null);
	});

	it("stops a notebook with no code before it costs a probe", async () => {
		render(<NotebookIntake setSource={() => {}} />);
		await choose(nb("notes.ipynb", markdownOnly));
		expect((await screen.findByRole("alert")).textContent).toMatch(/no code cells/);
	});

	it("says the check-in limit when the code is too big", async () => {
		render(<NotebookIntake setSource={() => {}} />);
		const code = "x = 1\n".repeat(1_000_000);
		await choose(nb("big.ipynb", JSON.stringify({ cells: [{ cell_type: "code", source: code }] })));
		expect((await screen.findByRole("alert")).textContent).toMatch(/over 5 MB, more than a check-in carries/);
	});

	it("turns away a file that isn't .ipynb", async () => {
		render(<NotebookIntake setSource={() => {}} />);
		await choose(new File(["print(1)"], "train.py"));
		expect((await screen.findByRole("alert")).textContent).toMatch(/\.ipynb/);
	});

	it("takes a notebook dropped on it", async () => {
		const setSource = vi.fn();
		render(<NotebookIntake setSource={setSource} />);
		const zone = screen.getByTestId("notebook-drop");
		await act(async () => {
			fireEvent.drop(zone, { dataTransfer: { files: [nb("colab_lora.ipynb", colab)] } });
		});
		await waitFor(() => expect(setSource).toHaveBeenCalledWith(expect.objectContaining({ kind: "notebook" })));
	});

	it("announces progress to screen readers", async () => {
		render(<NotebookIntake setSource={() => {}} />);
		const status = screen.getByRole("status");
		await choose(nb("colab_lora.ipynb", colab));
		await waitFor(() => expect(status.textContent).toMatch(/runs as colab_lora\.py/));
	});

	it("keeps the newest file when an older one finishes reading later", async () => {
		const setSource = vi.fn();
		render(<NotebookIntake setSource={setSource} />);
		let finishOld: (text: string) => void = () => {};
		const old = nb("old.ipynb", "");
		old.text = () => new Promise<string>((resolve) => (finishOld = resolve));
		await choose(old);
		await choose(nb("new.ipynb", colab));
		await waitFor(() => expect(screen.getByText("new.ipynb runs as new.py")).toBeTruthy());
		await act(async () => finishOld(colab));
		expect(screen.getByText("new.ipynb runs as new.py")).toBeTruthy();
		expect(Object.keys(setSource.mock.calls.at(-1)?.[0].files)).toEqual(["new.ipynb"]);
	});

	it("clears the desk's source while a new file is read", async () => {
		const setSource = vi.fn();
		render(<NotebookIntake setSource={setSource} />);
		const slow = nb("slow.ipynb", "");
		slow.text = () => new Promise<string>(() => {});
		await choose(slow);
		expect(setSource).toHaveBeenLastCalledWith(null);
		expect(screen.getByRole("status").textContent).toMatch(/Reading slow\.ipynb/);
	});

	it("doesn't hand the desk a source after the tab is closed", async () => {
		const setSource = vi.fn();
		const { unmount } = render(<NotebookIntake setSource={setSource} />);
		let finish: (text: string) => void = () => {};
		const file = nb("late.ipynb", "");
		file.text = () => new Promise<string>((resolve) => (finish = resolve));
		await choose(file);
		unmount();
		await act(async () => finish(colab));
		expect(setSource.mock.calls.every(([s]) => s === null)).toBe(true);
	});

	it("says check-in from a notebook isn't open yet when the desk can't take it", async () => {
		render(<NotebookIntake />);
		await choose(nb("colab_lora.ipynb", colab));
		await waitFor(() => expect(screen.getByText(/isn't open yet/)).toBeTruthy());
		expect(screen.getByText(/isn't open yet/).textContent).toMatch(/\.py/);
		expect(screen.getByText(/isn't open yet/).textContent).toMatch(/Repo link/);
	});
});
