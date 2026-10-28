/** English copy for notebook intake (D5), registered into C's t() under addon.notebook.*. */
import { registerStrings } from "../../lib/i18n";

export const NOTEBOOK_EN: Record<string, string> = {
	"addon.notebook.tab": "Notebook",
	"addon.notebook.choose": "Choose a notebook (.ipynb)",
	"addon.notebook.choose_again": "Choose another notebook",
	"addon.notebook.drop": "or drop it here",
	"addon.notebook.privacy": "Only the code is sent. Saved outputs, images and widget state stay on your computer.",
	"addon.notebook.reading": "Reading {name}",
	"addon.notebook.runs_as": "{name} runs as {script}",
	"addon.notebook.renamed": "Renamed from {name} so it can run.",
	"addon.notebook.code_cells.one": "1 code cell",
	"addon.notebook.code_cells.other": "{n} code cells",
	"addon.notebook.other_cells.one": "1 other cell skipped",
	"addon.notebook.other_cells.other": "{n} other cells skipped",
	"addon.notebook.magic.one": "1 shell or magic line becomes a comment",
	"addon.notebook.magic.other": "{n} shell or magic lines become comments",
	"addon.notebook.packages.one": "1 package to install",
	"addon.notebook.packages.other": "{n} packages to install",
	"addon.notebook.removed": "{size} of saved outputs left out",
	"addon.notebook.what_runs": "What will run",
	"addon.notebook.more_lines": "and {n} more lines",
	"addon.notebook.not_open": "Check-in from a notebook isn't open yet. You can still see what will run here. To fly it today, download it as a Python file (.py), put that on GitHub and paste its link under Repo link.",
	"addon.notebook.error.not_ipynb": "That file isn't a notebook. Choose a file that ends in .ipynb.",
	"addon.notebook.error.not_json": "This file can't be read as a notebook. Download it again from Colab or Jupyter (File, then Download .ipynb) and choose the new copy.",
	"addon.notebook.error.no_cells": "This file can't be read as a notebook: it has no cells. Download it again from Colab or Jupyter and choose the new copy.",
	"addon.notebook.error.cell_not_text": "One of the cells can't be read as text. Open the notebook in Jupyter or Colab, save it, and choose it again.",
	"addon.notebook.error.no_code": "This notebook has no code cells, so there's nothing to run. Add your training code and choose it again.",
	"addon.notebook.error.too_big": "The code in this notebook is over {limit}, more than a check-in carries. Load data from Hugging Face instead of pasting it into cells.",
	"addon.notebook.error.too_big_file": "This file is over {size}, mostly saved outputs. In Jupyter or Colab, clear all outputs, save, and choose it again.",
	"addon.notebook.error.read_failed": "Your browser couldn't open that file. Try choosing it again.",
	"addon.notebook.error.bad_text": "This notebook has characters that aren't valid text. Open it in Jupyter or Colab, save it, and choose it again.",
	"addon.notebook.error.unexpected": "Preflight couldn't read this notebook. Try saving it again in Jupyter or Colab and choosing it again.",
};

export function registerNotebookStrings(): void {
	registerStrings("en", NOTEBOOK_EN);
}
