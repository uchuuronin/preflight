/**
 * D5 · Notebook intake. Imported automatically by C's loadAddons().
 *
 * A "Notebook" tab on the check-in desk: choose or drop a .ipynb, see exactly what will run, and
 * check it in as a `notebook` source that A converts. Behind flag d.notebook.
 */
import { registerAppPanel } from "../../registry";
import { NotebookIntake } from "./NotebookIntake";
import { registerNotebookStrings } from "./strings";
import "./notebook.css";

export const FLAG = "d.notebook";

registerNotebookStrings();

registerAppPanel({
	id: "notebook.intake",
	slot: "intake",
	titleKey: "addon.notebook.tab",
	order: 10,
	flag: FLAG,
	component: NotebookIntake,
});
