/**
 * Exports (moved from C to D, Oct 31). Imported automatically by C's loadAddons().
 *
 * In C's "Take it with you" section (registerExportsPanel): the plan, the edits to the script, a
 * Slurm batch file, the command and the record's link, formatted from C's ExportPlan. Behind the
 * existing flag c.exports (off). The CLI item has its own flag, d.exports_cli (off until the
 * repo is public).
 */
import { registerExportsPanel } from "../../registry";
import { ExportsPanel } from "./ExportsPanel";
import { FLAG } from "./flags";
import { registerExportsStrings } from "./strings";
import "./exports.css";

export { CLI_FLAG, FLAG } from "./flags";

registerExportsStrings();

registerExportsPanel({
	id: "exports.plan",
	titleKey: "addon.exports.title",
	order: 90,
	flag: FLAG,
	component: ExportsPanel,
});
