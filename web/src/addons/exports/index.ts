/**
 * Exports (moved from C to D, Oct 31). Imported automatically by C's loadAddons().
 *
 * A band under the cockpit's whole run: the plan, the edits to the script, a Slurm batch file,
 * the command and the record's link, from the record and the settings on screen. Behind the
 * existing flag c.exports (off). The CLI item has its own flag, d.exports_cli (off until the
 * repo is public).
 */
import { registerPanel } from "../../registry";
import { ExportsPanel } from "./ExportsPanel";
import { FLAG } from "./flags";
import { registerExportsStrings } from "./strings";
import "./exports.css";

export { CLI_FLAG, FLAG } from "./flags";

registerExportsStrings();

registerPanel({
	id: "exports.plan",
	slot: "below",
	titleKey: "addon.exports.title",
	order: 90,
	flag: FLAG,
	component: ExportsPanel,
});
