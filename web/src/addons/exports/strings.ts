/** English copy for the exports panel, registered into C's t() under addon.exports.*. */
import { registerStrings } from "../../lib/i18n";

export const EXPORTS_EN: Record<string, string> = {
	"addon.exports.title": "Take this plan with you",
	"addon.exports.lede": "Built from this record and the settings on screen, so they change as you move the knobs.",
	"addon.exports.illustrative": "This is a sample: its numbers are illustrative, and the files say so.",
	"addon.exports.copy": "Copy",
	"addon.exports.download": "Download",
	"addon.exports.copied": "Copied: {what}.",
	"addon.exports.copy_failed": "Couldn't copy. Select the text and copy it instead.",
	"addon.exports.download_failed": "This browser can't save the file. Open it, then copy the text instead.",

	"addon.exports.plan.name": "Plan",
	"addon.exports.plan.say": "Every setting, what you changed, and the predicted readouts with their ranges, as JSON.",

	"addon.exports.edits.name": "Edits to your script",
	"addon.exports.edits.none": "Nothing changed from the record's settings yet. Move a knob and its edit appears here.",
	"addon.exports.edits.count_one": "1 setting changed:",
	"addon.exports.edits.count_other": "{n} settings changed:",
	"addon.exports.edits.run": "Run it with sh preflight-edits.sh from your project's top folder. It checks every line first and changes nothing if one has moved since Preflight read it.",
	"addon.exports.edits.exact": "{n} made by the script",
	"addon.exports.edits.exact_one": "1 made by the script",
	"addon.exports.edits.by_hand": "{n} to make by hand",
	"addon.exports.edits.by_hand_one": "1 to make by hand",
	"addon.exports.edits.code": "{n} need a code change from Muse Code",
	"addon.exports.edits.code_one": "1 needs a code change from Muse Code",
	"addon.exports.edits.launch": "{n} about where it runs, in the batch file",
	"addon.exports.edits.launch_one": "1 about where it runs, in the batch file",
	"addon.exports.patches": "Muse Code patches for this record",
	"addon.exports.patch.passed": "Verified",
	"addon.exports.patch.failed": "Verify failed",
	"addon.exports.patch.error": "Verify didn't finish",
	"addon.exports.patch.pending": "Not verified yet",

	"addon.exports.sbatch.name": "Slurm batch file",
	"addon.exports.sbatch.say": "Asks for {gpus} and {time}: the slow end of the predicted range plus 15 minutes to set up.",
	"addon.exports.sbatch.say_api": "Asks for no GPU and {time}: the slow end of the predicted range plus 15 minutes to set up.",
	"addon.exports.sbatch.say_capped": "Asks for {gpus} and {time}, your time limit. The plan needs about {need}, so save checkpoints and resubmit to finish.",
	"addon.exports.sbatch.say_api_capped": "Asks for no GPU and {time}, your time limit. The plan needs about {need}, so save checkpoints and resubmit to finish.",
	"addon.exports.sbatch.say_api_untimed": "Asks for no GPU. Preflight has no time estimate at these settings, so set the time limit yourself.",
	"addon.exports.sbatch.say_untimed": "Asks for {gpus}. Preflight has no time estimate at these settings, so set the time limit yourself.",

	"addon.exports.command.name": "Command",
	"addon.exports.command.say": "Runs the job as the record found it, from your project's top folder. Make the edits first if you changed any settings.",

	"addon.exports.recheck.name": "Check it again on your machine",
	"addon.exports.recheck.say": "Installs Preflight's command-line tool and measures a few steps of the edited job on your machine, on its GPU if it has one, for a new flight record.",

	"addon.exports.link.name": "Link to this record",
	"addon.exports.link.say": "Opens this record at its own settings. Your changes travel in the plan file.",
};

export function registerExportsStrings(): void {
	registerStrings("en", EXPORTS_EN);
}
