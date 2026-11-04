/** English copy for the free-tier add-on (D4), registered into C's t() under addon.freetier.*. */
import { registerStrings } from "../../lib/i18n";

export const FREETIER_EN: Record<string, string> = {
	"addon.freetier.title": "Free GPUs",
	"addon.freetier.lede": "Whether this plan finishes inside one free session, judged on the slow end of the time range plus 15 minutes to set up.",
	"addon.freetier.name.colab": "Google Colab (free)",
	"addon.freetier.name.kaggle": "Kaggle Notebooks",
	"addon.freetier.name.modal": "Modal (Starter credit)",
	"addon.freetier.name.lightning": "Lightning AI (free plan)",
	"addon.freetier.cap": "{h} h sessions",
	"addon.freetier.or": " or ",
	"addon.freetier.chip.one_session": "One session",
	"addon.freetier.chip.sessions": "{n} sessions",
	"addon.freetier.chip.no_memory": "Won't fit",
	"addon.freetier.chip.other_gpu": "Measure on {gpus}",
	"addon.freetier.chip.abstain": "Not measured here",
	"addon.freetier.say.one_session": "Finishes in one session, with time to set up.",
	"addon.freetier.say.sessions": "Runs past one session. Save a checkpoint at least every {every} and resume from it.",
	"addon.freetier.say.no_memory": "Needs up to {need} at its peak (the top of its {pct}% range; the estimate is {point}, {status}); a {gpu} has {have}. Lower the batch or sequence length, or turn on checkpointing.",
	"addon.freetier.say.other_gpu": "Memory fits. Time was predicted for {current}, so run a probe on a {gpus} to check the session length.",
	"addon.freetier.say.abstain": "Memory or time isn't predicted at these settings yet.",
	"addon.freetier.note.colab": "Colab doesn't publish its GPU or limits; users report a T4, and sessions can end sooner when demand is high or the tab sits idle.",
	"addon.freetier.note.kaggle": "Your weekly GPU quota shows in Kaggle's settings; users report about 30 hours.",
	"addon.freetier.note.modal": "Up to 24 hours per run. $30 of free compute a month, about 37 GPU-hours on an L4 at Modal's list rate.",
	"addon.freetier.over_credit": "This run would use more than one month's {credit} free credit.",
	"addon.freetier.fp16": "No bf16 on T4 or P100: train in fp16 with loss scaling.",
	"addon.freetier.fp16_switch": "This plan uses bf16, which a {gpu} can't run. Switch precision to fp16 with loss scaling.",
	"addon.freetier.note.lightning": "One free Studio, restarted every 4 hours. GPUs draw on a one-time grant of up to 30 credits.",
	"addon.freetier.sources": "Sources",
	"addon.freetier.read": "read {date}",
};

export function registerFreeTierStrings(): void {
	registerStrings("en", FREETIER_EN);
}
