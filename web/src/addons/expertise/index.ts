/**
 * D1 · Expertise dial. Imported automatically by C's loadAddons().
 *
 * Ships now, through C's existing extension points:
 *   - the detail level (new, practitioner, expert) in the Logbook's pilot card;
 *   - a glossary card in every knob's control, worded for that level.
 * Waiting on C's knob-view hook (docs/requests.md): fewer knobs and plainer labels at "new",
 * using tiers.ts.
 */
import { registerAppPanel, registerKnobDecoration } from "../../registry";
import { whenReady } from "../_shared/whenReady";
import { DetailLevel } from "./DetailLevel";
import { GlossaryCard } from "./GlossaryCard";
import { registerExpertiseStrings } from "./strings";
import "./expertise.css";

export const FLAG = "d.expertise";

registerExpertiseStrings();

// See _shared/whenReady.ts: the registry may not be ready while add-ons load.
export const ready: Promise<void> = whenReady(() => {
	registerAppPanel({
		id: "expertise.detail-level",
		slot: "logbook-settings",
		titleKey: "addon.expertise.dial.title",
		order: 10,
		flag: FLAG,
		component: DetailLevel,
	});
	registerKnobDecoration({
		id: "expertise.glossary",
		flag: FLAG,
		component: GlossaryCard,
	});
});
