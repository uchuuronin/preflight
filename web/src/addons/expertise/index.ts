/**
 * D1 · Expertise dial. Imported by C's loadAddons(), after the registry exists.
 *
 * - The detail level (New to this, Practitioner, Expert) in the cockpit header.
 * - A view over the knob dock: at "New to this", fewer knobs with plainer names (knobView.ts).
 * - A glossary card in every knob's control, worded for the level.
 * - At "New to this", the readouts restated in one plain line above the status line (PlainLine).
 * All behind flag d.expertise.
 */
import { registerKnobDecoration, registerKnobView, registerPanel } from "../../registry";
import { DetailLevel } from "./DetailLevel";
import { GlossaryCard } from "./GlossaryCard";
import { PlainLine } from "./PlainLine";
import { createKnobView } from "./knobView";
import { subscribeTier } from "./store";
import { registerExpertiseStrings } from "./strings";
import "./expertise.css";

export const FLAG = "d.expertise";

registerExpertiseStrings();

registerPanel({
	id: "expertise.detail-level",
	slot: "cockpit-head",
	titleKey: "addon.expertise.dial.title",
	order: 10,
	flag: FLAG,
	component: DetailLevel,
});

registerPanel({
	id: "expertise.plain-line",
	slot: "cockpit-head",
	titleKey: "addon.expertise.plain.label",
	order: 20,
	flag: FLAG,
	component: PlainLine,
});

registerKnobDecoration({
	id: "expertise.glossary",
	flag: FLAG,
	component: GlossaryCard,
});

const view = { ...createKnobView(), flag: FLAG };
registerKnobView(view);

// The dock redraws on registry changes; re-registering the same view tells it the level moved.
subscribeTier(() => registerKnobView({ ...view }));
