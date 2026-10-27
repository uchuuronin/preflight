/**
 * D4 · Free-tier presets. Imported automatically by C's loadAddons().
 *
 * A band under the cockpit's whole run: for Google Colab, Kaggle and Lightning AI's free plan,
 * whether this plan finishes inside one session (or how many, with how often to save), or
 * won't fit the free GPU's memory. Behind flag d.freetier.
 */
import type { Lens } from "../../workspace/record/types";
import { registerPanel } from "../../registry";
import { whenReady } from "../_shared/whenReady";
import { FreeTierPanel } from "./FreeTierPanel";
import { registerFreeTierStrings } from "./strings";
import "./freetier.css";

export const FLAG = "d.freetier";

/** Jobs that run inside a GPU session; agent runs call APIs and serving never finishes. */
export const LENSES: Lens[] = ["finetune", "pretrain", "rl", "eval", "retrieval", "gnn", "vision"];

registerFreeTierStrings();

// See _shared/whenReady.ts: the registry may not be ready while add-ons load.
export const ready: Promise<void> = whenReady(() => {
	registerPanel({
		id: "freetier.sessions",
		slot: "below",
		titleKey: "addon.freetier.title",
		lenses: LENSES,
		order: 40,
		flag: FLAG,
		component: FreeTierPanel,
	});
});
