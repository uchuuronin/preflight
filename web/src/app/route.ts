/**
 * Hash routes, so the static site works on any host without rewrites:
 *   #/               home
 *   #/new            new flight (check-in and samples)
 *   #/boarding/<id>  probes running for job <id>
 *   #/cockpit/<ref>  a record by sample id or share token
 *   #/logbook        flights this browser opened
 *   #/checks         held-out checks of the engine on real runs (documentation, not in the nav)
 *   #/help           how to fill in a check-in: every input and knob (linked from Home only)
 */
import { useSyncExternalStore } from "react";

export type Screen = "home" | "new" | "boarding" | "cockpit" | "logbook" | "checks" | "help";

export interface Route {
	screen: Screen;
	param: string | null;
}

export const SCREEN_ORDER: Screen[] = ["home", "new", "boarding", "cockpit", "logbook", "checks", "help"];

export function parseRoute(hash: string): Route {
	const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
	const [head, param] = parts;
	switch (head) {
		case "new":
			return { screen: "new", param: null };
		case "boarding":
			return { screen: "boarding", param: param ?? null };
		case "cockpit":
			return { screen: "cockpit", param: param ? decodeURIComponent(param) : null };
		case "logbook":
			return { screen: "logbook", param: null };
		case "checks":
			return { screen: "checks", param: null };
		case "help":
			return { screen: "help", param: null };
		default:
			return { screen: "home", param: null };
	}
}

export function href(screen: Screen, param?: string | null): string {
	if (screen === "home") return "#/";
	return param ? `#/${screen}/${encodeURIComponent(param)}` : `#/${screen}`;
}

export function go(screen: Screen, param?: string | null): void {
	window.location.hash = href(screen, param);
}

function subscribe(fn: () => void): () => void {
	window.addEventListener("hashchange", fn);
	return () => window.removeEventListener("hashchange", fn);
}

export function useRoute(): Route {
	const hash = useSyncExternalStore(subscribe, () => window.location.hash, () => "");
	return parseRoute(hash);
}
