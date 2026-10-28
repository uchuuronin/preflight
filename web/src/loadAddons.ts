/**
 * Loads every add-on (web/src/addons/<name>/index.ts or .tsx) after the registry exists.
 *
 * Each add-on is its own dynamic import, so one that throws while loading is logged and
 * skipped instead of blanking the page; the rest still register. The app renders once all
 * have settled or after LOAD_WAIT_MS, whichever is first; a slower add-on still registers
 * later and its panels appear (the registry is a live store).
 */
import "./registry";

export const LOAD_WAIT_MS = 1500;

export interface AddonFailure {
	path: string;
	error: unknown;
}

const modules = import.meta.glob(["./addons/*/index.ts", "./addons/*/index.tsx"]);

let failures: AddonFailure[] = [];

/** Add-ons that failed to load, for the status page and tests. */
export function addonFailures(): readonly AddonFailure[] {
	return failures;
}

export async function loadAddons(
	mods: Record<string, () => Promise<unknown>> = modules,
	waitMs = LOAD_WAIT_MS,
): Promise<readonly AddonFailure[]> {
	failures = [];
	const all = Promise.allSettled(
		Object.entries(mods).map(async ([path, load]) => {
			try {
				await load();
			} catch (error) {
				failures.push({ path, error });
				console.error(`[preflight] add-on ${path} failed to load and was skipped`, error);
			}
		}),
	);
	await Promise.race([all, new Promise((r) => setTimeout(r, waitMs))]);
	return failures;
}
