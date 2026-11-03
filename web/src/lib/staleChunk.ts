/**
 * After a deploy, a page loaded before it asks for code chunks that no longer exist ("Failed to fetch
 * dynamically imported module", seen on the live site, Nov 3, opening a sample): reload once to get the
 * new build. A per-tab marker stops a loop when the chunk is missing for another reason; it expires so
 * a later deploy in the same tab still reloads.
 */
const KEY = "pf.reloadedForChunk";
/** Source: judgment: a second miss within a minute isn't a stale page. */
const WINDOW_MS = 60_000;

export function reloadOnceForStaleChunk(now = Date.now(), store: Pick<Storage, "getItem" | "setItem"> | null = safeSession(), reload = () => location.reload()): boolean {
	try {
		const last = Number(store?.getItem(KEY) ?? 0);
		if (last && now - last < WINDOW_MS) return false;
		store?.setItem(KEY, String(now));
	} catch {
		return false; // no storage: don't risk a reload loop
	}
	reload();
	return true;
}

function safeSession(): Storage | null {
	try {
		return sessionStorage;
	} catch {
		return null;
	}
}

export function installStaleChunkReload(): void {
	window.addEventListener("vite:preloadError", (e) => {
		if (reloadOnceForStaleChunk()) e.preventDefault();
	});
}

/** For a caught import error: is it a missing chunk? */
export function isStaleChunkError(e: unknown): boolean {
	const m = e instanceof Error ? e.message : String(e);
	return /Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module/i.test(m);
}
