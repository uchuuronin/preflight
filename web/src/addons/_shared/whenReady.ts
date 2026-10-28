/**
 * Run an add-on's registration once C's registry can take it.
 *
 * registry.ts loads add-ons with an eager import.meta.glob, which Vite hoists above the
 * registry's own state, so an add-on that registers while it is being loaded reaches the
 * registry before its `state` exists (a ReferenceError from the temporal dead zone). This
 * retries on the next tick until the registry is ready, and only for that error.
 * Remove once C moves the glob out of registry.ts (docs/requests.md, D -> C, Oct 27).
 */
export function whenReady(fn: () => void, maxRetries = 100): Promise<void> {
	return new Promise((resolve, reject) => {
		let retries = 0;
		const attempt = () => {
			try {
				fn();
				resolve();
			} catch (err) {
				if (err instanceof ReferenceError && retries < maxRetries) {
					retries++;
					setTimeout(attempt, 0);
				} else {
					reject(err);
				}
			}
		};
		attempt();
	});
}
