/**
 * While the visitor is pressing keys or dragging, the map's decorative loops (the flowing dots,
 * the drifting clouds, the pulsing stage dot) pause, so every frame goes to the knob's answer.
 * Sets `data-busy` on <html> for BUSY_MS after the last pointer or key press; styles.css
 * pauses the loops under it. They resume where they stopped, so nothing jumps.
 *
 * Measured Oct 27 (production build, headless Chromium, 4x CPU slowdown, 24 knob steps x 3
 * runs): slowest interaction p98 1,030 ms without the pause, 570 ms with it.
 */

/**
 * Source: guess, longer than the gap between steps of a held arrow key (about 33 ms) or a
 * slider drag's events, short enough that the dots restart as soon as the hand stops.
 * Revisited with the stranger test's interaction log.
 */
export const BUSY_MS = 700;

export function installBusyFlag(root: HTMLElement = document.documentElement): () => void {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const on = (): void => {
		root.dataset.busy = "";
		clearTimeout(timer);
		timer = setTimeout(() => {
			delete root.dataset.busy;
		}, BUSY_MS);
	};
	const opts = { capture: true, passive: true } as const;
	for (const type of ["pointerdown", "keydown", "input"] as const) window.addEventListener(type, on, opts);
	return () => {
		clearTimeout(timer);
		delete root.dataset.busy;
		for (const type of ["pointerdown", "keydown", "input"] as const) window.removeEventListener(type, on, opts);
	};
}
