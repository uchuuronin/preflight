/**
 * Browser timings the contract's service levels are measured by. Knob latency is the
 * time from a knob change to recomputed readouts; the nightly Playwright test reads
 * window.__preflight.knobLatencyP95() and the session sends a summary with its events.
 */

const samples: number[] = [];
const MAX = 500;

export function recordKnobLatency(ms: number): void {
	samples.push(ms);
	if (samples.length > MAX) samples.shift();
}

export function knobLatencyP95(): number | null {
	if (!samples.length) return null;
	const sorted = [...samples].sort((a, b) => a - b);
	return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ?? null;
}

declare global {
	interface Window {
		__preflight?: { knobLatencyP95: () => number | null; samples: () => number[] };
	}
}

if (typeof window !== "undefined") {
	window.__preflight = { knobLatencyP95, samples: () => [...samples] };
}
