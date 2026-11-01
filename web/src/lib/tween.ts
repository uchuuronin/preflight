/**
 * One value clock: a number on screen travels to its new value on the same curve and duration as
 * the arc or ring drawn beside it (tokens.ts `ease` and `dur-value`), so a knob move reads as one
 * change instead of a number that snaps while its arc is still sweeping.
 *
 * The hook writes the element's text directly on each frame (no React render per frame: the
 * interaction budget is measured, lib/busy.ts). The element must hold its formatted value as its
 * only child, so React and the tween agree on the final text. A move that interrupts one in
 * flight carries on from the number on screen. Reduced motion: the number changes at once.
 */
import { useLayoutEffect, useRef, type RefObject } from "react";
import { duskTokens } from "../theme/tokens";

/** Source: tokens.ts `dur-value` (600ms), the dials' and memory ring's transition. */
export const VALUE_MS = Number.parseFloat(duskTokens["dur-value"]);

/** CSS cubic-bezier(x1, y1, x2, y2) as a function of progress 0..1 (Newton steps, then bisection). */
export function cubicBezier(x1: number, y1: number, x2: number, y2: number): (t: number) => number {
	const cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
	const cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
	const x = (s: number) => ((ax * s + bx) * s + cx) * s;
	const y = (s: number) => ((ay * s + by) * s + cy) * s;
	const dx = (s: number) => (3 * ax * s + 2 * bx) * s + cx;
	return (t: number) => {
		if (t <= 0) return 0;
		if (t >= 1) return 1;
		let s = t;
		for (let i = 0; i < 6; i++) {
			const e = x(s) - t;
			const d = dx(s);
			if (Math.abs(e) < 1e-6) return y(s);
			if (Math.abs(d) < 1e-6) break;
			s -= e / d;
		}
		let lo = 0, hi = 1;
		s = t;
		for (let i = 0; i < 30; i++) {
			const e = x(s) - t;
			if (Math.abs(e) < 1e-6) break;
			if (e > 0) hi = s;
			else lo = s;
			s = (lo + hi) / 2;
		}
		return y(s);
	};
}

/** Parse "cubic-bezier(a, b, c, d)" from the theme. */
function parseBezier(css: string): (t: number) => number {
	const m = css.match(/cubic-bezier\(([^)]+)\)/);
	const n = m ? m[1]!.split(",").map((v) => Number.parseFloat(v)) : [];
	return n.length === 4 && n.every(Number.isFinite) ? cubicBezier(n[0]!, n[1]!, n[2]!, n[3]!) : (t) => t;
}

/** Source: tokens.ts `ease`, the curve the dial arcs use. */
export const valueEase = parseBezier(duskTokens.ease);
/** Source: tokens.ts `ease-move` and the map's 0.8 s glide (styles.css .pf-mring, .react-flow__node-onode). */
export const moveEase = parseBezier(duskTokens["ease-move"]);
export const MOVE_MS = 800;

export function reducedMotion(): boolean {
	return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Tween the text of `ref` from the number it last showed to `value`, formatted by `format`.
 * `value` null: nothing to tween (an abstaining readout); the element shows React's text.
 */
export function useTweenedText(
	ref: RefObject<HTMLElement | null>,
	value: number | null,
	format: (v: number) => string,
	ms = VALUE_MS,
	ease: (t: number) => number = valueEase,
): void {
	const shown = useRef<number | null>(value);
	const frame = useRef(0);
	const fmtRef = useRef(format);
	fmtRef.current = format;
	const easeRef = useRef(ease);
	easeRef.current = ease;

	useLayoutEffect(() => {
		const el = ref.current;
		const from = shown.current;
		cancelAnimationFrame(frame.current);
		if (!el || value === null || from === null || from === value || !Number.isFinite(from) || reducedMotion() || ms <= 0) {
			shown.current = value;
			return;
		}
		const start = performance.now();
		const paint = (now: number) => {
			const p = Math.min(1, (now - start) / ms);
			const v = p >= 1 ? value : from + (value - from) * easeRef.current(p);
			shown.current = v;
			el.textContent = fmtRef.current(v);
			if (p < 1) frame.current = requestAnimationFrame(paint);
		};
		// First frame before paint, so the new value never flashes ahead of its arc.
		paint(start);
		frame.current = requestAnimationFrame(paint);
		return () => cancelAnimationFrame(frame.current);
	}, [ref, value, ms]);

	useLayoutEffect(() => () => cancelAnimationFrame(frame.current), []);
}
