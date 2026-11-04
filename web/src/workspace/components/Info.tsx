/**
 * The cockpit's one place for explanation: an "i" that shows its text on hover, on keyboard
 * focus, or on a tap (touch screens have no hover). The cockpit itself shows labels, numbers,
 * intervals and status words only.
 */
import { type ReactNode, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { t } from "../../lib/i18n";

export function Info({ label, children, align = "center" }: { label: string; children: ReactNode; align?: "center" | "start" | "end" }) {
	const id = useId();
	const [open, setOpen] = useState(false);
	const ref = useRef<HTMLSpanElement>(null);
	useEffect(() => {
		if (!open) return;
		const close = (e: Event) => {
			if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
		};
		document.addEventListener("pointerdown", close);
		document.addEventListener("keydown", close);
		return () => {
			document.removeEventListener("pointerdown", close);
			document.removeEventListener("keydown", close);
		};
	}, [open]);
	// Opened, the text stays on the screen: shifted (CSS translate, on top of its alignment's transform) back inside a 12 px margin where its alignment would
	// run it off an edge (PM's audit, Nov 3: the Map and Fit "i" were cut off at 390 px).
	const pop = useRef<HTMLSpanElement>(null);
	const [dx, setDx] = useState(0);
	useLayoutEffect(() => {
		if (!open) {
			setDx(0);
			return;
		}
		const fit = () => {
			const el = pop.current;
			if (!el) return;
			el.style.translate = "0px 0px";
			const r = el.getBoundingClientRect();
			const vw = document.documentElement.clientWidth;
			const shift = r.left < 12 ? 12 - r.left : r.right > vw - 12 ? Math.max(12 - r.left, vw - 12 - r.right) : 0;
			el.style.translate = "";
			setDx(Math.round(shift));
		};
		fit();
		window.addEventListener("resize", fit);
		return () => window.removeEventListener("resize", fit);
	}, [open]);
	return (
		<span ref={ref} className={`pf-info pf-info--${align}${open ? " is-open" : ""}`}>
			<button type="button" className="pf-info__btn" aria-label={t("info.about", { what: label })} aria-describedby={id} aria-expanded={open} onClick={() => setOpen(!open)}>
				i
			</button>
			<span id={id} ref={pop} role="tooltip" className="glass pf-info__pop" style={dx ? { translate: `${dx}px 0px` } : undefined}>
				{children}
			</span>
		</span>
	);
}
