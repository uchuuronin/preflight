/**
 * The cockpit's one place for explanation: an "i" that shows its text on hover, on keyboard
 * focus, or on a tap (touch screens have no hover). The cockpit itself shows labels, numbers,
 * intervals and status words only.
 */
import { type ReactNode, useEffect, useId, useRef, useState } from "react";
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
	return (
		<span ref={ref} className={`pf-info pf-info--${align}${open ? " is-open" : ""}`}>
			<button type="button" className="pf-info__btn" aria-label={t("info.about", { what: label })} aria-describedby={id} aria-expanded={open} onClick={() => setOpen(!open)}>
				i
			</button>
			<span id={id} role="tooltip" className="glass pf-info__pop">
				{children}
			</span>
		</span>
	);
}
