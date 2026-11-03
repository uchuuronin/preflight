import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { t } from "../../lib/i18n";
import { isFixture } from "../record/model";
import type { EstimateStatus } from "../record/types";
import { useOptionalWorkspace } from "../state";
import type { Fit } from "../status";

/** Icon shape plus a word: never colour alone. */
export function FitMark({ fit, label }: { fit: Fit; label?: string }) {
	return (
		<span className={`pf-status pf-status--${fit}`}>
			<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
				{fit === "fits" ? <circle cx="6" cy="6" r="4" /> : null}
				{fit === "tight" ? <path d="M6 1.5 L10.5 10 H1.5 Z" /> : null}
				{fit === "fails" ? <path d="M2.5 2.5 L9.5 9.5 M9.5 2.5 L2.5 9.5" /> : null}
				{fit === "unknown" ? <path d="M2.5 6 H9.5" /> : null}
			</svg>
			<span>{label ?? t(`fit.${fit}`)}</span>
		</span>
	);
}

/**
 * The word for where a number comes from. A sample (a fixture record) says once, in a banner over the
 * cockpit, that its numbers are examples (PM, Nov 3: one banner, not a word on every dial); each
 * number keeps its own status, but none says "Measured", since no real run measured it.
 */
export function provenanceKey(status: EstimateStatus, illustrative: boolean): string {
	return illustrative && status === "measured" ? "provenance.sample_probe" : `provenance.${status}`;
}

/** Where the open tip sits: fixed to the window, kept inside it, below the word or above near the bottom. */
type TipAt = { top: number; left: number; width: number; above: boolean };

const TIP_MAX = 230;
const EDGE = 8;

function placeTip(word: DOMRect): TipAt {
	const vw = window.innerWidth;
	const width = Math.min(TIP_MAX, Math.round(vw * 0.7));
	const left = Math.max(EDGE, Math.min(word.left, vw - width - EDGE));
	const above = word.bottom + 90 > window.innerHeight;
	return { top: above ? word.top - 6 : word.bottom + 6, left, width, above };
}

/**
 * Measured / predicted / extrapolated / not measured here, as a short word with its line style. The
 * word explains itself in one line on hover, on keyboard focus and on a tap (Chiya, Nov 2: four words
 * are a lot to learn, and a title attribute never shows on a phone). The tip is drawn in a layer of its
 * own, fixed to the window, so a scrolling side column or a phone's edge can't cut it off; Escape, a
 * second tap, a tap elsewhere or a scroll closes it. Screen readers get the same line as the word's
 * description, always.
 */
export function Provenance({ status }: { status: EstimateStatus }) {
	const ws = useOptionalWorkspace();
	const key = provenanceKey(status, ws ? isFixture(ws.record) : false);
	const desc = useId();
	const word = useRef<HTMLButtonElement>(null);
	const [at, setAt] = useState<TipAt | null>(null);
	// How it opened: a tap (or click) keeps it until a second tap, a tap elsewhere, Escape or blur;
	// a mouse hover closes when the pointer leaves. A phone's tap also sends pointer and focus events
	// first, so only a tap that found it already tapped open closes it.
	const how = useRef<"hover" | "focus" | "tap" | null>(null);
	const show = (by: "hover" | "focus" | "tap") => {
		how.current = by;
		if (word.current) setAt(placeTip(word.current.getBoundingClientRect()));
	};
	const hide = () => {
		how.current = null;
		setAt(null);
	};
	useEffect(() => {
		if (!at) return;
		const away = (e: PointerEvent) => {
			if (!word.current?.contains(e.target as Node)) hide();
		};
		document.addEventListener("pointerdown", away, true);
		window.addEventListener("scroll", hide, true);
		window.addEventListener("resize", hide);
		return () => {
			document.removeEventListener("pointerdown", away, true);
			window.removeEventListener("scroll", hide, true);
			window.removeEventListener("resize", hide);
		};
	}, [at]);
	return (
		<>
			<button
				ref={word}
				type="button"
				className={`pf-prov pf-prov--${status}`}
				aria-describedby={desc}
				onClick={(e) => {
					e.stopPropagation(); // a tap on the word inside a clickable area explains the word, nothing else
					if (at && how.current === "tap") hide();
					else show("tap");
				}}
				onPointerEnter={(e) => {
					if (e.pointerType === "mouse" && !at) show("hover");
				}}
				onPointerLeave={(e) => {
					if (e.pointerType === "mouse" && how.current === "hover") hide();
				}}
				onFocus={() => {
					if (!at) show("focus");
				}}
				onBlur={hide}
				onKeyDown={(e) => {
					if (e.key === "Escape" && at) {
						e.stopPropagation();
						hide();
					}
				}}
			>
				<span className="pf-prov__line" aria-hidden="true" />
				{t(key)}
			</button>
			<span id={desc} className="pf-visually-hidden">
				{t(`${key}.help`)}
			</span>
			{at && typeof document !== "undefined"
				? createPortal(
						<span
							className={`pf-prov__tip${at.above ? " is-above" : ""}`}
							aria-hidden="true"
							style={{ top: at.top, left: at.left, maxWidth: at.width }}
						>
							{t(`${key}.help`)}
						</span>,
						document.body,
					)
				: null}
		</>
	);
}
