/**
 * First visit to a cockpit: three short notes, one at a time, each lighting the part of the
 * screen it's about (the map, the four readouts, the knobs and the guide). Shown once per
 * browser; "Skip" or Escape ends it. The seen marker is a per-browser convenience in
 * localStorage, read and written inside try/catch, so a private window just sees it again.
 * Behind `c.onboarding`.
 */
import { useEffect, useRef, useState } from "react";
import { flag } from "../lib/flags";
import { t } from "../lib/i18n";

const SEEN_KEY = "pf.onboarding.v1";

export const STEPS = [
	{ id: "map", target: ".pf-orbit:not(.pf-orbit--compact)" },
	{ id: "readouts", target: ".pf-dials" },
	{ id: "knobs", target: ".pf-dock__row" },
	{ id: "guide", target: ".pf-guide" },
] as const;

function seen(): boolean {
	try {
		return localStorage.getItem(SEEN_KEY) === "1";
	} catch {
		return false;
	}
}

function markSeen(): void {
	try {
		localStorage.setItem(SEEN_KEY, "1");
	} catch {
		/* storage blocked: the notes come back next visit, nothing else changes */
	}
}

export function Onboarding() {
	const [step, setStep] = useState<number | null>(() => (flag("c.onboarding") && !seen() ? 0 : null));
	const card = useRef<HTMLDivElement>(null);
	const lit = useRef<Element | null>(null);

	useEffect(() => {
		lit.current?.classList.remove("pf-onboard-lit");
		lit.current = null;
		if (step === null) return;
		const el = document.querySelector(STEPS[step]!.target);
		if (el) {
			el.classList.add("pf-onboard-lit");
			lit.current = el;
			// Above the card, below the top bar (scroll-margin-top in the CSS).
			const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
			el.scrollIntoView?.({ block: "start", behavior: reduce ? "auto" : "smooth" });
		}
		card.current?.focus();
		return () => {
			lit.current?.classList.remove("pf-onboard-lit");
		};
	}, [step]);

	if (step === null) return null;
	const end = () => {
		markSeen();
		setStep(null);
	};
	const last = step === STEPS.length - 1;
	const id = STEPS[step]!.id;
	return (
		<div
			ref={card}
			className="glass pf-onboard"
			role="dialog"
			aria-modal="false"
			aria-labelledby="pf-onboard-title"
			tabIndex={-1}
			onKeyDown={(e) => {
				if (e.key === "Escape") end();
			}}
		>
			<p className="pf-onboard__step">{t("onboard.step", { n: step + 1, of: STEPS.length })}</p>
			<b id="pf-onboard-title">{t(`onboard.${id}.title`)}</b>
			<p>{t(`onboard.${id}.body`)}</p>
			<div className="pf-guide__actions">
				<button type="button" className="pf-btn pf-btn--primary pf-btn--small" onClick={() => (last ? end() : setStep(step + 1))}>
					{last ? t("onboard.done") : t("onboard.next")}
				</button>
				{last ? null : (
					<button type="button" className="pf-btn pf-btn--small pf-btn--quiet" onClick={end}>
						{t("onboard.skip")}
					</button>
				)}
			</div>
		</div>
	);
}
