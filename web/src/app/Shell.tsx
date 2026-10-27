/**
 * The app frame: one tall sky that climbs as you move through the screens, a glass nav
 * capsule, and five stacked screens that rise and un-blur into place (D's design). Only the
 * active screen is interactive; the others are hidden from assistive tech.
 */
import { useEffect, useState } from "react";
import { t } from "../lib/i18n";
import { useAppPanels } from "../registry";
import { Plane } from "./Plane";
import { applyTokenSet, currentTokenSet, DEFAULT_SET } from "../theme/tokens";
import { Boarding } from "./screens/Boarding";
import { CockpitScreen } from "./screens/CockpitScreen";
import { Home } from "./screens/Home";
import { Logbook } from "./screens/Logbook";
import { NewFlight } from "./screens/NewFlight";
import { go, href, SCREEN_ORDER, type Screen, useRoute } from "./route";

const SKY_Y: Record<Screen, number> = { home: -47, new: -38, boarding: -26, cockpit: -12, logbook: 0 };
const THEME_KEY = "pf.theme";

function savedTheme(): string {
	try {
		return localStorage.getItem(THEME_KEY) ?? DEFAULT_SET;
	} catch {
		return DEFAULT_SET;
	}
}

const TABS: Array<{ screen: Screen; key: string }> = [
	{ screen: "home", key: "nav.home" },
	{ screen: "new", key: "nav.new" },
	{ screen: "cockpit", key: "nav.cockpit" },
	{ screen: "logbook", key: "nav.logbook" },
];

export function Shell() {
	const route = useRoute();
	const [theme, setTheme] = useState(() => applyTokenSet(savedTheme()));
	const navEnd = useAppPanels("nav-end");
	// The cockpit keeps the last record it showed, so leaving and coming back is instant.
	const [cockpitRef, setCockpitRef] = useState<string | null>(route.screen === "cockpit" ? route.param : null);
	useEffect(() => {
		if (route.screen === "cockpit" && route.param && route.param !== cockpitRef) setCockpitRef(route.param);
	}, [route, cockpitRef]);

	const current = SCREEN_ORDER.indexOf(route.screen);
	const pos = (s: Screen) => {
		const i = SCREEN_ORDER.indexOf(s);
		return i === current ? "active" : i < current ? "before" : "after";
	};
	const toggleSky = () => {
		const next = currentTokenSet() === "dusk" ? "day" : "dusk";
		setTheme(applyTokenSet(next));
		try {
			localStorage.setItem(THEME_KEY, next);
		} catch {
			// theme simply isn't remembered
		}
	};

	useEffect(() => {
		document.title = t(`title.${route.screen}`);
	}, [route.screen]);

	return (
		<div className="pf">
			<div className="pf-sky" style={{ transform: `translateY(${SKY_Y[route.screen]}%)` }} aria-hidden="true">
				<div className="pf-cloud pf-cloud--1" />
				<div className="pf-cloud pf-cloud--2" />
				<div className="pf-cloud pf-cloud--3" />
				<div className="pf-cloud pf-cloud--4" />
				<div className="pf-cloud pf-cloud--5" />
				<div className="pf-trail" />
			</div>

			<nav className="glass pf-nav" aria-label={t("nav.label")}>
				<a className="pf-nav__brand" href={href("home")} aria-label={t("nav.brand_aria")}>
					<Plane />
					{t("app.name")}
				</a>
				<span className="pf-nav__sep" aria-hidden="true" />
				{TABS.map((tab) => {
					const on = route.screen === tab.screen || (tab.screen === "new" && route.screen === "boarding");
					const target = tab.screen === "cockpit" ? href("cockpit", cockpitRef ?? "lora-7b") : href(tab.screen);
					return (
						<a key={tab.screen} className="pf-nav__tab" href={target} aria-current={on ? "page" : undefined}>
							{t(tab.key)}
						</a>
					);
				})}
				<span className="pf-nav__sep" aria-hidden="true" />
				{navEnd.map((p) => (
					<p.component key={p.id} />
				))}
				<button type="button" className="pf-nav__sky" onClick={toggleSky} aria-label={theme === "dusk" ? t("nav.to_day") : t("nav.to_dusk")}>
					{theme === "dusk" ? (
						<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true">
							<path d="M20 14.5A8 8 0 019.5 4a8 8 0 1010.5 10.5z" />
						</svg>
					) : (
						<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true">
							<circle cx="12" cy="12" r="4.2" />
							<path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6" />
						</svg>
					)}
				</button>
			</nav>

			<main className="pf-screen" data-pos={pos("home")} aria-hidden={pos("home") !== "active"} inert={pos("home") !== "active"}>
				<Home />
			</main>
			<main className="pf-screen" data-pos={pos("new")} aria-hidden={pos("new") !== "active"} inert={pos("new") !== "active"}>
				<NewFlight onSample={(id) => go("cockpit", id)} />
			</main>
			<main className="pf-screen" data-pos={pos("boarding")} aria-hidden={pos("boarding") !== "active"} inert={pos("boarding") !== "active"}>
				<Boarding jobId={route.screen === "boarding" ? route.param : null} />
			</main>
			<main className="pf-screen" data-pos={pos("cockpit")} aria-hidden={pos("cockpit") !== "active"} inert={pos("cockpit") !== "active"}>
				{cockpitRef || route.screen === "cockpit" ? <CockpitScreen recordRef={cockpitRef ?? route.param ?? "lora-7b"} /> : null}
			</main>
			<main className="pf-screen" data-pos={pos("logbook")} aria-hidden={pos("logbook") !== "active"} inert={pos("logbook") !== "active"}>
				<Logbook active={route.screen === "logbook"} />
			</main>
		</div>
	);
}
