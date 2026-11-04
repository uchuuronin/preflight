/**
 * Home: a short scroll story. The plane crosses as you scroll and the three steps (probe,
 * simulate, verify) rise in, in the shapes that carry their meaning: a circle for what
 * repeats, a capsule for what you press, a ticket for what you keep.
 */
import { useRef, useState } from "react";
import { flag } from "../../lib/flags";
import { t } from "../../lib/i18n";
import { useAppPanels } from "../../registry";
import { DEFAULT_SAMPLE } from "../../workspace/record/load";
import { href } from "../route";
import { Plane } from "../Plane";
import { AddonBoundary } from "../../components/AddonBoundary";

const REPO_URL = "https://github.com/uchuuronin/preflight";

export function Home() {
	const [p, setP] = useState(0);
	const raf = useRef(0);
	const extraBeats = useAppPanels("home-beats");
	const reduce = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
	const progress = reduce ? 0.5 : p;
	return (
		<div
			className="pf-scroll"
			onScroll={(e) => {
				const el = e.currentTarget;
				const v = el.scrollTop / Math.max(1, el.scrollHeight - el.clientHeight);
				cancelAnimationFrame(raf.current);
				raf.current = requestAnimationFrame(() => setP(v));
			}}
		>
			<div className="pf-inner">
				<section className="pf-hero" style={reduce ? undefined : { transform: `translateY(${(-p * 140).toFixed(1)}px)`, opacity: Math.max(0.15, 1 - p * 1.6) }}>
					<span className="pf-hero__plane" style={{ left: `${(6 + p * 160).toFixed(1)}%` }}>
						<Plane size={46} />
					</span>
					<span className="pf-eyebrow">{t("home.eyebrow")}</span>
					<h1 className="pf-display">
						{t("home.title.a")}
						<br />
						<em>{t("home.title.em")}</em> {t("home.title.b")}
					</h1>
					<p className="pf-lede">{t("home.lede")}</p>
					<div className="pf-cta">
						<a className="pf-btn pf-btn--primary" href={href("new")}>
							{t("home.start")}
						</a>
						<a className="pf-btn" href={href("cockpit", DEFAULT_SAMPLE)}>
							{t("home.sample")}
						</a>
						{/* The Help page's one way in (Chiya, Nov 3): a quiet text link beside the two actions. */}
						{flag("c.help") ? (
							<a className="pf-link" href={href("help")}>
								{t("home.manual")}
							</a>
						) : null}
					</div>
					<span className="pf-scroll-hint">
						<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
							<path d="M7 2v10M3 8l4 4 4-4" />
						</svg>
						{t("home.scroll")}
					</span>
				</section>
				<section className="pf-beats" aria-label={t("home.how")}>
					<div className={`glass pf-beat pf-shape-circle${progress > 0.12 ? " is-in" : ""}`}>
						<span className="pf-beat__num">{t("home.step", { n: 1 })}</span>
						<h2>{t("home.probe.title")}</h2>
						<p>{t("home.probe.body")}</p>
					</div>
					<div className={`glass pf-beat pf-shape-capsule${progress > 0.26 ? " is-in" : ""}`}>
						<span className="pf-beat__num">{t("home.step", { n: 2 })}</span>
						<h2>{t("home.simulate.title")}</h2>
						<p>{t("home.simulate.body")}</p>
						{/* Named only when the live guide runs on Muse Spark in this build (audit finding 12). */}
						{flag("c.guide.live") ? <span className="pf-model">{t("home.simulate.model")}</span> : null}
					</div>
					<div className={`glass pf-beat pf-shape-ticket${progress > 0.4 ? " is-in" : ""}`}>
						<span className="pf-beat__num">{t("home.step", { n: 3 })}</span>
						<h2>{t("home.verify.title")}</h2>
						<p>{t("home.verify.body")}</p>
					</div>
					{extraBeats.map((b) => (
						<AddonBoundary key={b.id} id={b.id}>
							<b.component />
						</AddonBoundary>
					))}
				</section>
				<section className="pf-closing">
					<h2 className="pf-display pf-display--md">{t("home.closing")}</h2>
					<a className="pf-btn pf-btn--primary" href={href("new")}>
						{t("home.start")}
					</a>
				</section>
				{/* The rules ask that judges can tell what was built during the event (PM, Nov 3). */}
				<footer className="pf-home__foot">
					{t("home.built_during")} ·{" "}
					<a href={REPO_URL} target="_blank" rel="noreferrer">
						GitHub
					</a>
				</footer>
			</div>
		</div>
	);
}
