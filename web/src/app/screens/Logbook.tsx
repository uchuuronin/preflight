/**
 * Logbook: flights this browser opened, as boarding passes, with what each one caught.
 * Settings that belong to D's add-ons (expertise, currency) dock in the pilot card through
 * the registry's "logbook-settings" slot.
 */
import { useEffect, useState } from "react";
import { t } from "../../lib/i18n";
import { useAppPanels } from "../../registry";
import { clearLogbook, type LogEntry, readLogbook } from "../logbook";
import { Plane } from "../Plane";
import { href } from "../route";
import { AddonBoundary } from "../../components/AddonBoundary";
import { WatchRun } from "../../workspace/live/WatchRun";

const WORD: Record<NonNullable<LogEntry["worst"]>, { cls: string; key: string }> = {
	critical: { cls: "fails", key: "logbook.status.needs_fix" },
	error: { cls: "fails", key: "logbook.status.needs_fix" },
	warn: { cls: "tight", key: "logbook.status.check" },
	info: { cls: "fits", key: "logbook.status.clear" },
};

export function Logbook({ active }: { active: boolean }) {
	const settings = useAppPanels("logbook-settings");
	const [entries, setEntries] = useState<LogEntry[]>([]);
	useEffect(() => {
		if (active) setEntries(readLogbook());
	}, [active]);
	const caught = entries.filter((e) => e.worst === "error" || e.worst === "critical").length;
	const lenses = new Set(entries.map((e) => e.lens)).size;
	return (
		<div className="pf-scroll">
			<div className="pf-inner pf-log">
				<section className="glass pf-pilot" aria-label={t("logbook.profile")}>
					<div className="pf-avatar" aria-hidden="true">
						<Plane size={34} />
					</div>
					<div className="pf-pilot__name">
						<h1>{t("logbook.you")}</h1>
						<p>{t("logbook.local")}</p>
					</div>
					<div className="pf-stats">
						<div className="pf-stat">
							<b>{entries.length}</b>
							<small>{t("logbook.stat.flights")}</small>
						</div>
						<div className="pf-stat">
							<b>{caught}</b>
							<small>{t("logbook.stat.caught")}</small>
						</div>
						<div className="pf-stat">
							<b>{lenses}</b>
							<small>{t("logbook.stat.lenses")}</small>
						</div>
					</div>
					{settings.map((s) => (
						<div key={s.id} className="pf-setting">
							<span>{t(s.titleKey)}</span>
							<AddonBoundary id={s.id}>
								<s.component />
							</AddonBoundary>
						</div>
					))}
					<a className="pf-link" href={href("checks")}>
						{t("checks.link")}
					</a>
					{entries.length ? (
						<button
							type="button"
							className="pf-btn pf-btn--small"
							onClick={() => {
								clearLogbook();
								setEntries([]);
							}}
						>
							{t("logbook.clear")}
						</button>
					) : null}
				</section>
				<WatchRun />
				<section className="pf-passes" aria-label={t("logbook.title")}>
					<h2 className="pf-h2">{t("logbook.title")}</h2>
					{entries.length === 0 ? (
						<div className="glass pf-empty">
							<p>{t("logbook.empty")}</p>
							<a className="pf-btn pf-btn--primary" href={href("new")}>
								{t("logbook.empty.cta")}
							</a>
						</div>
					) : (
						entries.map((e) => {
							const w = e.worst ? WORD[e.worst] : WORD.info;
							return (
								<a key={e.ref} className="glass pf-pass" href={href("cockpit", e.ref)} aria-label={t("logbook.open", { title: e.title })}>
									<span className="pf-pass__main">
										<span className="pf-pass__route">
											{t(`lens.${e.lens}`)}
											<Plane size={20} />
											{t("logbook.plan")}
										</span>
										<b>{e.title}</b>
										<span>{e.caught ? t("logbook.caught", { what: e.caught }) : t("logbook.nothing_caught")}</span>
									</span>
									<span className="pf-pass__stub">
										<span>{new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short" }).format(new Date(e.openedAt))}</span>
										<b>{e.gpu ?? t("logbook.gpu_unknown")}</b>
										<span className={`pf-status pf-status--${w.cls}`}>{t(w.key)}</span>
									</span>
								</a>
							);
						})
					)}
				</section>
			</div>
		</div>
	);
}
