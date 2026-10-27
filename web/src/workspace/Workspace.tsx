import { useState } from "react";
import { GuidePanel } from "../guide/GuidePanel";
import { t } from "../lib/i18n";
import { usePanels } from "../registry";
import { FlightPlan } from "./components/FlightPlan";
import { KnobRail } from "./components/KnobRail";
import { NodeDrawer } from "./components/NodeDrawer";
import { Readouts } from "./components/Readouts";
import { JobMap } from "./map/JobMap";
import { useWorkspace } from "./state";

export function Workspace() {
	const { record, state, estimates } = useWorkspace();
	const [selected, setSelected] = useState<string | null>(null);
	const views = usePanels("views", record.job.lens);
	const headerEnd = usePanels("header-end", record.job.lens);
	const [view, setView] = useState<string | null>(null);
	const activeView = views.find((v) => v.id === view) ?? null;

	return (
		<div className="pf-app">
			<header className="pf-top">
				<div className="pf-brand">
					<span className="pf-brand__mark" aria-hidden="true">
						<svg viewBox="0 0 24 24" width="20" height="20">
							<path d="M2 13.5 L22 4 L14.5 21 L11.5 14.5 Z" />
						</svg>
					</span>
					<span className="pf-brand__name">{t("app.name")}</span>
				</div>
				<h1 className="pf-job-title">
					{record.job.title ?? t(`lens.${record.job.lens}`)}
					<span className="pf-job-lens">{t(`lens.${record.job.lens}`)}</span>
				</h1>
				{record.fixture ? <p className="pf-fixture">{t("app.fixture_notice")}</p> : null}
				<div className="pf-top__end">
					{headerEnd.map((p) => (
						<p.component key={p.id} record={record} knobs={state.knobs} estimates={estimates} />
					))}
				</div>
			</header>

			<Readouts />

			<main className="pf-main">
				<KnobRail />
				<section className="pf-center">
					<div className="pf-center__map">
						<JobMap onSelect={setSelected} />
						{selected ? <NodeDrawer nodeId={selected} onClose={() => setSelected(null)} /> : null}
					</div>
					{views.length ? (
						<div className="pf-views">
							<div className="pf-views__tabs" role="tablist" aria-label={t("views.label")}>
								{views.map((v) => (
									<button
										key={v.id}
										type="button"
										role="tab"
										aria-selected={activeView?.id === v.id}
										className={`pf-tab${activeView?.id === v.id ? " is-on" : ""}`}
										onClick={() => setView(activeView?.id === v.id ? null : v.id)}
									>
										{t(v.titleKey)}
									</button>
								))}
							</div>
							{activeView ? (
								<div role="tabpanel" className="pf-views__panel">
									<activeView.component record={record} knobs={state.knobs} estimates={estimates} />
								</div>
							) : null}
						</div>
					) : null}
				</section>
				<GuidePanel />
			</main>

			<FlightPlan />
		</div>
	);
}
