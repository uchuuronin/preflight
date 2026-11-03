/**
 * The cockpit: dials on the left, the orbit job map in the middle, the guide (and a node's
 * detail when one is focused) on the right, the knob dock underneath, and the whole run
 * below. Everything reads one flight record through one PredictionProvider.
 */
import { useEffect, useMemo, useState } from "react";
import { Info } from "../../workspace/components/Info";
import { Measured } from "../../workspace/cockpit/Measured";
import { ProbeHere } from "../../workspace/cockpit/ProbeHere";
import { statusLine } from "../../workspace/cockpit/statusLine";
import { fitMapKnobs } from "../../workspace/record/fitmap";
import { GuidePanel } from "../../guide/GuidePanel";
import { fmt } from "../../lib/fmt";
import { errorCopy } from "../../lib/errors";
import { t } from "../../lib/i18n";
import { nothingMeasured, type ProbeSeen } from "../checkinTrouble";
import { usePanels } from "../../registry";
import { ExportsSection } from "../../workspace/export/ExportsSection";
import { CampaignCalendar } from "../../workspace/campaign/Calendar";
import { planFromRecord } from "../../workspace/campaign/plan";
import { currentKnobs } from "../../workspace/record/model";
import { estimateFactor } from "../../workspace/rate";
import { Onboarding } from "../Onboarding";
import { FlightPlanBand, MemoryBand } from "../../workspace/cockpit/Bands";
import { RlBand } from "../../workspace/cockpit/RlBand";
import { CostLine } from "../../workspace/cockpit/CostLine";
import { CostFrontier } from "../../workspace/cockpit/CostFrontier";
import { FitMap } from "../../workspace/cockpit/FitMap";
import { flag } from "../../lib/flags";
import { DetailSheet } from "../../workspace/cockpit/DetailSheet";
import { Dials } from "../../workspace/cockpit/Dials";
import { Dock } from "../../workspace/cockpit/Dock";
import { Orbit, useOrbitLayout } from "../../workspace/map/Orbit";
import { OrbitGlance } from "../../workspace/map/OrbitGlance";
import { FindingsBand, FindingsChip } from "../../workspace/cockpit/Findings";
import { GridProvider } from "../../workspace/provider/GridProvider";
import { loadRecord } from "../../workspace/record/load";
import { LIVE_REF, useLive } from "../../workspace/live/liveFile";
import { LivePanel } from "../../workspace/live/LivePanel";
import { isFixture, targetFor } from "../../workspace/record/model";
import type { LoadedRecord } from "../../workspace/record/types";
import { useWorkspace, WorkspaceProvider } from "../../workspace/state";
import { logFlight } from "../logbook";
import { go, href } from "../route";
import { AddonBoundary } from "../../components/AddonBoundary";

type Load = { kind: "loading" } | { kind: "ready"; record: LoadedRecord } | { kind: "error"; message: string };

export function CockpitScreen({ recordRef }: { recordRef: string }) {
	const [load, setLoad] = useState<Load>({ kind: "loading" });
	const provider = useMemo(() => new GridProvider(), []);
	// A live record picked again (another run) is a new record: reload. Its rewrites as the run goes
	// on only change the live probe, which LivePanel reads straight from the file.
	const liveId = useLive().record?.record_id;
	const reloadKey = recordRef === LIVE_REF ? `${recordRef}:${liveId ?? ""}` : recordRef;
	useEffect(() => {
		let alive = true;
		setLoad({ kind: "loading" });
		loadRecord(recordRef)
			.then((record) => {
				if (!alive) return;
				setLoad({ kind: "ready", record });
				if (recordRef !== LIVE_REF) logFlight(recordRef, record); // a live file isn't a link to reopen
			})
			.catch((e: unknown) => alive && setLoad({ kind: "error", message: e instanceof Error ? e.message : String(e) }));
		return () => {
			alive = false;
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [reloadKey]);

	if (load.kind === "loading") {
		return (
			<div className="pf-center-screen" role="status">
				{t("app.loading")}
			</div>
		);
	}
	if (load.kind === "error") {
		return (
			<div className="pf-center-screen" role="alert">
				<h1 className="pf-title">{t("app.load_failed")}</h1>
				<p className="pf-lede pf-lede--sm">{load.message}</p>
				<a className="pf-btn pf-btn--primary" href={href("new")}>
					{t("app.open_sample")}
				</a>
			</div>
		);
	}
	return (
		<WorkspaceProvider key={reloadKey} record={load.record} recordRef={recordRef} provider={provider}>
			<Cockpit />
		</WorkspaceProvider>
	);
}

function Cockpit() {
	const { record, state, estimates } = useWorkspace();
	// Every probe stopped before measuring (A's records keep each probe's status and code): nothing to read here.
	const nothingStop = isFixture(record) ? null : nothingMeasured(record.probes as ProbeSeen[] | undefined);
	const nothing = nothingStop?.step === "measure";
	const nothingWhy = errorCopy(nothingStop?.step === "measure" ? nothingStop.code : null);
	const headPanels = usePanels("cockpit-head", record.job.lens);
	const below = usePanels("below", record.job.lens);
	const layout = useOrbitLayout();
	const spent = (record.costs?.gpu_usd ?? 0) + (record.costs?.llm_usd ?? 0);
	const source = layout.shareSource;
	const memUnit = targetFor(record, "memory")?.unit;
	const knobs = new Map(record.knobs.map((k) => [k.id, k]));
	const where = Object.entries(source.probeKnobs ?? {})
		.slice(0, 2)
		.map(([k, v]) => `${t(knobs.get(k)?.label_key ?? k).toLowerCase()} ${String(v)}`)
		.join(", ");
	// Map or Fit on the stage: with Fit, the orbit steps aside into the side column.
	const canFit = flag("c.fit_map") && fitMapKnobs(record).length >= 2;
	const [view, setView] = useState<"map" | "fit">("map");
	const fit = canFit && view === "fit";
	const line = statusLine(record, estimates, state.knobs);
	// A record that carries a campaign (B's to_campaign) draws its calendar; behind its flag.
	const campaign = useMemo(
		() => (flag("c.campaign_calendar") && record.campaign ? planFromRecord(record.campaign, record.models, (k, e) => estimateFactor(record, { ...currentKnobs(record), ...k }, e, state.rate), record) : null),
		// B's plan is priced at the record's own setting, so only the visitor's rate changes its dollars.
		[record, state.rate],
	);
	const mapKey = (
		<Info label={t("map.label")} align="start">
			<span className="pf-map-key">
				<span>
					<i className="pf-key-band" />
					{source.kind === "prediction" ? t("map.key.band") : source.kind === "probe" ? t(isFixture(record) ? "map.key.band_probe_illustrative" : "map.key.band_probe", { where }) : t("map.key.band_none")}
				</span>
				<span>
					<i className="pf-key-ring" />
					{memUnit === "fraction"
						? t("map.key.ring_fraction")
						: layout.memory.capacity
							? t("map.key.ring", { cap: fmt(layout.memory.capacity, "bytes") })
							: t("map.key.ring_unknown")}
				</span>
				<span>
					<i className="pf-key-dot" />
					{t("map.key.dots")}
				</span>
			</span>
		</Info>
	);
	return (
		<div className="pf-scroll">
			<div className="pf-inner">
				<section className="pf-cockpit">
					<header className="pf-cockpit__head">
						<h1>{record.job.title ?? t(`lens.${record.job.lens}`)}</h1>

						{/* A sample's own cost is part of its fixture, not money anyone spent. */}
						{spent > 0 && !isFixture(record) ? <span className="pf-tag">{t("cockpit.made_for", { cost: fmt(spent, "usd") })}</span> : null}
						<Measured />
						{headPanels.map((p) => (
							<AddonBoundary key={p.id} id={p.id}>
								<p.component record={record} knobs={state.knobs} estimates={estimates} />
							</AddonBoundary>
						))}
						<p className={`pf-statusline is-${line.tone}`} aria-live="polite">
							{line.text}
						</p>
						<LivePanel />
						<CostLine />
						<FindingsChip />
						<ProbeHere />
					</header>
					{nothing ? (
						// A check-in whose probes never ran (PM, Nov 3): say so above the dials, not empty readouts under "done".
						<div className="pf-sample-banner pf-nothing-banner" role="status">
							<b>{t("cockpit.nothing_measured")}</b> {nothingWhy.code ? nothingWhy.message : t("checkin.measure.unknown")}
							{nothingWhy.code && nothingWhy.hint ? <> {nothingWhy.hint}</> : null}{" "}
							<button type="button" className="pf-link" onClick={() => go("new")}>
								{t("cockpit.nothing_again")}
							</button>
						</div>
					) : null}
					{isFixture(record) ? (
						<p className="pf-sample-banner" role="note">
							<b>{t("cockpit.illustrative")}</b>{" "}
							<button type="button" className="pf-link" onClick={() => go("new")}>
								{t("cockpit.check_in_own")}
							</button>
						</p>
					) : null}
					{/* A node's detail widens the side column, so its text runs wide instead of long. */}
					<div className={`pf-deck${fit ? " is-fit" : ""}${state.focus ? " has-detail" : ""}`}>
						<Dials />
						<div className="pf-stage-wrap">
							<div className="pf-stage-bar">
								{canFit ? (
									<div className="pf-seg pf-seg--small" role="tablist" aria-label={t("stage.view")}>
										{(["map", "fit"] as const).map((v) => (
											<button key={v} type="button" role="tab" aria-selected={(v === "fit") === fit} onClick={() => setView(v)}>
												{t(`stage.${v}`)}
											</button>
										))}
									</div>
								) : null}
								{fit ? null : mapKey}
							</div>
							{/* Whichever view has the stage fills it: as wide and tall as the space between the dials and the side column allows. */}
							<div className="pf-stage-fill">{fit ? <FitMap /> : <Orbit />}</div>
						</div>
						<div className="pf-side">
							{fit ? (
								<div className="glass pf-mini-orbit">
									<OrbitGlance />
									<button type="button" className="pf-mini-orbit__back" aria-label={t("stage.back_to_map")} title={t("stage.back_to_map")} onClick={() => setView("map")} />
								</div>
							) : null}
							<DetailSheet />
							<GuidePanel />
						</div>
					</div>
					<Dock />
					<Onboarding />
				</section>
				<section className="pf-below" aria-label={t("below.label")}>
					<h2 className="pf-h2">{t("below.title")}</h2>
					<FindingsBand />
					<FlightPlanBand />
					<MemoryBand />
					<AddonBoundary id="rl-band">
						<RlBand />
					</AddonBoundary>
					<AddonBoundary id="cost-frontier">
						<CostFrontier />
					</AddonBoundary>
					{campaign ? (
						<AddonBoundary id="campaign-calendar">
							<CampaignCalendar plan={campaign} />
						</AddonBoundary>
					) : null}
					{below.map((p) => (
						<AddonBoundary key={p.id} id={p.id}>
							<p.component record={record} knobs={state.knobs} estimates={estimates} />
						</AddonBoundary>
					))}
				</section>
				<ExportsSection />
			</div>
		</div>
	);
}
