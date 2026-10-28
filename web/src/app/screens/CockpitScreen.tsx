/**
 * The cockpit: dials on the left, the orbit job map in the middle, the guide (and a node's
 * detail when one is focused) on the right, the knob dock underneath, and the whole run
 * below. Everything reads one flight record through one PredictionProvider.
 */
import { useEffect, useMemo, useState } from "react";
import { GuidePanel } from "../../guide/GuidePanel";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { usePanels } from "../../registry";
import { FlightPlanBand, MemoryBand, ProbesBand } from "../../workspace/cockpit/Bands";
import { DetailSheet } from "../../workspace/cockpit/DetailSheet";
import { Dials } from "../../workspace/cockpit/Dials";
import { Dock } from "../../workspace/cockpit/Dock";
import { Orbit, useOrbitLayout } from "../../workspace/map/Orbit";
import { GridProvider } from "../../workspace/provider/GridProvider";
import { loadRecord } from "../../workspace/record/load";
import { isFixture, targetFor } from "../../workspace/record/model";
import type { LoadedRecord } from "../../workspace/record/types";
import { useWorkspace, WorkspaceProvider } from "../../workspace/state";
import { logFlight } from "../logbook";
import { href } from "../route";
import { AddonBoundary } from "../../components/AddonBoundary";

type Load = { kind: "loading" } | { kind: "ready"; record: LoadedRecord } | { kind: "error"; message: string };

export function CockpitScreen({ recordRef }: { recordRef: string }) {
	const [load, setLoad] = useState<Load>({ kind: "loading" });
	const provider = useMemo(() => new GridProvider(), []);
	useEffect(() => {
		let alive = true;
		setLoad({ kind: "loading" });
		loadRecord(recordRef)
			.then((record) => {
				if (!alive) return;
				setLoad({ kind: "ready", record });
				logFlight(recordRef, record);
			})
			.catch((e: unknown) => alive && setLoad({ kind: "error", message: e instanceof Error ? e.message : String(e) }));
		return () => {
			alive = false;
		};
	}, [recordRef]);

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
		<WorkspaceProvider key={recordRef} record={load.record} recordRef={recordRef} provider={provider}>
			<Cockpit />
		</WorkspaceProvider>
	);
}

function Cockpit() {
	const { record, state, estimates } = useWorkspace();
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
	return (
		<div className="pf-scroll">
			<div className="pf-inner">
				<section className="pf-cockpit">
					<header className="pf-cockpit__head">
						<h1>{record.job.title ?? t(`lens.${record.job.lens}`)}</h1>
						{record.job.objective?.description ? <span className="pf-dim">{record.job.objective.description}</span> : null}
						{isFixture(record) ? <span className="pf-tag">{t("cockpit.illustrative")}</span> : null}
						{spent > 0 ? <span className="pf-tag">{t("cockpit.made_for", { cost: fmt(spent, "usd") })}</span> : null}
						{headPanels.map((p) => (
							<AddonBoundary key={p.id} id={p.id}>
								<p.component record={record} knobs={state.knobs} estimates={estimates} />
							</AddonBoundary>
						))}
					</header>
					<div className="pf-deck">
						<Dials />
						<div className="pf-stage-wrap">
							<Orbit />
							<div className="pf-map-key">
								<span>
									<i className="pf-key-band" />
									{source.kind === "prediction"
										? t("map.key.band")
										: source.kind === "probe"
											? t("map.key.band_probe", { where })
											: t("map.key.band_none")}
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
							</div>
						</div>
						<div className="pf-side">
							<DetailSheet />
							<GuidePanel />
						</div>
					</div>
					<Dock />
				</section>
				<section className="pf-below" aria-label={t("below.label")}>
					<h2 className="pf-h2">{t("below.title")}</h2>
					<FlightPlanBand />
					<MemoryBand />
					<ProbesBand />
					{below.map((p) => (
						<AddonBoundary key={p.id} id={p.id}>
							<p.component record={record} knobs={state.knobs} estimates={estimates} />
						</AddonBoundary>
					))}
				</section>
			</div>
		</div>
	);
}
