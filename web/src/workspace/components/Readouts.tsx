import { memo } from "react";
import { fmt, fmtDelta, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useReadouts } from "../../registry";
import type { Estimate, TargetSpec } from "../record/types";
import { useWorkspace } from "../state";
import { capacityBytes, memoryFit, trusted } from "../status";
import { FitMark, Provenance } from "./StatusMark";

/** The four always-on readouts plus anything an add-on registers. */
export function Readouts() {
	const { record, estimates, state } = useWorkspace();
	const entries = useReadouts(record.job.lens);
	const targets = new Map(record.predictions.targets.map((tg) => [tg.id, tg]));
	return (
		<section className="pf-readouts" aria-label={t("readouts.label")}>
			{entries.map((entry) => {
				const target = targets.get(entry.targetId);
				if (!target) return null;
				return (
					<Readout
						key={entry.id}
						title={t(entry.titleKey)}
						target={target}
						est={estimates[entry.targetId]}
						prev={state.previous?.[entry.targetId]}
						extra={
							entry.targetId === "peak_memory" ? (
								<MemoryExtra />
							) : null
						}
						accent={state.highlight.includes(`readout:${entry.targetId}`)}
					/>
				);
			})}
		</section>
	);
}

function MemoryExtra() {
	const { record, estimates, state } = useWorkspace();
	const cap = capacityBytes(record, estimates, String(state.knobs.gpu ?? ""));
	const fit = memoryFit(estimates, record);
	return (
		<div className="pf-readout__extra">
			<FitMark fit={fit} />
			{cap ? <span className="pf-muted">{t("readout.of_capacity", { cap: fmt(cap, "bytes") })}</span> : null}
		</div>
	);
}

const Readout = memo(function Readout({
	title,
	target,
	est,
	prev,
	extra,
	accent,
}: {
	title: string;
	target: TargetSpec;
	est: Estimate | undefined;
	prev: Estimate | undefined;
	extra: React.ReactNode;
	accent: boolean;
}) {
	const ok = trusted(est);
	const delta = ok && trusted(prev) ? est.point - prev.point : 0;
	const better = target.better ?? "lower";
	const good = delta !== 0 && (better === "lower" ? delta < 0 : delta > 0);
	// Deltas smaller than the interval's half-width are noise; say so instead of celebrating.
	const halfWidth = ok ? (est.hi - est.lo) / 2 : 0;
	const meaningful = Math.abs(delta) > halfWidth * 0.25;
	return (
		<article className={`pf-readout${accent ? " is-accent" : ""}${ok ? "" : " is-abstain"}`}>
			<header className="pf-readout__head">
				<h2 className="pf-readout__title">{title}</h2>
				{est ? <Provenance status={est.status} /> : null}
			</header>
			<div className="pf-readout__value" aria-live="polite">
				{ok ? fmt(est.point, target.unit) : t("readout.abstain")}
			</div>
			<div className="pf-readout__range">
				{ok ? t("readout.interval", { range: fmtRange(est.lo, est.hi, target.unit) }) : t(`abstain.${est?.reason ?? "off_grid"}`)}
			</div>
			{delta !== 0 && ok ? (
				<span
					key={`${est.point}`}
					className={`pf-delta ${meaningful ? (good ? "is-good" : "is-bad") : "is-noise"}`}
					title={meaningful ? undefined : t("readout.within_noise")}
				>
					{fmtDelta(delta, target.unit)}
				</span>
			) : null}
			{extra}
		</article>
	);
});
