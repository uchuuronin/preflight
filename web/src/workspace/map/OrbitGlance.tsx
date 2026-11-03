/**
 * The orbit at a glance, in the side column while another view has the stage: the whole loop
 * framed to fit (never a cropped corner of the full map), with only what reads at that size. The
 * step-time band (each phase as long as its share of a step), the memory ring against this GPU
 * (with the dashed overflow past it), the peak in the middle with its fits, tight or fails mark,
 * and the three phases that take the most of a step. No nodes, chips or moving dots: the full map
 * is one press away (the side column's "Back to the map" covers it).
 */
import { useMemo } from "react";
import { fmt } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { FitMark } from "../components/StatusMark";
import { targetFor } from "../record/model";
import { useWorkspace } from "../state";
import { fitOf, trusted } from "../status";
import { memoryRings, useOrbitLayout } from "./Orbit";
import { arcDash } from "./orbitLayout";

/**
 * The glance's own geometry (a 240 unit square), not the full map's: at this size the memory ring
 * is drawn larger against the band so the peak and its mark fit inside it.
 */
const G = { size: 240, c: 120, band: 110, mem: 80, over: 93 };

export function OrbitGlance() {
	const { record, estimates, state } = useWorkspace();
	const layout = useOrbitLayout();
	const { segs, untrustedArc, over, cOver } = memoryRings(layout.memory, G.mem, G.over);
	const memT = targetFor(record, "memory");
	const peak = memT ? estimates[memT.id] : undefined;
	const fit = fitOf(record, estimates, state.knobs);
	const cap = layout.memory.capacity;
	const top = useMemo(() => [...layout.arcs].filter((a) => a.share > 0).sort((a, b) => b.share - a.share).slice(0, 3), [layout.arcs]);
	const rot = (deg: number) => `rotate(${deg.toFixed(2)} ${G.c} ${G.c})`;
	return (
		<figure className="pf-glance" aria-label={t("map.label")}>
			<div className="pf-glance__dial">
				<svg viewBox={`0 0 ${G.size} ${G.size}`} aria-hidden="true">
					{layout.arcs.map((a) => (
						<circle key={a.phase} className="pf-glance__arc" cx={G.c} cy={G.c} r={G.band} strokeDasharray={arcDash(a.sweepDeg, G.band)} transform={rot(a.startDeg)} />
					))}
					<circle className="pf-glance__track" cx={G.c} cy={G.c} r={G.mem} />
					{segs.map((s) => (
						<circle key={s.key} className={`pf-glance__mring pf-mring--${s.slot}`} cx={G.c} cy={G.c} r={G.mem} strokeDasharray={s.dash} transform={rot(s.rot)} />
					))}
					{untrustedArc ? <circle className="pf-glance__untrusted" cx={G.c} cy={G.c} r={G.mem} strokeDasharray={untrustedArc.dash} transform={rot(untrustedArc.rot)} /> : null}
					{over > 0 ? (
						<circle className="pf-mem-over" cx={G.c} cy={G.c} r={G.over} strokeDasharray={`${(over * cOver).toFixed(1)} ${cOver.toFixed(1)}`} transform={rot(-90)} />
					) : null}
				</svg>
				<div className="pf-glance__core">
					<span className={`pf-glance__num${trusted(peak) ? "" : " is-words"}`}>{trusted(peak) ? fmt(peak.point, memT?.unit ?? "bytes") : t(fit === "fails" ? "map.core.oom" : fit === "tight" ? "map.core.tight" : "readout.abstain_short")}</span>
					{cap && memT?.unit === "bytes" ? <span className="pf-glance__cap">{t("glance.of", { cap: fmt(cap, "bytes") })}</span> : null}
					{fit !== "unknown" ? <FitMark fit={fit} /> : null}
				</div>
			</div>
			{top.length ? (
				<figcaption className="pf-glance__phases">
					<span className="pf-visually-hidden">{t("glance.step")}</span>
					{top.map((a) => (
						<span key={a.phase} className="pf-glance__phase">
							<i aria-hidden="true" style={{ width: `${Math.max(6, a.share * 100)}%` }} />
							{hasKey(`phase.${a.phase}`) ? t(`phase.${a.phase}`) : a.phase} <b>{fmt(a.share, "fraction")}</b>
						</span>
					))}
				</figcaption>
			) : null}
		</figure>
	);
}
