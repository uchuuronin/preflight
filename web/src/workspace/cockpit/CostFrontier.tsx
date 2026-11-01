/**
 * Cost frontier for a model sweep (workspace/frontier.ts): one point per model at its mean cost per
 * task (across) and success rate (up), each with its 95% intervals, the models no other beats on
 * both joined by a line, and every point named beside it. Hover or focus a point for its numbers;
 * the same numbers are in a table under the chart. Shown only for a sweep (two or more models with
 * scored tasks), so a single-model run draws nothing here.
 */
import { useEffect, useRef, useState } from "react";
import { fmt, fmtRange } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { type FrontierPoint, frontierPoints } from "../frontier";
import { useWorkspace } from "../state";

/** Source: guess, a plot that reads on a phone and doesn't sprawl on a desktop; drawn at its real width, never scaled. */
const W_MAX = 640;
const H = 300;
const PAD = { l: 52, r: 24, t: 16, b: 40 };

function modelName(id: string): string {
	return hasKey(`model.${id}`) ? t(`model.${id}`) : id;
}

function said(p: FrontierPoint): string {
	return t("frontier.point", {
		model: modelName(p.model),
		cost: fmt(p.costMean, "usd_small"),
		costRange: fmtRange(p.costLo, p.costHi, "usd_small"),
		k: fmt(p.success.k, "count"),
		n: fmt(p.success.n, "count"),
		rate: fmt(p.success.p, "probability"),
		rateRange: fmtRange(p.success.lo, p.success.hi, "probability"),
		front: t(p.dominated ? "frontier.point_off" : "frontier.point_on"),
	});
}

export function CostFrontier() {
	const { record } = useWorkspace();
	const pts = frontierPoints(record);
	const [hover, setHover] = useState<string | null>(null);
	const box = useRef<HTMLDivElement>(null);
	const [W, setW] = useState(W_MAX);
	useEffect(() => {
		const el = box.current;
		if (!el || typeof ResizeObserver === "undefined") return;
		const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.min(W_MAX, Math.round(e!.contentRect.width)))));
		ro.observe(el);
		return () => ro.disconnect();
	}, [pts.length >= 2]);
	if (pts.length < 2) return null;
	const xMax = Math.max(...pts.map((p) => p.costHi)) * 1.1 || 1;
	const x = (v: number) => PAD.l + (v / xMax) * (W - PAD.l - PAD.r);
	const y = (v: number) => H - PAD.b - v * (H - PAD.t - PAD.b);
	const front = pts.filter((p) => !p.dominated);
	// Three cost ticks when narrow, so the labels never run into each other.
	const ticksX = (W < 480 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map((f) => f * xMax);
	const ticksY = [0, 0.25, 0.5, 0.75, 1];
	const shown = pts.find((p) => p.model === hover);

	return (
		<section className="glass pf-band pf-frontier" aria-label={t("frontier.title")}>
			<div className="pf-band__head">
				<b>{t("frontier.title")}</b>
			</div>
			<p className="pf-frontier__lede">{t("frontier.lede", { n: fmt(pts.length, "count"), front: front.map((p) => modelName(p.model)).join(", ") })}</p>
			<div className="pf-frontier__plot" ref={box}>
				<svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={t("frontier.aria", { n: fmt(pts.length, "count") })}>
					{ticksY.map((v) => (
						<g key={`y${v}`}>
							<line className="pf-frontier__grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
							<text className="pf-frontier__tick" x={PAD.l - 8} y={y(v) + 4} textAnchor="end">
								{fmt(v, "probability")}
							</text>
						</g>
					))}
					{ticksX.map((v) => (
						<text key={`x${v}`} className="pf-frontier__tick" x={x(v)} y={H - PAD.b + 18} textAnchor="middle">
							{fmt(v, "usd_small")}
						</text>
					))}
					<text className="pf-frontier__axis" x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle">
						{t("frontier.x")}
					</text>
					<text className="pf-frontier__axis" x={12} y={(PAD.t + H - PAD.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(PAD.t + H - PAD.b) / 2})`}>
						{t("frontier.y")}
					</text>
					{front.length > 1 ? <polyline className="pf-frontier__line" points={front.map((p) => `${x(p.costMean)},${y(p.success.p)}`).join(" ")} /> : null}
					{pts.map((p) => (
						<g
							key={p.model}
							className={`pf-frontier__pt${p.dominated ? " is-dominated" : ""}${hover === p.model ? " is-hover" : ""}`}
							tabIndex={0}
							role="img"
							aria-label={said(p)}
							onPointerEnter={() => setHover(p.model)}
							onPointerLeave={() => setHover(null)}
							onFocus={() => setHover(p.model)}
							onBlur={() => setHover(null)}
						>
							<line className="pf-frontier__err" x1={x(p.costLo)} x2={x(p.costHi)} y1={y(p.success.p)} y2={y(p.success.p)} />
							<line className="pf-frontier__err" x1={x(p.costMean)} x2={x(p.costMean)} y1={y(p.success.lo)} y2={y(p.success.hi)} />
							{/* A larger invisible target than the mark, for pointer and touch. */}
							<circle className="pf-frontier__hit" cx={x(p.costMean)} cy={y(p.success.p)} r={14} />
							<circle className="pf-frontier__dot" cx={x(p.costMean)} cy={y(p.success.p)} r={5} />
							<text className="pf-frontier__label" x={x(p.costMean) + 9} y={y(p.success.p) - 9}>
								{modelName(p.model)}
							</text>
						</g>
					))}
				</svg>
				{shown ? (
					<div
						className="pf-frontier__tip"
						role="status"
						style={{ left: `${Math.max(0, Math.min(x(shown.costMean) - 24, W - 244))}px`, top: `${Math.max(0, y(shown.success.hi) - 8)}px` }}
					>
						{said(shown)}
					</div>
				) : null}
			</div>
			<details className="pf-frontier__table">
				<summary>{t("frontier.table")}</summary>
				<table>
					<thead>
						<tr>
							<th scope="col">{t("frontier.col.model")}</th>
							<th scope="col">{t("frontier.col.cost")}</th>
							<th scope="col">{t("frontier.col.success")}</th>
							<th scope="col">{t("frontier.col.front")}</th>
						</tr>
					</thead>
					<tbody>
						{pts.map((p) => (
							<tr key={p.model}>
								<th scope="row">{modelName(p.model)}</th>
								<td>
									{fmt(p.costMean, "usd_small")} ({fmtRange(p.costLo, p.costHi, "usd_small")})
								</td>
								<td>
									{t("frontier.k_of_n", { k: fmt(p.success.k, "count"), n: fmt(p.success.n, "count") })} ({fmtRange(p.success.lo, p.success.hi, "probability")})
								</td>
								<td>{p.dominated ? t("frontier.beaten") : t("frontier.on_front")}</td>
							</tr>
						))}
					</tbody>
				</table>
			</details>
		</section>
	);
}
