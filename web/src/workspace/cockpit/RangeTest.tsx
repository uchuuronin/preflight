/**
 * The learning-rate range test's curve, inside the learning-rate knob's editor (D's spec, Nov 2): loss
 * against the rate on a log axis, with the steepest fall (the suggestion), where the loss blew up, and
 * the run's own rate marked. The run's marker follows the knob, so moving the slider moves it along the
 * curve, and a click on the curve moves the knob to its nearest choice there. Only on a record whose
 * range-test probe ran (`probes.lr_range`); never a point on the fits (B's `api._fit_view`).
 *
 * A ramp whose updates didn't follow the rate keeps its losses but no rates (A's `with_lr_range`):
 * then the losses are drawn by step and the words say why no rate is read. A flat curve says
 * "No clear slope". Red marks only the stretch past where the loss blew up.
 */
import { useEffect, useId, useMemo, useState } from "react";
import { t } from "../../lib/i18n";
import type { KnobValue } from "../record/types";
import { nearestRate, rateText, rateVerdict, smoothed, type RangeTest as Test } from "./rangeTest";

const H = 168;
const PAD = { l: 40, r: 12, t: 26, b: 26 };
const IH = H - PAD.t - PAD.b;
/** The plot is drawn at its own width in pixels (one unit a pixel), so its text stays 10 px on a phone. */
const MIN_W = 300;
const MAX_W = 640;
/** Source: guess, the top of the y axis: the 90th percentile of the raw losses (or the smoothed curve's top,
 * if higher), with 8% air; a late spike past it is clipped to the edge and the label says "+". */
const TOP_Q = 0.9;

function useWidth(): [(el: HTMLDivElement | null) => void, number] {
	const [w, setW] = useState(520);
	const [el, setEl] = useState<HTMLDivElement | null>(null);
	useEffect(() => {
		if (!el || typeof ResizeObserver === "undefined") return;
		const ro = new ResizeObserver(([e]) => {
			const next = Math.round(Math.min(MAX_W, Math.max(MIN_W, e?.contentRect.width ?? 520)));
			setW((prev) => (prev === next ? prev : next));
		});
		ro.observe(el);
		return () => ro.disconnect();
	}, [el]);
	return [setEl, w];
}

export function RangeTestCurve({ test, value, choices, onPick }: { test: Test; value: KnobValue | undefined; choices: readonly KnobValue[]; onPick?: ((v: number) => void) | null }) {
	const id = useId();
	const [hover, setHover] = useState<number | null>(null);
	const rate = typeof value === "number" && value > 0 ? value : null;
	const byRate = test.checked && test.points.length > 1 && test.points.every((p) => p.lr !== null);

	if (test.points.length < 2) {
		// Nothing to draw (the recorder couldn't read the loss): the words alone (review, Nov 2).
		return (
			<figure className="pf-lrr" aria-labelledby={`${id}-cap`}>
				<figcaption id={`${id}-cap`} className="pf-lrr__cap">
					<b>{t("lrr.title")}</b>
				</figcaption>
				<p className="pf-lrr__said">{t("lrr.no_curve")}</p>
			</figure>
		);
	}
	return <Curve test={test} rate={rate} choices={choices} onPick={onPick ?? null} byRate={byRate} id={id} hover={hover} setHover={setHover} />;
}

/** The words under the curve: the flat curve, why no rate is read, or where the run's rate sits. */
function said(test: Test, byRate: boolean, verdict: ReturnType<typeof rateVerdict>, rate: number | null): string {
	if (test.flat) return t("lrr.flat");
	if (!byRate) return test.checked ? t("lrr.too_few") : t(`lrr.unchecked.${test.reasons[0] ?? "unverified"}`);
	if (verdict === "unknown") return rate === null ? t("lrr.no_rate") : t("lrr.verdict.unknown");
	return t(`lrr.verdict.${verdict}`, { v: rate ? rateText(rate) : "" });
}

function Curve({ test, rate, choices, onPick, byRate, id, hover, setHover }: { test: Test; rate: number | null; choices: readonly KnobValue[]; onPick: ((v: number) => void) | null; byRate: boolean; id: string; hover: number | null; setHover: (v: number | null) => void }) {
	const [plotRef, W] = useWidth();
	const IW = W - PAD.l - PAD.r;
	const smooth = useMemo(() => smoothed(test.points.map((p) => p.loss)), [test]);
	const geo = useMemo(() => {
		const losses = test.points.map((p) => p.loss);
		const sorted = [...losses].sort((a, b) => a - b);
		const lo = Math.min(...losses, ...smooth);
		const hiRaw = Math.max(...losses);
		const q = sorted[Math.min(sorted.length - 1, Math.floor(TOP_Q * (sorted.length - 1)))] ?? hiRaw;
		const top = Math.max(q, ...smooth);
		const hi = Math.min(hiRaw, top + (top - lo) * 0.08);
		const span = hi - lo || Math.abs(hi) || 1;
		const yOf = (loss: number) => PAD.t + IH - ((Math.min(loss, hi) - lo) / span) * IH;
		if (byRate) {
			const rates = test.points.map((p) => p.lr as number);
			const all = [...rates, ...(test.diverged ? [test.diverged] : []), ...(rate ? [rate] : [])];
			const x0 = Math.floor(Math.log10(Math.min(...all)));
			const x1 = Math.ceil(Math.log10(Math.max(...all)));
			const xOf = (r: number) => PAD.l + ((Math.log10(r) - x0) / Math.max(1, x1 - x0)) * IW;
			const rateAt = (px: number) => 10 ** (x0 + ((px - PAD.l) / IW) * Math.max(1, x1 - x0));
			const ticks = Array.from({ length: x1 - x0 + 1 }, (_, i) => x0 + i).filter((d, i, a) => a.length <= 8 || (d - x0) % 2 === 0);
			return { xOf, yOf, rateAt, ticks, lo, hi, clipped: hiRaw > hi };
		}
		const n = Math.max(1, test.points.length - 1);
		const xOf = (i: number) => PAD.l + (i / n) * IW;
		return { xOf, yOf, rateAt: null, ticks: [] as number[], lo, hi, clipped: hiRaw > hi };
	}, [test, byRate, rate, smooth, IW]);

	const xAt = (i: number) => (byRate ? geo.xOf(test.points[i]!.lr as number) : geo.xOf(i));
	const path = test.points.map((p, i) => `${i ? "L" : "M"}${xAt(i).toFixed(1)},${geo.yOf(p.loss).toFixed(1)}`).join(" ");
	// The curve the suggestion is read from (D's smoothing), drawn over the raw losses so the slope reads.
	const smoothPath = smooth.map((y, i) => `${i ? "L" : "M"}${xAt(i).toFixed(1)},${geo.yOf(y).toFixed(1)}`).join(" ");
	const verdict = rateVerdict(test, rate);
	const pickAt = (clientX: number, box: DOMRect) => {
		if (!geo.rateAt) return null;
		const px = ((clientX - box.left) / box.width) * W;
		if (px < PAD.l || px > W - PAD.r) return null;
		return geo.rateAt(px);
	};
	const nearSuggested = test.suggested ? nearestRate(choices, test.suggested) : null;
	const hoverChoice = hover !== null ? nearestRate(choices, hover) : null;
	const point = (r: number) => {
		// the curve's loss nearest a rate, for the hover readout
		let best = test.points[0];
		for (const p of test.points) if (p.lr !== null && best?.lr != null && Math.abs(Math.log10(p.lr) - Math.log10(r)) < Math.abs(Math.log10(best.lr) - Math.log10(r))) best = p;
		return best;
	};
	const summary = byRate
		? t("lrr.aria", {
				suggested: test.suggested ? rateText(test.suggested) : t("lrr.none"),
				diverged: test.diverged ? rateText(test.diverged) : t("lrr.none"),
				yours: rate ? rateText(rate) : t("lrr.none"),
			})
		: t("lrr.aria_unchecked");

	return (
		<figure className="pf-lrr" aria-labelledby={`${id}-cap`}>
			<figcaption id={`${id}-cap`} className="pf-lrr__cap">
				<b>{t("lrr.title")}</b>
				<span>{test.gpu ? t("lrr.where", { gpu: test.gpu }) : t("lrr.where_any")}</span>
			</figcaption>
			<div
				ref={plotRef}
				className={`pf-lrr__plot${byRate && onPick ? " is-live" : ""}`}
				onPointerMove={(e) => setHover(pickAt(e.clientX, e.currentTarget.getBoundingClientRect()))}
				onPointerLeave={() => setHover(null)}
				onClick={(e) => {
					const r = pickAt(e.clientX, e.currentTarget.getBoundingClientRect());
					const v = r !== null ? nearestRate(choices, r) : null;
					if (v !== null) onPick?.(v);
				}}
			>
				<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={summary}>
					{/* the stretch past where the loss blew up: red, the app's one meaning for it */}
					{byRate && test.diverged ? (
						<rect className="pf-lrr__blowup" x={geo.xOf(test.diverged)} y={PAD.t} width={Math.max(0, W - PAD.r - geo.xOf(test.diverged))} height={IH} />
					) : null}
					<line className="pf-lrr__axis" x1={PAD.l} x2={W - PAD.r} y1={PAD.t + IH} y2={PAD.t + IH} />
					{geo.ticks.map((d) => (
						<g key={d}>
							<line className="pf-lrr__grid" x1={geo.xOf(10 ** d)} x2={geo.xOf(10 ** d)} y1={PAD.t} y2={PAD.t + IH} />
							<text className="pf-lrr__tick" x={geo.xOf(10 ** d)} y={H - 8} textAnchor="middle">
								{d === 0 ? "1" : `1e${d}`}
							</text>
						</g>
					))}
					<text className="pf-lrr__tick" x={PAD.l - 6} y={geo.yOf(geo.lo) + 3} textAnchor="end">
						{geo.lo.toPrecision(2)}
					</text>
					<text className="pf-lrr__tick" x={PAD.l - 6} y={geo.yOf(geo.hi) + 3} textAnchor="end">
						{geo.clipped ? `${geo.hi.toPrecision(2)}+` : geo.hi.toPrecision(2)}
					</text>
					<text className="pf-lrr__tick pf-lrr__ylab" x={8} y={PAD.t + IH / 2} transform={`rotate(-90 8 ${PAD.t + IH / 2})`} textAnchor="middle">
						{t("lrr.loss")}
					</text>
					<path className="pf-lrr__raw" d={path} />
					<path className="pf-lrr__line" d={smoothPath} />
					{byRate && test.suggested ? <Mark w={W} x={geo.xOf(test.suggested)} kind="suggested" label={t("lrr.mark.suggested", { v: rateText(test.suggested) })} /> : null}
					{byRate && test.diverged ? <Mark w={W} x={geo.xOf(test.diverged)} kind="diverged" label={t("lrr.mark.diverged", { v: rateText(test.diverged) })} below /> : null}
					{byRate && rate ? <Mark w={W} x={geo.xOf(rate)} kind={verdict === "past_divergence" ? "yours-bad" : "yours"} label={t("lrr.mark.yours", { v: rateText(rate) })} low /> : null}
					{byRate && hover !== null && hoverChoice !== null ? (
						<g className="pf-lrr__hover" aria-hidden="true">
							<line x1={geo.xOf(hover)} x2={geo.xOf(hover)} y1={PAD.t} y2={PAD.t + IH} />
							<circle cx={geo.xOf(hover)} cy={geo.yOf(point(hover)?.loss ?? geo.lo)} r={3.5} />
						</g>
					) : null}
				</svg>
				{byRate && onPick && hover !== null && hoverChoice !== null ? (
					<span className="pf-lrr__tip" aria-hidden="true">
						{t("lrr.click_to", { v: rateText(hoverChoice) })}
					</span>
				) : null}
			</div>
			<p className={`pf-lrr__said is-${verdict}`} aria-live="polite">
				{said(test, byRate, verdict, rate)}
			</p>
			{byRate && onPick && nearSuggested !== null && nearSuggested !== rate ? (
				<button type="button" className="pf-link" onClick={() => onPick(nearSuggested)}>
					{t("lrr.use", { v: rateText(nearSuggested) })}
				</button>
			) : null}
			<p className="pf-lrr__note pf-dim">{t("lrr.note")}</p>
		</figure>
	);
}

function Mark({ w, x, kind, label, below, low }: { w: number; x: number; kind: "suggested" | "diverged" | "yours" | "yours-bad"; label: string; below?: boolean; low?: boolean }) {
	// Labels sit above the plot (suggested), inside its top (yours) or at its foot (diverged), and
	// turn to the left near the right edge, so three marks close together don't print over each other.
	const right = x > w - 120;
	const y = below ? PAD.t + IH - 6 : low ? PAD.t + 12 : PAD.t - 8;
	// Placed by a transform, not by x: CSS can ease a transform, so the run's marker glides with the knob.
	return (
		<g className={`pf-lrr__mark is-${kind}`} style={{ transform: `translateX(${x.toFixed(1)}px)` }}>
			<line x1={0} x2={0} y1={PAD.t} y2={PAD.t + IH} />
			<text x={right ? -5 : 5} y={y} textAnchor={right ? "end" : "start"}>
				{label}
			</text>
		</g>
	);
}
