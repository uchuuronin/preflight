/**
 * "Where it fits": two knobs against each other, every combination marked fits, tight, fails or
 * not trusted (shape and word, never colour alone), with the out-of-memory boundary drawn
 * between squares from interpolated memory (record/fitmap.ts). Pressing a square flies there.
 * Behind the c.fit_map flag.
 */
import { useDeferredValue, useId, useMemo, useState } from "react";
import { valueText } from "../../guide/GuidePanel";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { FitMark } from "../components/StatusMark";
import { type CellStatus, defaultAxes, type FitCell, type FitMapData, fitMap, fitMapKnobs } from "../record/fitmap";
import { chosenGpu } from "../record/terms";
import { useWorkspace } from "../state";

function statusWord(s: CellStatus): string {
	return s === "abstain" ? t("provenance.abstain") : t(`fit.${s}`);
}

function cellLabel(data: FitMapData, c: FitCell): string {
	const parts = [
		t("fitmap.cell", {
			x: `${t(data.x.label_key)} ${valueText(data.x, c.x)}`,
			y: `${t(data.y.label_key)} ${valueText(data.y, c.y)}`,
			status: statusWord(c.status),
		}),
	];
	if (c.peak !== null && c.capacity) parts.push(t("fitmap.peak", { peak: fmt(c.peak, "bytes"), cap: fmt(c.capacity, "bytes") }));
	if (c.risk !== null) parts.push(t("fitmap.risk", { risk: fmt(c.risk, "probability") }));
	if (c.here) parts.push(t("fitmap.here"));
	return parts.join(". ");
}

/** The boundary as an SVG path in cell units: a vertical stroke per row, joined at row edges. */
function boundaryPath(cross: Array<number | null>, rowsTopDown: number[]): string {
	let d = "";
	let prev: number | null = null;
	rowsTopDown.forEach((row, r) => {
		const c = cross[row];
		if (c === null || c === undefined) {
			prev = null;
			return;
		}
		const x = c + 0.5;
		if (prev === null) d += `M${x.toFixed(3)} ${r}`;
		else d += `H${x.toFixed(3)}`;
		d += `V${r + 1}`;
		prev = x;
	});
	return d;
}

export function FitMap() {
	const { record, provider, state, actions } = useWorkspace();
	// Off the critical path: the map catches up after the dials have answered a knob change.
	const knobs = useDeferredValue(state.knobs);
	const id = useId();
	const options = useMemo(() => fitMapKnobs(record), [record]);
	const initial = useMemo(() => defaultAxes(record, provider, knobs), [record, provider]); // eslint-disable-line react-hooks/exhaustive-deps
	const [axes, setAxes] = useState<[string, string] | null>(initial);
	const data = useMemo(() => (axes ? fitMap(record, provider, knobs, axes[0], axes[1]) : null), [record, provider, knobs, axes]);

	if (!axes || !data || options.length < 2) {
		return (
			<div className="glass pf-band">
				<div className="pf-band__head">
					<b>{t("fitmap.title")}</b>
					<span className="pf-dim">{t("fitmap.none")}</span>
				</div>
			</div>
		);
	}
	const gpu = chosenGpu(record, knobs);
	const cap = data.rows[0]?.[0]?.capacity ?? null;
	const rowsTopDown = data.ys.map((_, i) => data.ys.length - 1 - i);
	const cols = data.xs.length;
	const rows = data.ys.length;
	const pick = (which: 0 | 1, value: string) => {
		const other = axes[which === 0 ? 1 : 0];
		const next: [string, string] = which === 0 ? [value, value === other ? axes[0] : other] : [value === other ? axes[1] : other, value];
		setAxes(next);
	};

	return (
		<div className="glass pf-band pf-fitmap" aria-labelledby={`${id}-title`}>
			<div className="pf-band__head">
				<b id={`${id}-title`}>{t("fitmap.title")}</b>
				{gpu && cap ? <span>{t("fitmap.on_gpu", { gpu, cap: fmt(cap, "bytes") })}</span> : null}
				<span className="pf-fitmap__axes">
					<label>
						{t("fitmap.across")}
						<select value={axes[0]} onChange={(e) => pick(0, e.currentTarget.value)}>
							{options.map((k) => (
								<option key={k.id} value={k.id}>
									{t(k.label_key)}
								</option>
							))}
						</select>
					</label>
					<label>
						{t("fitmap.up")}
						<select value={axes[1]} onChange={(e) => pick(1, e.currentTarget.value)}>
							{options.map((k) => (
								<option key={k.id} value={k.id}>
									{t(k.label_key)}
								</option>
							))}
						</select>
					</label>
				</span>
			</div>
			<p className="pf-dim pf-fitmap__help">{t("fitmap.help")}</p>
			<div className="pf-fitmap__frame" style={{ gridTemplateColumns: `auto repeat(${cols}, minmax(34px, 1fr))` }}>
				{rowsTopDown.map((ri, r) => {
					const row = data.rows[ri]!;
					return [
						<span key={`y${ri}`} className="pf-fitmap__tick pf-fitmap__tick--y" style={{ gridRow: r + 1, gridColumn: 1 }}>
							{valueText(data.y, data.ys[ri])}
						</span>,
						...row.map((c, xi) => (
							<button
								key={`${ri}-${xi}`}
								type="button"
								className={`pf-fitcell pf-fitcell--${c.status}${c.here ? " is-here" : ""}`}
								style={{ gridRow: r + 1, gridColumn: xi + 2 }}
								aria-label={cellLabel(data, c)}
								aria-pressed={c.here}
								title={cellLabel(data, c)}
								onClick={() => {
									actions.setKnob(data.x.id, c.x, "user");
									actions.setKnob(data.y.id, c.y, "user");
								}}
							>
								{c.status === "abstain" ? null : <FitMark fit={c.status} label="" />}
							</button>
						)),
					];
				})}
				{data.xs.map((x, xi) => (
					<span key={String(x)} className="pf-fitmap__tick" style={{ gridRow: rows + 1, gridColumn: xi + 2 }}>
						{valueText(data.x, x)}
					</span>
				))}
				<span className="pf-fitmap__xlabel" style={{ gridRow: rows + 2, gridColumn: `2 / span ${cols}` }}>
					{t(data.x.label_key)}
				</span>
				<span className="pf-fitmap__ylabel" style={{ gridRow: rows + 2, gridColumn: 1 }}>
					{t("fitmap.y_up", { knob: t(data.y.label_key) })}
				</span>
				<svg
					className="pf-fitmap__lines"
					style={{ gridColumn: `2 / span ${cols}`, gridRow: `1 / span ${rows}` }}
					viewBox={`0 0 ${cols} ${rows}`}
					preserveAspectRatio="none"
					aria-hidden="true"
				>
					<path className="pf-fitmap__line pf-fitmap__line--fits" d={boundaryPath(data.frontier.fits, rowsTopDown)} vectorEffect="non-scaling-stroke" />
					<path className="pf-fitmap__line pf-fitmap__line--fails" d={boundaryPath(data.frontier.fails, rowsTopDown)} vectorEffect="non-scaling-stroke" />
				</svg>
			</div>
			<ul className="pf-legend">
				<li>
					<FitMark fit="fits" />
				</li>
				<li>
					<FitMark fit="tight" />
				</li>
				<li>
					<FitMark fit="fails" />
				</li>
				<li>
					<i className="pf-fitcell pf-fitcell--abstain pf-fitmap__swatch" />
					{t("provenance.abstain")}
				</li>
				<li>
					<i className="pf-fitmap__key pf-fitmap__key--fits" />
					{t("fitmap.line.fits")}
				</li>
				<li>
					<i className="pf-fitmap__key pf-fitmap__key--fails" />
					{t("fitmap.line.fails")}
				</li>
			</ul>
		</div>
	);
}
