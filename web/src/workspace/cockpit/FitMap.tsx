/**
 * The fit map, on the stage when the visitor switches from Map to Fit: two knobs against each
 * other, everything else as on the dials. Each square is that setting: its fill is the predicted
 * peak memory as a share of the GPU's, its colour and mark the chance of running out of memory
 * (fits, tight, fails, or hatched where the prediction isn't trusted; shape and word, never colour
 * alone), and the edges between squares are placed from memory interpolated between them
 * (record/fitmap.ts).
 *
 * It's a place to move the run, not a table to read: the run's own square is a puck that glides
 * when any knob moves and can be dragged across the map; pointing at a square previews its
 * memory, time and cost in the readout before anything changes; pressing it flies there; the GPU
 * can be switched on the map itself so the edges move under the visitor's eye. Arrow keys move
 * between squares. Behind the c.fit_map flag.
 */
import { type CSSProperties, useDeferredValue, useId, useMemo, useRef, useState } from "react";
import { valueText } from "../../guide/GuidePanel";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { Info } from "../components/Info";
import { FitMark } from "../components/StatusMark";
import { type CellStatus, defaultAxes, type FitCell, type FitMapData, fitMap, fitMapKnobs } from "../record/fitmap";
import { knobChoices, targetFor } from "../record/model";
import { chosenGpu, FAILS_AT, FITS_BELOW, GPU_KNOBS } from "../record/terms";
import { useWorkspace } from "../state";

/** Grid gap between squares, px (styles.css .pf-fitmap__frame); the puck steps by a square plus this. */
const GAP_PX = 4;

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

/** Peak as a share of the GPU's memory: the square's fill. Over 1 is drawn full, with an over mark. */
function share(c: FitCell): number | null {
	return c.peak !== null && c.capacity ? c.peak / c.capacity : null;
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

/** Where a square sits for the puck: column and row from the top, as CSS custom properties. */
function puckStyle(xi: number, r: number, cols: number, rows: number): CSSProperties {
	return {
		width: `calc((100% - ${(cols - 1) * GAP_PX}px) / ${cols})`,
		height: `calc((100% - ${(rows - 1) * GAP_PX}px) / ${rows})`,
		transform: `translate(calc(${xi} * (100% + ${GAP_PX}px)), calc(${r} * (100% + ${GAP_PX}px)))`,
	};
}

export function FitMap() {
	const { record, provider, state, estimates, actions } = useWorkspace();
	// Off the critical path: the map catches up after the dials have answered a knob change.
	const knobs = useDeferredValue(state.knobs);
	const id = useId();
	const options = useMemo(() => fitMapKnobs(record), [record]);
	const initial = useMemo(() => defaultAxes(record, provider, knobs), [record, provider]); // eslint-disable-line react-hooks/exhaustive-deps
	const [axes, setAxes] = useState<[string, string] | null>(initial);
	const data = useMemo(() => (axes ? fitMap(record, provider, knobs, axes[0], axes[1]) : null), [record, provider, knobs, axes]);
	// The square being pointed at or focused: previewed in the readout, nothing changes yet.
	const [peek, setPeek] = useState<{ xi: number; ri: number } | null>(null);
	const drag = useRef<{ pointer: number; last: string } | null>(null);
	const frame = useRef<HTMLDivElement>(null);

	if (!axes || !data || options.length < 2) {
		return (
			<div className="pf-fit pf-fit--empty" role="status">
				{t("fitmap.none")}
			</div>
		);
	}
	const gpu = chosenGpu(record, knobs);
	const gpuKnob = record.knobs.find((k) => (GPU_KNOBS as readonly string[]).includes(k.id));
	const gpuChoices = gpuKnob ? knobChoices(record, gpuKnob).filter((v): v is string => typeof v === "string") : [];
	const cap = data.rows[0]?.[0]?.capacity ?? null;
	const rowsTopDown = data.ys.map((_, i) => data.ys.length - 1 - i);
	const cols = data.xs.length;
	const rows = data.ys.length;
	let here: { xi: number; r: number } | null = null;
	rowsTopDown.forEach((ri, r) => data.rows[ri]!.forEach((c, xi) => (c.here ? (here = { xi, r }) : null)));
	const hereAt = here as { xi: number; r: number } | null;

	const pick = (which: 0 | 1, value: string) => {
		const other = axes[which === 0 ? 1 : 0];
		setAxes(which === 0 ? [value, value === other ? axes[0] : other] : [value === other ? axes[1] : other, value]);
	};
	const fly = (c: FitCell) => {
		actions.setKnob(data.x.id, c.x, "user");
		actions.setKnob(data.y.id, c.y, "user");
	};
	const cellAt = (clientX: number, clientY: number): { xi: number; ri: number } | null => {
		const el = document.elementFromPoint(clientX, clientY)?.closest<HTMLElement>("[data-xi]");
		if (!el || !frame.current?.contains(el)) return null;
		return { xi: Number(el.dataset.xi), ri: Number(el.dataset.ri) };
	};
	const focusCell = (xi: number, r: number) => {
		const ri = rowsTopDown[r];
		if (ri === undefined || xi < 0 || xi >= cols) return;
		frame.current?.querySelector<HTMLButtonElement>(`[data-xi="${xi}"][data-ri="${ri}"]`)?.focus();
	};
	const axisSelect = (which: 0 | 1) => (
		<select aria-label={which === 0 ? t("fitmap.across") : t("fitmap.up")} value={axes[which]} onChange={(e) => pick(which, e.currentTarget.value)}>
			{options.map((k) => (
				<option key={k.id} value={k.id}>
					{t(k.label_key)}
				</option>
			))}
		</select>
	);

	// The readout: the square pointed at, else the run's own setting.
	const shown = peek ? data.rows[peek.ri]?.[peek.xi] : hereAt ? data.rows[rowsTopDown[hereAt.r]!]?.[hereAt.xi] : undefined;
	const tT = targetFor(record, "time");
	const cT = targetFor(record, "cost");
	const nowTime = tT ? estimates[tT.id] : undefined;
	const nowCost = cT ? estimates[cT.id] : undefined;
	const delta = (v: number | null, now: typeof nowTime, unit: string) => {
		if (v === null || !shown || shown.here || !now || now.point === null) return null;
		const d = v - now.point;
		if (Math.abs(d) < Math.abs(now.point) * 0.005) return null;
		return `${d > 0 ? "+" : "−"}${fmt(Math.abs(d), unit)}`;
	};

	return (
		<div className="pf-fit" aria-labelledby={`${id}-t`}>
			<div className="pf-fit__head">
				<b id={`${id}-t`} className="pf-visually-hidden">
					{t("fitmap.title")}
				</b>
				<span className="pf-fit__axis" title={t("fitmap.across")}>
					<span aria-hidden="true">↔</span>
					{axisSelect(0)}
				</span>
				<span className="pf-fit__axis" title={t("fitmap.up")}>
					<span aria-hidden="true">↕</span>
					{axisSelect(1)}
				</span>
				{gpuKnob && gpuChoices.length > 1 && gpu ? (
					<label className="pf-fit__gpu">
						<span>{t("fitmap.gpu")}</span>
						<select value={gpu} onChange={(e) => actions.setKnob(gpuKnob.id, e.currentTarget.value, "user")}>
							{gpuChoices.map((g) => (
								<option key={g} value={g}>
									{g}
								</option>
							))}
						</select>
						{cap ? <span className="pf-fit__cap">{fmt(cap, "bytes")}</span> : null}
					</label>
				) : gpu && cap ? (
					<span className="pf-tag pf-fit__gpu">{t("fitmap.on_gpu", { gpu, cap: fmt(cap, "bytes") })}</span>
				) : null}
				<Info label={t("fitmap.title")} align="end">
					{t("fitmap.help", { fits: fmt(FITS_BELOW, "probability"), fails: fmt(FAILS_AT, "probability") })}
				</Info>
			</div>
			<p className={`pf-fit__readout${peek ? " is-peek" : ""}`}>
				{shown ? (
					<>
						<span className="pf-fit__where">
							{peek && !shown.here ? null : <span className="pf-fit__herechip">{t("fitmap.here_short")}</span>}
							{t(data.x.label_key)} <b>{valueText(data.x, shown.x)}</b>, {t(data.y.label_key)} <b>{valueText(data.y, shown.y)}</b>
						</span>
						{shown.status === "abstain" ? (
							<span className="pf-dim">{t("provenance.abstain")}</span>
						) : (
							<>
								<FitMark fit={shown.status} />
								{shown.peak !== null && shown.capacity ? (
									<span>{t("fitmap.readout.peak", { peak: fmt(shown.peak, "bytes"), share: fmt(share(shown)!, "fraction") })}</span>
								) : null}
							</>
						)}
						{shown.time !== null ? (
							<span>
								{fmt(shown.time, tT?.unit ?? "s")}
								{delta(shown.time, nowTime, tT?.unit ?? "s") ? <small> {delta(shown.time, nowTime, tT?.unit ?? "s")}</small> : null}
							</span>
						) : null}
						{shown.cost !== null ? (
							<span>
								{fmt(shown.cost, "usd")}
								{delta(shown.cost, nowCost, "usd") ? <small> {delta(shown.cost, nowCost, "usd")}</small> : null}
							</span>
						) : null}
						<span className="pf-fit__hint">{peek && !shown.here ? t("fitmap.hint.fly") : t("fitmap.hint.drag")}</span>
					</>
				) : (
					<span className="pf-dim">{t("fitmap.hint.point")}</span>
				)}
			</p>
			<div className="pf-fit__body">
				<div
					ref={frame}
					className="pf-fitmap__frame"
					style={{ gridTemplateColumns: `auto repeat(${cols}, minmax(0, 1fr))`, gap: GAP_PX, "--rows": rows } as CSSProperties}
					onPointerLeave={() => (drag.current ? null : setPeek(null))}
					onPointerMove={(e) => {
						const at = cellAt(e.clientX, e.clientY);
						if (!drag.current) {
							if (at && (at.xi !== peek?.xi || at.ri !== peek?.ri)) setPeek(at);
							return;
						}
						if (!at) return;
						const key = `${at.xi}:${at.ri}`;
						if (key === drag.current.last) return;
						drag.current.last = key;
						const c = data.rows[at.ri]?.[at.xi];
						if (c) fly(c);
						setPeek(null);
					}}
					onPointerUp={(e) => {
						if (drag.current?.pointer === e.pointerId) drag.current = null;
					}}
					onPointerCancel={() => {
						drag.current = null;
					}}
					onKeyDown={(e) => {
						const el = (e.target as HTMLElement).closest<HTMLElement>("[data-xi]");
						if (!el) return;
						const xi = Number(el.dataset.xi);
						const r = rowsTopDown.indexOf(Number(el.dataset.ri));
						const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
						if (!step) return;
						e.preventDefault();
						focusCell(xi + step[0]!, r + step[1]!);
					}}
				>
					{rowsTopDown.map((ri, r) => {
						const row = data.rows[ri]!;
						return [
							<span key={`y${ri}`} className="pf-fitmap__tick pf-fitmap__tick--y" style={{ gridRow: r + 1, gridColumn: 1 }}>
								{valueText(data.y, data.ys[ri])}
							</span>,
							...row.map((c, xi) => {
								const f = share(c);
								return (
									<button
										key={`${ri}-${xi}`}
										type="button"
										data-xi={xi}
										data-ri={ri}
										className={`pf-fitcell pf-fitcell--${c.status}${c.here ? " is-here" : ""}${f !== null && f > 1 ? " is-over" : ""}`}
										style={{ gridRow: r + 1, gridColumn: xi + 2, "--fill": f === null ? 0 : Math.min(1, f) } as CSSProperties}
										aria-label={cellLabel(data, c)}
										aria-pressed={c.here}
										onFocus={() => setPeek({ xi, ri })}
										onBlur={() => setPeek(null)}
										onPointerDown={(e) => {
											if (e.button !== 0) return;
											// The run's own square is the puck: dragging from it moves the run across the map.
											if (c.here) {
												drag.current = { pointer: e.pointerId, last: `${xi}:${ri}` };
												frame.current?.setPointerCapture(e.pointerId);
											}
										}}
										onClick={() => (c.here ? null : fly(c))}
									>
										<span className="pf-fitcell__fill" aria-hidden="true" />
										{c.status === "abstain" ? null : <FitMark fit={c.status} label="" />}
										{f !== null ? (
											<span className="pf-fitcell__pct" aria-hidden="true">
												{fmt(f, "fraction")}
											</span>
										) : null}
									</button>
								);
							}),
						];
					})}
					{data.xs.map((x, xi) => (
						<span key={String(x)} className="pf-fitmap__tick" style={{ gridRow: rows + 1, gridColumn: xi + 2 }}>
							{valueText(data.x, x)}
						</span>
					))}
					<div className="pf-fitmap__overlay" style={{ gridColumn: `2 / span ${cols}`, gridRow: `1 / span ${rows}` }} aria-hidden="true">
						<svg className="pf-fitmap__lines" viewBox={`0 0 ${cols} ${rows}`} preserveAspectRatio="none">
							<path className="pf-fitmap__line pf-fitmap__line--fits" d={boundaryPath(data.frontier.fits, rowsTopDown) || "M0 0"} vectorEffect="non-scaling-stroke" />
							<path className="pf-fitmap__line pf-fitmap__line--fails" d={boundaryPath(data.frontier.fails, rowsTopDown) || "M0 0"} vectorEffect="non-scaling-stroke" />
						</svg>
						{peek && !(hereAt && rowsTopDown[hereAt.r] === peek.ri && hereAt.xi === peek.xi) ? (
							<span className="pf-fitmap__ghost" style={puckStyle(peek.xi, rowsTopDown.indexOf(peek.ri), cols, rows)} />
						) : null}
						{hereAt ? <span className="pf-fitmap__puck" style={puckStyle(hereAt.xi, hereAt.r, cols, rows)} /> : null}
					</div>
				</div>
			</div>
			<ul className="pf-fit__key" aria-label={t("fitmap.key")}>
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
					<i className="pf-fitmap__key pf-fitmap__key--fill" />
					{t("fitmap.key.fill")}
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
