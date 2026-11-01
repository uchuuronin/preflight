/**
 * The run cost as quantity x rate = total, under the status line: "4.2 to 5.1 GPU-hours x
 * US$0.80/h = US$3.40 to US$4.10", or tokens x a $/M rate for an API run. The rate is a button: it
 * opens a small box to type the visitor's own rate ($/GPU-hour, or $/M input and $/M output
 * tokens), with the table's rate, source and date in small text. $0 is valid (an assigned
 * university GPU). Saved, every dollar readout moves at once (workspace/rate.ts); a "your rate"
 * tag shows it's the visitor's, with a reset. Kept per job in this browser and carried in the
 * export plan.
 */
import { useEffect, useId, useRef, useState } from "react";
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { target, targetFor } from "../record/model";
import { appliesHere, chosenModel, tableRate, tokenMix } from "../rate";
import { useWorkspace } from "../state";
import { trusted } from "../status";

/** "Jul 2026" from the table's "2026-07" (or a full date). */
function asOf(s: string): string {
	const d = new Date(/^\d{4}-\d{2}$/.test(s) ? `${s}-01T00:00:00Z` : s);
	return Number.isFinite(d.getTime()) ? new Intl.DateTimeFormat("en-GB", { month: "short", year: "numeric", timeZone: "UTC" }).format(d) : s;
}

export function CostLine() {
	const { record, state, estimates, tableEstimates, actions } = useWorkspace();
	const [open, setOpen] = useState(false);
	const id = useId();
	const box = useRef<HTMLDivElement>(null);
	const trigger = useRef<HTMLButtonElement>(null);
	const tr = tableRate(record, state.knobs);
	const costT = targetFor(record, "cost");
	const cost = costT ? estimates[costT.id] : undefined;
	const tableCost = costT ? tableEstimates[costT.id] : undefined;
	const o = state.rate;
	const mine = appliesHere(record, state.knobs, o);

	useEffect(() => {
		if (!open) return;
		const onKey = (e: KeyboardEvent) => {
			if (e.key === "Escape") {
				setOpen(false);
				trigger.current?.focus();
			}
		};
		const onDown = (e: PointerEvent) => {
			if (box.current && !box.current.contains(e.target as Node) && !trigger.current?.contains(e.target as Node)) setOpen(false);
		};
		window.addEventListener("keydown", onKey);
		window.addEventListener("pointerdown", onDown);
		return () => {
			window.removeEventListener("keydown", onKey);
			window.removeEventListener("pointerdown", onDown);
		};
	}, [open]);

	if (!tr || !trusted(cost) || !trusted(tableCost)) return null;

	// The quantity is the engine's own: its dollars at the table's rate, divided by that rate.
	let quantity: string;
	let rateText: string;
	const num = (v: number, unit: "gpu_h" | "tokens") => fmt(v, unit).replace(unit === "gpu_h" ? / GPU-hours$/ : / tokens$/, "");
	if (tr.kind === "gpu") {
		const h = (v: number | null) => (v ?? tableCost.point ?? 0) / tr.usdPerHour;
		quantity = t("cost.line.gpu_hours", { lo: num(h(tableCost.lo), "gpu_h"), hi: num(h(tableCost.hi), "gpu_h") });
		const r = mine && o?.kind === "gpu" ? o.usdPerHour : tr.usdPerHour;
		rateText = t("cost.line.per_hour", { rate: fmt(r, "usd_small") });
	} else {
		const tokT = target(record, "tokens.total");
		const tok = tokT ? tableEstimates[tokT.id] : undefined;
		quantity = trusted(tok) ? t("cost.line.tokens", { lo: num(tok.lo ?? tok.point, "tokens"), hi: num(tok.hi ?? tok.point, "tokens") }) : t("cost.line.tokens_unknown");
		const mix = tokenMix(record)!;
		const perM = mine && o?.kind === "tokens" ? (mix.input * o.inputPerM + mix.output * o.outputPerM) / (mix.input + mix.output) : tr.usdPerToken * 1e6;
		rateText = t("cost.line.per_m", { rate: fmt(perM, "usd_small") });
	}
	const total = cost.lo !== cost.hi ? fmtRange(cost.lo ?? cost.point, cost.hi ?? cost.point, "usd") : fmt(cost.point, "usd");

	return (
		<div className="pf-costline">
			<span className="pf-costline__label">{t("cost.line.label")}</span>{" "}
			<span>{quantity}</span> <span aria-hidden="true">×</span>{" "}
			<button ref={trigger} type="button" className="pf-costline__rate" aria-expanded={open} aria-controls={id} aria-label={t("cost.line.edit", { rate: rateText })} onClick={() => setOpen(!open)}>
				{rateText}
			</button>
			{mine ? (
				<span className="pf-tag pf-costline__mine">
					{t("cost.line.yours")}
					<button type="button" className="pf-costline__reset" aria-label={t("cost.line.reset")} title={t("cost.line.reset")} onClick={() => actions.setRate(null)}>
						×
					</button>
				</span>
			) : null}{" "}
			<span aria-hidden="true">=</span> <b>{total}</b>
			{open ? <RateBox id={id} boxRef={box} onClose={() => setOpen(false)} /> : null}
		</div>
	);
}

function RateBox({ id, boxRef, onClose }: { id: string; boxRef: React.RefObject<HTMLDivElement | null>; onClose: () => void }) {
	const { record, state, actions } = useWorkspace();
	const tr = tableRate(record, state.knobs)!;
	const o = state.rate;
	const first = useRef<HTMLInputElement>(null);
	useEffect(() => first.current?.focus(), []);
	const gpuName = tr.kind === "gpu" ? tr.row.gpu : "";
	const model = chosenModel(record, state.knobs) ?? "";
	const [a, setA] = useState(() => String(o?.kind === "gpu" ? o.usdPerHour : o?.kind === "tokens" ? o.inputPerM : tr.kind === "gpu" ? tr.usdPerHour : tr.row.input_per_m));
	const [b, setB] = useState(() => String(o?.kind === "tokens" ? o.outputPerM : tr.kind === "tokens" ? tr.row.output_per_m : 0));
	const num = (s: string) => (s.trim() === "" ? Number.NaN : Number(s));
	const ok = Number.isFinite(num(a)) && num(a) >= 0 && (tr.kind === "gpu" || (Number.isFinite(num(b)) && num(b) >= 0));
	const save = () => {
		if (!ok) return;
		actions.setRate(tr.kind === "gpu" ? { kind: "gpu", gpu: gpuName, usdPerHour: num(a) } : { kind: "tokens", model, inputPerM: num(a), outputPerM: num(b) });
		onClose();
	};
	const src = tr.row;
	return (
		<div id={id} ref={boxRef} className="glass pf-ratebox" role="dialog" aria-label={t("cost.box.title")}>
			<form
				onSubmit={(e) => {
					e.preventDefault();
					save();
				}}
			>
				<b className="pf-ratebox__title">{tr.kind === "gpu" ? t("cost.box.title_gpu", { gpu: gpuName }) : t("cost.box.title_tokens", { model })}</b>
				{tr.kind === "gpu" ? (
					<label className="pf-ratebox__field">
						<span>{t("cost.box.per_hour")}</span>
						<input ref={first} type="number" inputMode="decimal" min={0} step="any" value={a} onChange={(e) => setA(e.currentTarget.value)} />
					</label>
				) : (
					<>
						<label className="pf-ratebox__field">
							<span>{t("cost.box.input_per_m")}</span>
							<input ref={first} type="number" inputMode="decimal" min={0} step="any" value={a} onChange={(e) => setA(e.currentTarget.value)} />
						</label>
						<label className="pf-ratebox__field">
							<span>{t("cost.box.output_per_m")}</span>
							<input type="number" inputMode="decimal" min={0} step="any" value={b} onChange={(e) => setB(e.currentTarget.value)} />
						</label>
					</>
				)}
				<p className="pf-ratebox__src">
					{tr.kind === "gpu"
						? t("cost.box.table_gpu", { rate: fmt(tr.usdPerHour, "usd_small"), provider: src.provider, source: src.source, date: asOf(src.as_of) })
						: t("cost.box.table_tokens", {
								input: fmt(tr.row.input_per_m, "usd_small"),
								output: fmt(tr.row.output_per_m, "usd_small"),
								source: src.source,
								date: asOf(src.as_of),
								share: tokenMix(record) ? fmt(tokenMix(record)!.input / (tokenMix(record)!.input + tokenMix(record)!.output), "probability") : t("value.none"),
							})}
				</p>
				{!ok ? <p className="pf-ratebox__err">{t("cost.box.invalid")}</p> : null}
				<div className="pf-ratebox__actions">
					<button type="submit" className="pf-btn pf-btn--primary pf-btn--small" disabled={!ok}>
						{t("cost.box.save")}
					</button>
					{o ? (
						<button
							type="button"
							className="pf-btn pf-btn--small"
							onClick={() => {
								actions.setRate(null);
								onClose();
							}}
						>
							{t("cost.line.reset")}
						</button>
					) : null}
				</div>
			</form>
		</div>
	);
}
