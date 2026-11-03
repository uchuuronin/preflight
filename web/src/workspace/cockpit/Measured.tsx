/**
 * What was measured: a capsule in the cockpit head ("Measured 3 · US$0.20") opens a drawer of
 * probe chips. A chip is a status mark, the settings it ran at, its peak and its cost; its
 * error and hint are on hover; pressing it flies the knobs there.
 */
import { useEffect, useId, useRef, useState } from "react";
import { valueText } from "../../guide/GuidePanel";
import { errorCopy } from "../../lib/errors";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { FitMark } from "../components/StatusMark";
import { isFixture } from "../record/model";
import type { KnobValue } from "../record/types";
import { useWorkspace } from "../state";
import type { Fit } from "../status";

const PROBE_FIT: Record<string, Fit> = {
	ok: "fits",
	cached: "fits",
	oom: "fails",
	error: "fails",
	timeout: "fails",
	cancelled: "unknown",
	queued: "unknown",
	running: "unknown",
};

export function Measured() {
	const { record, state, actions } = useWorkspace();
	const [open, setOpen] = useState(false);
	const id = useId();
	const panel = useRef<HTMLElement>(null);
	const opener = useRef<HTMLButtonElement>(null);
	useEffect(() => {
		if (!open) return;
		panel.current?.focus();
		const onKey = (e: KeyboardEvent) => {
			if (e.key === "Escape") {
				setOpen(false);
				opener.current?.focus();
			}
		};
		document.addEventListener("keydown", onKey);
		return () => document.removeEventListener("keydown", onKey);
	}, [open]);
	if (!record.probes.length) return null;
	const knobs = new Map(record.knobs.map((k) => [k.id, k]));
	const total = record.probes.reduce((a, p) => a + (p.cost?.usd ?? 0), 0);
	const failed = record.probes.filter((p) => PROBE_FIT[p.status] === "fails").length;
	// "ran out" when every failure is memory (the usual case); "failed" when any probe errored or timed out.
	const allOom = failed > 0 && record.probes.every((p) => PROBE_FIT[p.status] !== "fails" || p.status === "oom");
	// One column order for every chip: the knobs any probe set, in the record's order; a probe
	// holds a knob it didn't list at the record's current value.
	const cols = record.knobs.filter((k) => record.probes.some((p) => p.knobs && k.id in p.knobs));
	// A sample's probes are part of its illustrative fixture, not a measurement: say so, and leave
	// out the made-up probe cost (the cockpit head says "Illustrative numbers, not measured").
	const illustrative = isFixture(record);
	return (
		<>
			<button
				ref={opener}
				type="button"
				className={`pf-tag pf-tag--button${open ? " is-on" : ""}`}
				aria-expanded={open}
				aria-controls={id}
				onClick={() => setOpen(!open)}
			>
				{illustrative
					? t("measured.capsule_illustrative", { n: fmt(record.probes.length, "count") })
					: t("measured.capsule", { n: fmt(record.probes.length, "count"), cost: fmt(total, "usd") })}
				{failed ? <FitMark fit="fails" label={t(allOom ? (failed === 1 ? "measured.oom_one" : "measured.oom_many") : failed === 1 ? "measured.failed_one" : "measured.failed_many", { n: fmt(failed, "count") })} /> : null}
			</button>
			<aside
				id={id}
				ref={panel}
				tabIndex={-1}
				className={`glass pf-drawer${open ? " is-open" : ""}`}
				aria-label={t(illustrative ? "measured.title_illustrative" : "measured.title")}
				aria-hidden={!open}
				inert={!open}
			>
				<div className="pf-drawer__head">
					<b>{t(illustrative ? "measured.title_illustrative" : "measured.title")}</b>
					{illustrative ? null : <span className="pf-dim">{fmt(total, "usd")}</span>}
					<button type="button" className="pf-drawer__close" aria-label={t("measured.close")} onClick={() => setOpen(false)}>
						×
					</button>
				</div>
				<ul className="pf-chips">
					{record.probes.map((p) => {
						const fit = PROBE_FIT[p.status] ?? "unknown";
						const settings = Object.entries(p.knobs ?? {}).filter((e): e is [string, KnobValue] => e[1] !== undefined && e[1] !== null);
						const here = settings.length > 0 && settings.every(([k, v]) => state.knobs[k] === v);
						const peak = p.memory?.peak_allocated_bytes ?? p.memory?.peak_reserved_bytes ?? null;
						const err = fit === "fails" ? errorCopy(p.error_code ?? (p.status === "oom" ? "oom" : p.status === "timeout" ? "timeout" : null), p.error) : null;
						const where = settings.length
							? settings.map(([k, v]) => `${t(knobs.get(k)?.label_key ?? k)} ${valueText(knobs.get(k), v)}`).join(" · ")
							: t("probes.as_written");
						const title = [where, err?.message, err?.hint].filter(Boolean).join("\n");
						return (
							<li key={p.id}>
								<button
									type="button"
									className={`pf-chip pf-chip--${fit}${here ? " is-here" : ""}`}
									title={title}
									aria-label={`${t(`probes.status.${p.status}`)}: ${title}`}
									aria-pressed={here}
									disabled={!settings.length}
									onClick={() => {
										for (const [k, v] of settings) if (knobs.has(k)) actions.setKnob(k, v, "user");
									}}
								>
									<FitMark fit={fit} label={t(`probes.status.${p.status}`)} />
									<span className="pf-chip__where">
										{cols.length ? cols.map((k) => valueText(k, (p.knobs?.[k.id] as KnobValue | undefined) ?? k.current)).join(" · ") : t("measured.as_written")}
									</span>
									<span className="pf-chip__nums">
										{[peak ? fmt(peak, "bytes") : null, p.cost?.usd && !illustrative ? fmt(p.cost.usd, "usd") : null].filter(Boolean).join(" · ")}
									</span>
								</button>
							</li>
						);
					})}
				</ul>
				{cols.length ? <p className="pf-chips__key pf-dim">{cols.map((k) => t(k.label_key)).join(" · ")}</p> : null}
			</aside>
		</>
	);
}
