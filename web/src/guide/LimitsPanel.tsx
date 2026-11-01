/**
 * "Your limits": what the guide may never do on this job. Allowed GPUs, a budget, a deadline, a
 * GPU-hours quota and "don't change" per knob (a pin), plus the values turned down with "Not
 * this". Limits written in plain words go to Muse Spark (POST /v1/guide/limits) and come back as
 * chips; none applies until the visitor confirms it. Suggestions from the job's own check-in
 * (record.job.constraints) are offered the same way. Confirmed limits are hard limits for both
 * tours (guide/limits.ts) and for the server (routes/guide.py).
 */
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { fmt } from "../lib/fmt";
import { flag } from "../lib/flags";
import { hasKey, t } from "../lib/i18n";
import type { Knob, KnobValue, LoadedRecord } from "../workspace/record/types";
import { useWorkspace } from "../workspace/state";
import { valueText } from "./GuidePanel";
import { type Limits, loadLimits, NO_LIMITS, saveLimits } from "./limits";

export type Chip =
	| { kind: "allowed"; knob_id: string; values: KnobValue[] }
	| { kind: "keep"; knob_id: string }
	| { kind: "budget_usd"; value: number }
	| { kind: "deadline_s"; value: number }
	| { kind: "gpu_hours"; value: number };

const GPU_KNOBS = ["gpu_type", "gpu"];

export function gpuKnob(record: LoadedRecord): Knob | undefined {
	return record.knobs.find((k) => GPU_KNOBS.includes(k.id) && k.kind === "categorical");
}

/** What the job's own check-in already says (budget, GPUs, deadline), offered as chips to confirm. */
export function fromCheckIn(record: LoadedRecord, now = Date.now()): Chip[] {
	const c = record.job.constraints;
	const out: Chip[] = [];
	const gpu = gpuKnob(record);
	const types = c?.gpu_types?.filter((g) => gpu?.domain?.values?.includes(g));
	if (gpu && types?.length && types.length < (gpu.domain?.values?.length ?? 0)) out.push({ kind: "allowed", knob_id: gpu.id, values: types });
	if (c?.budget_usd && c.budget_usd > 0) out.push({ kind: "budget_usd", value: c.budget_usd });
	const due = c?.deadline ? Date.parse(c.deadline) : Number.NaN;
	// A deadline already past says nothing about this run's length, so it isn't offered.
	if (Number.isFinite(due) && due > now) out.push({ kind: "deadline_s", value: Math.round((due - now) / 1000) });
	return out;
}

export function applyChip(limits: Limits, chip: Chip): Limits {
	switch (chip.kind) {
		case "allowed":
			return { ...limits, allowed: { ...limits.allowed, [chip.knob_id]: chip.values } };
		case "budget_usd":
			return { ...limits, budgetUsd: chip.value };
		case "deadline_s":
			return { ...limits, deadlineS: chip.value };
		case "gpu_hours":
			return { ...limits, gpuHours: chip.value };
		case "keep":
			return limits.frozen.includes(chip.knob_id) ? limits : { ...limits, frozen: [...limits.frozen, chip.knob_id] };
	}
}

function knobName(record: LoadedRecord, id: string): string {
	const k = record.knobs.find((x) => x.id === id);
	return k && hasKey(k.label_key) ? t(k.label_key) : id;
}

export function chipText(record: LoadedRecord, chip: Chip): string {
	const knob = "knob_id" in chip ? record.knobs.find((k) => k.id === chip.knob_id) : undefined;
	switch (chip.kind) {
		case "allowed":
			return t("limits.chip.allowed", { knob: knobName(record, chip.knob_id), values: chip.values.map((v) => valueText(knob, v)).join(", ") });
		case "keep":
			return t("limits.chip.keep", { knob: knobName(record, chip.knob_id) });
		case "budget_usd":
			return t("limits.chip.budget", { usd: fmt(chip.value, "usd") });
		case "deadline_s":
			return t("limits.chip.deadline", { time: fmt(chip.value, "s_short") });
		case "gpu_hours":
			return t("limits.chip.gpu_hours", { hours: fmt(chip.value, "count") });
	}
}

function NumberLimit({ id, label, value, unit, onChange }: { id: string; label: string; value: number | null; unit: string; onChange: (v: number | null) => void }) {
	const [draft, setDraft] = useState<string>(value === null ? "" : String(value));
	const commit = () => {
		const v = Number(draft);
		// Source: derived, the server takes limits up to 1e12; anything past a billion is a typo.
		onChange(draft.trim() === "" || !Number.isFinite(v) || v <= 0 || v > 1e9 ? null : v);
	};
	return (
		<label className="pf-limits__num" htmlFor={id}>
			<span>{label}</span>
			<span className="pf-limits__field">
				<input
					id={id}
					type="number"
					inputMode="decimal"
					min={0}
					step="any"
					value={draft}
					placeholder={t("limits.none")}
					onChange={(e) => setDraft(e.currentTarget.value)}
					onBlur={commit}
					onKeyDown={(e) => {
						if (e.key === "Enter") commit();
					}}
				/>
				<small>{unit}</small>
			</span>
		</label>
	);
}

export function LimitsPanel() {
	const { record, recordRef, state, actions } = useWorkspace();
	const uid = useId();
	const limits = state.limits;
	const gpu = gpuKnob(record);
	const [text, setText] = useState("");
	const [reading, setReading] = useState(false);
	const [proposed, setProposed] = useState<Chip[]>(() => fromCheckIn(record));
	const [source, setSource] = useState<"checkin" | "words">("checkin");
	const [notUnderstood, setNotUnderstood] = useState<string[]>([]);
	const [error, setError] = useState<string | null>(null);
	const live = flag("c.guide.live");

	// The server keeps this session's limits (the fix route reads only its copy): load them once,
	// so a reload keeps them, then save every change a moment after it settles.
	const loaded = useRef(false);
	const current = useRef(limits);
	current.current = limits;
	useEffect(() => {
		let gone = false;
		void loadLimits(recordRef).then((held) => {
			// The server's copy only fills an untouched panel: an edit made while it loaded stands.
			if (!gone && held && current.current === NO_LIMITS) actions.setLimits(held);
			loaded.current = true;
		});
		return () => {
			gone = true;
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [recordRef]);
	useEffect(() => {
		if (!loaded.current && limits === NO_LIMITS) return;
		const id = setTimeout(() => void saveLimits(recordRef, limits), 250);
		return () => clearTimeout(id);
	}, [recordRef, limits]);

	const proposedRef = useRef<HTMLDivElement>(null);
	const confirm = (chip: Chip) => {
		actions.setLimits(applyChip(limits, chip));
		setProposed((all) => all.filter((c) => c !== chip));
		proposedRef.current?.focus(); // the confirmed chip's button is gone: keep focus in the group
	};
	const confirmAll = () => {
		actions.setLimits(proposed.reduce(applyChip, limits));
		setProposed([]);
		summaryRef.current?.focus();
	};
	const summaryRef = useRef<HTMLElement>(null);

	const read = async () => {
		const words = text.trim();
		if (!words || reading) return;
		setReading(true);
		setError(null);
		try {
			const res = await fetch("/v1/guide/limits", {
				method: "POST",
				credentials: "same-origin",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ record: recordRef, text: words }),
			});
			const body = (await res.json().catch(() => null)) as { limits?: Chip[]; not_understood?: string[]; error?: { message?: string } } | null;
			if (!res.ok || !body?.limits) {
				setError(body?.error?.message ?? t("limits.read.failed"));
				return;
			}
			setSource("words");
			setProposed(body.limits);
			setNotUnderstood(body.not_understood ?? []);
			if (!body.limits.length && !(body.not_understood ?? []).length) setError(t("limits.read.nothing"));
		} catch {
			setError(t("limits.read.failed"));
		} finally {
			setReading(false);
		}
	};

	/** A number the visitor typed replaces any suggestion of the same kind still waiting. */
	const setNumber = (kind: "budget_usd" | "deadline_s" | "gpu_hours", v: number | null) => {
		setProposed((all) => all.filter((c) => c.kind !== kind));
		actions.setLimits({ ...limits, ...(kind === "budget_usd" ? { budgetUsd: v } : kind === "deadline_s" ? { deadlineS: v } : { gpuHours: v }) });
	};

	const allowedGpus = gpu ? (limits.allowed[gpu.id] ?? []) : [];
	const toggleGpu = (v: KnobValue) => {
		if (!gpu) return;
		setProposed((all) => all.filter((c) => !(c.kind === "allowed" && c.knob_id === gpu.id)));
		const on = allowedGpus.includes(v);
		const next = on ? allowedGpus.filter((x) => x !== v) : [...allowedGpus, v];
		const allowed = { ...limits.allowed };
		if (next.length) allowed[gpu.id] = next;
		else delete allowed[gpu.id];
		actions.setLimits({ ...limits, allowed });
	};
	const restore = (knobId: string, value: KnobValue) => {
		const rest = (limits.excluded[knobId] ?? []).filter((v) => v !== value);
		const excluded = { ...limits.excluded };
		if (rest.length) excluded[knobId] = rest;
		else delete excluded[knobId];
		// The knob comes back to the tours once nothing on it is ruled out any more.
		actions.setLimits({ ...limits, excluded, declined: rest.length ? limits.declined : limits.declined.filter((k) => k !== knobId) });
	};
	const toggleFrozen = (knobId: string) => {
		const on = limits.frozen.includes(knobId);
		actions.setLimits({ ...limits, frozen: on ? limits.frozen.filter((k) => k !== knobId) : [...limits.frozen, knobId] });
	};
	const excluded = Object.entries(limits.excluded).flatMap(([k, vs]) => vs.map((v) => [k, v] as const));
	const count = useMemo(
		() =>
			Object.keys(limits.allowed).length +
			excluded.length +
			limits.frozen.length +
			(limits.budgetUsd !== null ? 1 : 0) +
			(limits.deadlineS !== null ? 1 : 0) +
			(limits.gpuHours !== null ? 1 : 0),
		[limits, excluded.length],
	);

	return (
		<details className="pf-limits">
			<summary ref={summaryRef}>
				<b>{t("limits.title")}</b>
				<span className="pf-dim">{count ? t("limits.count", { n: fmt(count, "count") }) : t("limits.none_set")}</span>
			</summary>
			<p className="pf-dim">{t("limits.intro")}</p>

			{live ? (
				<form
					className="pf-limits__words"
					onSubmit={(e) => {
						e.preventDefault();
						void read();
					}}
				>
					<label htmlFor={`${uid}-words`}>{t("limits.words.label")}</label>
					<textarea id={`${uid}-words`} rows={2} maxLength={300} value={text} placeholder={t("limits.words.placeholder")} onChange={(e) => setText(e.currentTarget.value)} />
					<p className="pf-guide__fine">{t("limits.words.datause")}</p>
					<button type="submit" className="pf-btn pf-btn--small" disabled={reading || !text.trim()}>
						{reading ? t("limits.words.reading") : t("limits.words.read")}
					</button>
				</form>
			) : null}
			{error ? (
				<p className="pf-limits__error" role="alert">
					{error}
				</p>
			) : null}

			{proposed.length ? (
				<div ref={proposedRef} tabIndex={-1} className="pf-limits__proposed" role="group" aria-label={t("limits.proposed")}>
					<p className="pf-dim">{source === "words" ? t("limits.proposed.words") : t("limits.proposed.checkin")}</p>
					<ul className="pf-lchips">
						{proposed.map((c, i) => (
							<li key={`${c.kind}-${i}`} className="pf-lchip is-proposed">
								<span>{chipText(record, c)}</span>
								<button type="button" className="pf-lchip__btn" onClick={() => confirm(c)} aria-label={t("limits.confirm_one", { what: chipText(record, c) })}>
									{t("limits.confirm")}
								</button>
								<button type="button" className="pf-lchip__btn pf-lchip__btn--quiet" onClick={() => setProposed((all) => all.filter((x) => x !== c))} aria-label={t("limits.dismiss_one", { what: chipText(record, c) })}>
									×
								</button>
							</li>
						))}
					</ul>
					{proposed.length > 1 ? (
						<button type="button" className="pf-btn pf-btn--small pf-btn--primary" onClick={confirmAll}>
							{t("limits.confirm_all")}
						</button>
					) : null}
				</div>
			) : null}
			{notUnderstood.length ? <p className="pf-dim">{t("limits.not_understood", { what: notUnderstood.map((s) => `"${s}"`).join(", ") })}</p> : null}

			{gpu ? (
				<fieldset className="pf-limits__set">
					<legend>{t("limits.gpus")}</legend>
					<div className="pf-lchips">
						{(gpu.domain?.values ?? []).map((v) => (
							<button key={String(v)} type="button" className={`pf-lchip pf-lchip--toggle${allowedGpus.includes(v) ? " is-on" : ""}`} aria-pressed={allowedGpus.includes(v)} onClick={() => toggleGpu(v)}>
								{valueText(gpu, v)}
							</button>
						))}
					</div>
					<p className="pf-dim">{allowedGpus.length ? t("limits.gpus.some") : t("limits.gpus.any")}</p>
				</fieldset>
			) : null}

			<div className="pf-limits__nums">
				<NumberLimit key={`b${limits.budgetUsd}`} id={`${uid}-budget`} label={t("limits.budget")} unit="US$" value={limits.budgetUsd} onChange={(v) => setNumber("budget_usd", v)} />
				<NumberLimit
					key={`d${limits.deadlineS}`}
					id={`${uid}-deadline`}
					label={t("limits.deadline")}
					unit={t("limits.hours")}
					value={limits.deadlineS === null ? null : Math.round((limits.deadlineS / 3600) * 100) / 100}
					onChange={(v) => setNumber("deadline_s", v === null ? null : Math.min(v * 3600, 1e9))}
				/>
				<NumberLimit key={`q${limits.gpuHours}`} id={`${uid}-quota`} label={t("limits.quota")} unit={t("limits.gpu_hours")} value={limits.gpuHours} onChange={(v) => setNumber("gpu_hours", v)} />
			</div>

			<fieldset className="pf-limits__set">
				<legend>{t("limits.keep")}</legend>
				<div className="pf-lchips">
					{record.knobs.map((k) => {
						const on = limits.frozen.includes(k.id);
						return (
							<button key={k.id} type="button" className={`pf-lchip pf-lchip--toggle${on ? " is-on" : ""}`} aria-pressed={on} onClick={() => toggleFrozen(k.id)}>
								{t(k.label_key)}
								{on ? <small> · {valueText(k, state.knobs[k.id])}</small> : null}
							</button>
						);
					})}
				</div>
			</fieldset>

			{excluded.length ? (
				<fieldset className="pf-limits__set">
					<legend>{t("limits.ruled_out")}</legend>
					<ul className="pf-lchips">
						{excluded.map(([k, v]) => (
							<li key={`${k}-${String(v)}`} className="pf-lchip">
								<span>
									{knobName(record, k)} {valueText(record.knobs.find((x) => x.id === k), v)}
								</span>
								<button type="button" className="pf-lchip__btn pf-lchip__btn--quiet" onClick={() => restore(k, v)} aria-label={t("limits.restore", { what: `${knobName(record, k)} ${valueText(record.knobs.find((x) => x.id === k), v)}` })}>
									×
								</button>
							</li>
						))}
					</ul>
				</fieldset>
			) : null}
		</details>
	);
}
