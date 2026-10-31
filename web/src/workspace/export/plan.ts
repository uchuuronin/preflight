/**
 * The plan an export carries: what the visitor settled on in the cockpit, in data, so D's
 * exports panel (config, patch, sbatch, CLI command, share link) formats it and computes
 * nothing. Every number here is the provider's estimate at the current setting, with its
 * status and interval as the dials show it; an abstaining readout has no numbers, so an
 * export can never print one the cockpit wouldn't.
 */
import type { CodeLocation, Estimate, KnobValue, Lens, LoadedRecord, Patch, Readout, Source } from "../record/types";
import type { EstimateStatus } from "../record/types";

export interface PlanChange {
	knobId: string;
	labelKey: string;
	unit: string | null;
	/** The record's setting, as the script ran in the probes. */
	from: KnobValue;
	to: KnobValue;
	/** True when the script must be edited, not just run with a different flag (A's script map). */
	needsCode: boolean;
	patchTemplate: string | null;
	/** Where the script sets it, from A's script map, when known. */
	location: CodeLocation | null;
}

export interface PlanReadout {
	targetId: string;
	readout: Readout;
	unit: string;
	status: EstimateStatus;
	/** Null when the readout abstains: nothing to quote. */
	point: number | null;
	lo: number | null;
	hi: number | null;
}

export interface ExportPlan {
	lens: Lens;
	title: string | null;
	source: Pick<Source, "kind" | "uri" | "commit" | "entrypoint" | "args">;
	/** The command line that runs the script, from A's script map, when known. */
	launch: string | null;
	gpu: { name: string | null; count: number };
	/** Every knob's value at the current setting. */
	config: Record<string, KnobValue>;
	/** Only the knobs moved from the record's setting, in the record's knob order. */
	changes: PlanChange[];
	readouts: PlanReadout[];
	/** The upper end of the run's predicted wall time in seconds, for a scheduler's time limit. Null unless the time readout is trusted. */
	wallTimeHiS: number | null;
	patches: Patch[];
	/** Every code change has a patch whose verify passed. False when any is missing or unverified. */
	codeChangesVerified: boolean;
	/** A link that opens this record read-only (share token or sample id), or null without one. */
	shareUrl: string | null;
}

const TRUSTED: ReadonlySet<EstimateStatus> = new Set(["measured", "predicted", "extrapolated"]);

function same(a: KnobValue, b: KnobValue): boolean {
	return typeof a === "number" && typeof b === "number" ? Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b)) : a === b;
}

/**
 * `ref` is the record reference the cockpit was opened with (a share token or a sample id);
 * `base` is the app's address up to the hash (location.origin + location.pathname).
 */
export function exportPlan(
	record: LoadedRecord,
	knobs: Readonly<Record<string, KnobValue>>,
	estimates: Readonly<Record<string, Estimate | undefined>>,
	ref: string | null,
	base: string | null,
): ExportPlan {
	const map = record.job.script_map ?? null;
	const where = new Map((map?.knobs ?? []).map((k) => [k.id, k]));
	const config: Record<string, KnobValue> = {};
	const changes: PlanChange[] = [];
	for (const k of record.knobs) {
		const to = knobs[k.id] ?? k.current;
		config[k.id] = to;
		if (same(to, k.current)) continue;
		const sk = where.get(k.id);
		changes.push({
			knobId: k.id,
			labelKey: k.label_key,
			unit: k.unit ?? null,
			from: k.current,
			to,
			needsCode: Boolean(k.needs_code ?? sk?.needs_code),
			patchTemplate: k.patch_template ?? null,
			location: sk?.location ?? null,
		});
	}
	const readouts: PlanReadout[] = [];
	for (const target of record.predictions.targets) {
		if (!target.readout) continue;
		const e = estimates[target.id];
		const status = (e?.status ?? "abstain") as EstimateStatus;
		const ok = e !== undefined && TRUSTED.has(status);
		readouts.push({
			targetId: target.id,
			readout: target.readout,
			unit: target.unit,
			status,
			point: ok ? e.point ?? null : null,
			lo: ok ? e.lo ?? null : null,
			hi: ok ? e.hi ?? null : null,
		});
	}
	const time = readouts.find((r) => r.readout === "time" && r.unit === "s");
	const patches = record.patches ?? [];
	const code = changes.filter((c) => c.needsCode);
	const verified = (c: PlanChange) =>
		patches.some((p) => p.verify?.status === "passed" && (p.template_id ?? null) !== null && p.template_id === c.patchTemplate);
	const gpuKnob = config.gpu_type ?? config.gpu;
	const src = record.job.source;
	return {
		lens: record.job.lens,
		title: record.job.title ?? null,
		source: { kind: src.kind, uri: src.uri ?? null, commit: src.commit ?? null, entrypoint: src.entrypoint ?? map?.entrypoint ?? null, args: src.args ?? [] },
		launch: map?.launch ?? null,
		gpu: { name: typeof gpuKnob === "string" ? gpuKnob : (record.env?.gpu?.name ?? null), count: typeof config.gpus === "number" ? config.gpus : (record.env?.gpu?.count ?? 1) },
		config,
		changes,
		readouts,
		wallTimeHiS: time?.hi ?? time?.point ?? null,
		patches,
		codeChangesVerified: code.every(verified),
		shareUrl: ref && base ? `${base}#/cockpit/${encodeURIComponent(ref)}` : null,
	};
}
