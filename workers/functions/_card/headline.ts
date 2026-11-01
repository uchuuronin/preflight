/**
 * What a share card says about a flight record: a title, a subtitle and up to four headline
 * numbers, each with its status word as the app shows it (Measured, Predicted, Extrapolated,
 * Not trusted, Illustrative). Every number comes from the record itself: the engine's estimates
 * when the record has them, or else the probes' own measurements. Nothing is computed here
 * beyond unit conversion and picking the largest measured peak.
 */

export type StatusWord = "Measured" | "Predicted" | "Extrapolated" | "Not trusted" | "Illustrative";

export interface Line {
	label: string;
	value: string;
	status: StatusWord;
}

export interface Headline {
	title: string;
	subtitle: string;
	lines: Line[];
	/** True when any line is illustrative: the card says so in its footer. */
	illustrative: boolean;
}

interface Estimate {
	point?: number | null;
	lo?: number | null;
	hi?: number | null;
	unit?: string;
	status?: string;
	model_id?: string | null;
}

interface Probe {
	id?: string;
	status?: string;
	knobs?: Record<string, unknown>;
	memory?: { peak_allocated_bytes?: number | null } | null;
	step_time?: { p50_s?: number | null } | null;
}

/** The parts of a FlightRecord a card reads. Everything is optional: a card never throws on a record. */
export interface RecordLike {
	created_at?: string;
	job?: { lens?: string; title?: string | null; source?: { entrypoint?: string | null } | null } | null;
	env?: { gpu?: { name?: string; count?: number; memory_bytes?: number | null } | null } | null;
	probes?: Probe[] | null;
	predictions?: { estimates?: Record<string, Estimate> | null } | null;
	models?: Array<{ id?: string; fixture?: boolean }> | null;
}

const LENS: Record<string, string> = {
	finetune: "Fine-tune",
	pretrain: "Pretraining run",
	rl: "RL post-training run",
	agent: "Agent run",
	eval: "Evaluation sweep",
	serving: "Serving",
	retrieval: "Retrieval",
	gnn: "Graph neural network",
	vision: "Vision training",
	campaign: "Campaign",
};

/** The estimates a card prefers, in order, with the label each gets. */
const PREFERRED: Array<[string, string]> = [
	["memory.peak", "Peak memory"],
	["time.total", "Time to finish"],
	["cost.total", "Run cost"],
	["risk.oom", "Chance of running out of memory"],
];

const GIB = 1024 ** 3;

function num(x: unknown): x is number {
	return typeof x === "number" && Number.isFinite(x);
}

function sig(x: number): string {
	if (x >= 100) return x.toFixed(0);
	if (x >= 10) return x.toFixed(1).replace(/\.0$/, "");
	return x.toFixed(2).replace(/0$/, "").replace(/\.0$/, "");
}

export function formatValue(x: number, unit: string | undefined): string {
	switch (unit) {
		case "bytes":
			return `${sig(x / GIB)} GiB`;
		case "s":
			return duration(x);
		case "usd":
			return `US$${x >= 100 ? x.toFixed(0) : x.toFixed(2)}`;
		case "probability":
			return `${Math.round(x * 100)}%`;
		case "tokens":
			return x >= 1e6 ? `${sig(x / 1e6)}M tokens` : x >= 1e3 ? `${sig(x / 1e3)}k tokens` : `${Math.round(x)} tokens`;
		default:
			return unit ? `${sig(x)} ${unit}` : sig(x);
	}
}

export function duration(s: number): string {
	if (s < 60) return `${sig(s)} s`;
	const m = Math.round(s / 60);
	if (m < 60) return `${m} min`;
	const h = Math.floor(m / 60);
	const rest = m % 60;
	return rest ? `${h} h ${rest} min` : `${h} h`;
}

function range(e: Estimate): string {
	const { point, lo, hi, unit } = e;
	if (num(lo) && num(hi)) {
		const a = formatValue(lo, unit);
		const b = formatValue(hi, unit);
		if (a === b) return a;
		// "8.3 to 10.1 GiB"; durations and dollars keep both units ("1 h 13 min to 1 h 24 min").
		const unitOnce = / (GiB|M tokens|k tokens|tokens)$/.exec(a);
		return unitOnce && b.endsWith(unitOnce[0]) ? `${a.slice(0, -unitOnce[0].length)} to ${b}` : `${a} to ${b}`;
	}
	return num(point) ? formatValue(point, unit) : "";
}

function word(e: Estimate, fixtures: Set<string>): StatusWord {
	if (e.model_id && fixtures.has(e.model_id)) return "Illustrative";
	switch (e.status) {
		case "measured":
			return "Measured";
		case "extrapolated":
			return "Extrapolated";
		case "abstain":
			return "Not trusted";
		default:
			return "Predicted";
	}
}

function knobText(knobs: Record<string, unknown> | undefined): string {
	if (!knobs) return "";
	const parts: string[] = [];
	for (const [k, v] of Object.entries(knobs)) {
		if (typeof v !== "number" && typeof v !== "string") continue;
		parts.push(`${k.replace(/_/g, " ")} ${v}`);
		if (parts.length === 2) break;
	}
	return parts.join(", ");
}

export function headline(rec: RecordLike): Headline {
	const lens = rec.job?.lens ?? "";
	const entry = rec.job?.source?.entrypoint ?? "";
	const title = (rec.job?.title || [lens ? (LENS[lens] ?? "ML job") : "", entry].filter(Boolean).join(" · ")).trim();
	const gpu = rec.env?.gpu;
	const subtitleParts = [gpu?.name ? `${gpu.count && gpu.count > 1 ? `${gpu.count} x ` : ""}${gpu.name}` : "", rec.created_at ? rec.created_at.slice(0, 10) : ""];
	const fixtures = new Set((rec.models ?? []).filter((m) => m.fixture && m.id).map((m) => m.id as string));
	const estimates = rec.predictions?.estimates ?? {};
	const lines: Line[] = [];

	for (const [id, label] of PREFERRED) {
		const e = estimates[id];
		if (!e) continue;
		const status = word(e, fixtures);
		const value = status === "Not trusted" ? "No number" : range(e);
		if (value) lines.push({ label, value, status });
	}

	// No estimates (a record from before the engine ran on it): the probes' own measurements.
	if (!lines.length) {
		const probes = rec.probes ?? [];
		const oom = probes.find((p) => p.status === "oom");
		if (oom) lines.push({ label: "Ran out of memory while probing", value: knobText(oom.knobs) || "yes", status: "Measured" });
		let peak: Probe | null = null;
		for (const p of probes) {
			const b = p.memory?.peak_allocated_bytes;
			if (num(b) && (!peak || b > (peak.memory?.peak_allocated_bytes ?? 0))) peak = p;
		}
		const peakBytes = peak?.memory?.peak_allocated_bytes;
		if (num(peakBytes)) {
			const of = num(gpu?.memory_bytes) ? ` of ${sig(gpu.memory_bytes / GIB)}` : "";
			lines.push({ label: "Peak memory in a probe", value: `${sig(peakBytes / GIB)}${of} GiB`, status: "Measured" });
		}
		const ok = probes.find((p) => p.status === "ok" && num(p.step_time?.p50_s));
		const p50 = ok?.step_time?.p50_s;
		if (ok && num(p50)) {
			const at = knobText(ok.knobs);
			lines.push({ label: at ? `Step time at ${at}` : "Step time", value: duration(p50), status: "Measured" });
		}
	}

	return {
		title: title || "A Preflight flight record",
		subtitle: subtitleParts.filter(Boolean).join(" · "),
		lines: lines.slice(0, 4),
		illustrative: lines.some((l) => l.status === "Illustrative"),
	};
}
