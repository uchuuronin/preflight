/**
 * What a share card says about a flight record: a title, a subtitle and up to four headline
 * numbers, each with its status word as the app shows it (Measured, Predicted, Extrapolated,
 * Not measured here, Illustrative). Every number comes from the record itself: the engine's estimates
 * when the record has them, or else the probes' own measurements. Nothing is computed here
 * beyond unit conversion and picking the largest measured peak.
 */

export type StatusWord = "Measured" | "Predicted" | "Extrapolated" | "Not measured here" | "Illustrative";

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
	error_code?: string | null;
	memory?: { peak_allocated_bytes?: number | null } | null;
	/** The eval pass's own peak and batch size (an eval OOM peaks here, not in the train steps). */
	eval?: { peak_allocated_bytes?: number | null; batch_size?: number | null } | null;
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
	/** B's run cost breakdown: the money at risk if the run is launched as it is. */
	job_cost?: { at_risk_usd?: Estimate | null } | null;
	/** What Preflight spent making this record: GPU and model calls. */
	costs?: { gpu_usd?: number | null; llm_usd?: number | null } | null;
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
	if (x > 0 && x < 0.01) return x.toPrecision(1);
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
	if (s < 1) return `${sig(s * 1000)} ms`;
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
			return "Not measured here";
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
		const value = status === "Not measured here" ? "No number" : range(e);
		if (value) lines.push({ label, value, status });
	}

	// No estimates (a record from before the engine ran on it): the probes' own measurements.
	if (!lines.length) {
		const probes = rec.probes ?? [];
		const oom = probes.find((p) => p.status === "oom");
		if (oom) {
			// An eval OOM (error code oom.eval) happened in the eval pass, at the eval batch size.
			const inEval = oom.error_code === "oom.eval";
			const evalBatch = oom.eval?.batch_size;
			const value = inEval && num(evalBatch) ? `eval batch size ${evalBatch}` : knobText(oom.knobs) || "yes";
			lines.push({ label: inEval ? "Ran out of memory in the evaluation pass" : "Ran out of memory while probing", value, status: "Measured" });
		}
		// The largest measured peak, train steps or eval pass.
		let peakBytes: number | null = null;
		for (const p of probes) {
			for (const b of [p.memory?.peak_allocated_bytes, p.eval?.peak_allocated_bytes]) if (num(b) && (peakBytes === null || b > peakBytes)) peakBytes = b;
		}
		if (peakBytes !== null) {
			const of = num(gpu?.memory_bytes) ? ` of ${(gpu.memory_bytes / GIB).toFixed(2)}` : "";
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

/** Dollars for the at-risk line: cents, or "under US$0.01" (a sub-cent figure isn't worth its digits). */
function money(x: number): string {
	return x < 0.005 ? "under US$0.01" : formatValue(x, "usd");
}

/**
 * The share page's one money line (Chiya, Nov 3): "At risk if run as is: US$X (range, status) ·
 * spent to find out: US$Y". Both from the record: the run cost's at-risk estimate (chance of
 * failure x the cost up to the failure point, B's `job_cost.at_risk_usd`) and what Preflight spent
 * making the record (`costs`, GPU and model calls). Null unless both exist: an abstaining or missing
 * at-risk figure, an illustrative one, or no recorded spend gives no line.
 */
export function atRiskLine(rec: RecordLike): string | null {
	const e = rec.job_cost?.at_risk_usd;
	if (!e || e.status === "abstain" || !num(e.point)) return null;
	const fixtures = new Set((rec.models ?? []).filter((m) => m.fixture && m.id).map((m) => m.id as string));
	// Illustrative numbers, on this figure or anywhere on the card: a sample's spend isn't anyone's money.
	if ((e.model_id && fixtures.has(e.model_id)) || headline(rec).illustrative) return null;
	const spent = (num(rec.costs?.gpu_usd) ? rec.costs!.gpu_usd! : 0) + (num(rec.costs?.llm_usd) ? rec.costs!.llm_usd! : 0);
	if (!(spent > 0)) return null;
	const point = money(e.point);
	const lo = num(e.lo) ? money(e.lo) : point;
	const hi = num(e.hi) ? money(e.hi) : point;
	const status = (e.status ?? "predicted").toLowerCase();
	const inside = lo === hi ? status : `${lo === "under US$0.01" ? "US$0" : lo} to ${hi}, ${status}`;
	return `At risk if run as is: ${point} (${inside}) · spent to find out: ${money(spent)}`;
}
