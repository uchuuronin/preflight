/**
 * Exports (moved to D, Oct 31): what a visitor takes away from the cockpit, built from the record
 * and the knob values on screen. Pure: no React, no network, no clock, so every file is the same
 * for the same record and settings.
 *
 * Five things:
 * - config: the plan as JSON (settings, what changed, the predicted readouts with their intervals)
 * - edits: each changed setting at its line in the script; where the record names the exact
 *   expression, a shell script that makes that one edit and refuses if the line has moved
 * - sbatch: a Slurm batch file sized from the slow end of the predicted time
 * - command: the command that runs the job
 * - link: the cockpit's own address for this record
 *
 * A true unified diff needs the script's text, which the browser never has; Muse Code's patches
 * are listed by title and verify status until their diffs can be fetched (requests in from-d.md).
 */
import type { Estimate, KnobValue, LoadedRecord } from "../../workspace/record/types";

/** Setup time added to the slow end of the predicted time, as the free-tier band does (D4). */
export const SETUP_S = 15 * 60;

export interface ExportInput {
	record: LoadedRecord;
	knobs: Readonly<Record<string, KnobValue>>;
	estimates: Readonly<Record<string, Estimate | undefined>>;
	/** The cockpit's address for this record, e.g. https://host/#/cockpit/<token>. */
	link: string;
	/** True for sample records with illustrative numbers (C's isFixture). */
	illustrative: boolean;
}

export interface Change {
	id: string;
	from: KnobValue | null;
	to: KnobValue;
	file: string | null;
	line: number | null;
	/** The expression as the record read it, e.g. "per_device_train_batch_size=4". */
	expr: string | null;
	/** The same expression with the new value, when the edit can be made exactly. */
	newExpr: string | null;
	/** Muse Code template that makes this change, for settings that need code. */
	template: string | null;
	/** What kind of change: an exact edit, one to make by hand, a code change, or no code at all. */
	how: "exact" | "by_hand" | "code" | "launch";
}

/** Settings that aren't in the script: where the job runs. They go into sbatch and the config. */
const LAUNCH_KNOBS = new Set(["gpu_type", "gpu_count"]);

/**
 * Text from the record (an uploaded script's names, expressions, values) on one line, for
 * comments in the generated files: a line break would end the comment and run the rest.
 */
export function oneLine(s: string): string {
	// biome-ignore lint/suspicious/noControlCharactersInRegex: control characters are what this removes
	return s.replace(/[\u0000-\u001f\u007f\u0085\u2028\u2029]+/g, " ");
}

/** A Python literal for a value, as A's override writer makes it; null when there is none. */
export function pyLiteral(value: KnobValue): string | null {
	if (typeof value === "boolean") return value ? "True" : "False";
	if (typeof value === "number") return Number.isFinite(value) ? String(value) : null;
	return JSON.stringify(value);
}

/** "NAME = 4", "name: int = 4", "name=4", "default=4": the left side and the value text. */
const SIMPLE = /^(\s*[A-Za-z_][\w.]*\s*(?::\s*[\w.[\], |]+?)?\s*=\s*)(\S(?:.*\S)?)\s*$/;

/** Whether the value text in the record's expression is the knob's recorded value. */
function sameValue(text: string, from: KnobValue | null): boolean {
	if (from === null) return false;
	if (typeof from === "boolean") return text === (from ? "True" : "False");
	if (typeof from === "number") return text.replace(/_/g, "") !== "" && Number(text.replace(/_/g, "")) === from;
	try {
		return JSON.parse(text.replace(/^'(.*)'$/, '"$1"')) === from;
	} catch {
		return false;
	}
}

function newExpression(expr: string, from: KnobValue | null, to: KnobValue): string | null {
	// One line only: the edit script finds the expression within a single line of the file.
	// biome-ignore lint/suspicious/noControlCharactersInRegex: control characters are refused here
	if (/[\u0000-\u001f\u007f\u0085\u2028\u2029]/.test(expr)) return null;
	const m = SIMPLE.exec(expr);
	if (!m) return null;
	const [, lhs, value] = m;
	if (!sameValue(value, from)) return null;
	if (typeof to !== typeof from) return null;
	let lit = pyLiteral(to);
	if (lit === null) return null;
	// Keep a float a float: "lr = 1.0" set to 2 is written "lr = 2.0", not "lr = 2".
	if (typeof to === "number" && Number.isInteger(to) && /[.eE]/.test(value) && !/^0[xob]/i.test(value)) lit = `${lit}.0`;
	if (/[\u0000-\u001f]/.test(lit)) return null;
	return `${lhs}${lit}`;
}

export function changes(input: Pick<ExportInput, "record" | "knobs">): Change[] {
	const { record, knobs } = input;
	const located = new Map((record.job.script_map?.knobs ?? []).map((k) => [k.id, k]));
	const out: Change[] = [];
	for (const k of record.knobs) {
		if (!(k.id in knobs)) continue;
		const to = knobs[k.id];
		const from = (k.current ?? null) as KnobValue | null;
		if (to === from) continue;
		const loc = located.get(k.id)?.location ?? null;
		const base = {
			id: k.id,
			from,
			to,
			file: loc?.file ?? null,
			line: loc?.line ?? null,
			expr: loc?.expr ?? null,
			template: k.patch_template ?? null,
		};
		if (LAUNCH_KNOBS.has(k.id)) {
			out.push({ ...base, newExpr: null, how: "launch" });
		} else if (k.needs_code || (!loc && k.patch_template)) {
			out.push({ ...base, newExpr: null, how: "code" });
		} else {
			const newExpr = loc?.expr && loc.file && loc.line ? newExpression(loc.expr, from, to) : null;
			out.push({ ...base, newExpr, how: newExpr ? "exact" : "by_hand" });
		}
	}
	return out;
}

export function gpu(input: Pick<ExportInput, "record" | "knobs">): { name: string | null; count: number } {
	const { record, knobs } = input;
	const name = typeof knobs.gpu_type === "string" ? knobs.gpu_type : (record.env?.gpu?.name ?? null);
	const count = typeof knobs.gpu_count === "number" ? knobs.gpu_count : (record.env?.gpu?.count ?? 1);
	return { name, count: Math.max(1, Math.round(count)) };
}

function usable(e: Estimate | undefined): e is Estimate {
	return !!e && e.status !== "abstain" && Number.isFinite(e.point);
}

function finite(x: number | null | undefined): x is number {
	return typeof x === "number" && Number.isFinite(x);
}

/**
 * The wall time to ask for, in seconds: the slow end of the predicted range (or the point) plus
 * setup, rounded up to whole 5 minutes as the batch file asks for it; null with no estimate.
 */
export function wallTime(e: Estimate | undefined): number | null {
	const point = e?.point;
	if (!usable(e) || !finite(point)) return null;
	return Math.ceil(((finite(e.hi) ? e.hi : point) + SETUP_S) / 300) * 300;
}

/**
 * What the batch file asks for: the wall time it needs, and the limit when that is shorter (Slurm
 * won't start a job asking for more than the limit). Null with no time estimate.
 */
export function askedTime(input: Pick<ExportInput, "record" | "estimates">): { need: number; ask: number; capped: boolean } | null {
	const need = wallTime(input.estimates["time.total"]);
	if (need === null) return null;
	const cap = input.record.job.constraints?.wall_time_cap_s;
	if (finite(cap) && cap > 0 && need > cap) return { need, ask: Math.floor(cap / 60) * 60, capped: true };
	return { need, ask: need, capped: false };
}

/** The plan as JSON. Values not predicted at these settings are left out, not guessed. */
export function configText(input: ExportInput): string {
	const { record, knobs, estimates } = input;
	const settings: Record<string, KnobValue> = {};
	for (const k of record.knobs) {
		const v = k.id in knobs ? knobs[k.id] : k.current;
		if (v !== null && v !== undefined) settings[k.id] = v as KnobValue;
	}
	const changed: Record<string, { from: KnobValue | null; to: KnobValue }> = {};
	for (const c of changes(input)) changed[c.id] = { from: c.from, to: c.to };
	const predicted: Record<string, unknown> = {};
	const unpredicted: string[] = [];
	for (const target of record.predictions.targets) {
		const e = estimates[target.id];
		if (!usable(e)) {
			unpredicted.push(target.id);
			continue;
		}
		predicted[target.id] = {
			point: e.point,
			...(finite(e.lo) && finite(e.hi) ? { lo: e.lo, hi: e.hi, level: e.level ?? null } : {}),
			unit: e.unit,
			status: e.status,
		};
	}
	const g = gpu(input);
	const plan = {
		preflight: "plan",
		link: input.link,
		...(input.illustrative ? { note: "Sample record: the numbers are illustrative, not measured." } : {}),
		job: {
			title: record.job.title ?? null,
			lens: record.job.lens,
			entrypoint: record.job.source?.entrypoint ?? record.job.script_map?.entrypoint ?? null,
			command: commandText(input),
		},
		gpu: { type: g.name, count: g.count },
		settings,
		changed,
		predicted,
		...(unpredicted.length ? { not_predicted: unpredicted } : {}),
	};
	return `${JSON.stringify(plan, null, "\t")}\n`;
}

/** POSIX shell single quotes. */
export function shq(s: string): string {
	return /^[A-Za-z0-9_./:=@%+,-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`;
}

/**
 * The record's script arguments, shell-quoted; null when any holds a control character (a
 * carriage return or escape sequence can hide what a pasted command really says).
 */
function quotedArgs(record: LoadedRecord): string[] | null {
	const args = record.job.source?.args ?? [];
	return args.every((a) => oneLine(a) === a) ? args.map(shq) : null;
}

const ARGS_LEFT_OUT = "# The record's script arguments hold control characters, so they are left out: add them yourself.";

/** The command line that runs the job, from the record. */
export function commandText(input: Pick<ExportInput, "record">): string {
	const { record } = input;
	const entry = record.job.source?.entrypoint ?? record.job.script_map?.entrypoint ?? "train.py";
	const launch = oneLine(record.job.script_map?.launch ?? "").trim();
	// A's reader writes "python <entrypoint>" unquoted; quote the path so spaces survive.
	const head = !launch || launch === `python ${entry}` ? `python ${shq(entry)}` : launch;
	const args = quotedArgs(record);
	return args ? [head, ...args].join(" ") : `${ARGS_LEFT_OUT}\n${head}`;
}

/**
 * The edit script's two helpers. The Perl program is fixed text: the file, line and both
 * expressions reach it only as arguments and environment variables, never as code. An
 * expression counts only as a whole token (no letter, digit, "_" or "." touching either end),
 * and only when the line holds it exactly once.
 */
const HELPERS = [
	`PF_AT='sub at { my ($s, $o) = @_; my @at; my $p = -1; while (($p = index($s, $o, $p + 1)) >= 0) { my $pre = $p ? substr($s, $p - 1, 1) : ""; my $post = substr($s, $p + length($o), 1); push @at, $p if $pre !~ /[\\w.]/ && $post !~ /[\\w.]/ } return @at }'`,
	"check() {",
	`\tPF_LINE="$2" PF_OLD="$3" perl -ne "$PF_AT"' $ok = (at($_, $ENV{PF_OLD}) == 1) if $. == $ENV{PF_LINE}; END { exit($ok ? 0 : 1) }' -- "$1" ||`,
	`\t\t{ printf '%s line %s no longer reads: %s\\nNothing was changed.\\n' "$1" "$2" "$3" >&2; exit 1; }`,
	"}",
	"apply() {",
	`\tPF_LINE="$2" PF_OLD="$3" PF_NEW="$4" perl -i -pe "$PF_AT"' if ($. == $ENV{PF_LINE}) { my @at = at($_, $ENV{PF_OLD}); substr($_, $at[0], length $ENV{PF_OLD}) = $ENV{PF_NEW} if @at == 1 }' -- "$1"`,
	"}",
];

/** The arguments for check/apply of one exact edit; null for any other kind of change. */
export function editArgs(c: Change): string[] | null {
	if (c.how !== "exact" || !c.file || !c.line || !c.expr || !c.newExpr) return null;
	return [shq(c.file), String(c.line), shq(c.expr), shq(c.newExpr)];
}

/**
 * Settings changed in the cockpit as a shell script: it checks every exact edit first and
 * changes nothing unless all of them still hold, then makes them; the rest are notes.
 */
export function editsText(input: ExportInput): string {
	const list = changes(input).filter((c) => c.how !== "launch");
	const lines = ["#!/bin/sh", "# Settings changed in Preflight. Run it with sh from your project's top folder, where these paths start.", "set -e"];
	if (!list.length) return `${lines.join("\n")}\n# No edits to the script.\n`;
	const exact = list.filter((c) => editArgs(c));
	const notes: string[] = [];
	for (const c of list) {
		const what = oneLine(`${c.id}: ${String(c.from)} -> ${String(c.to)}`);
		if (editArgs(c)) {
			notes.push(oneLine(`# ${what} (${c.file} line ${c.line}): made below.`));
		} else if (c.how === "code") {
			notes.push(oneLine(`# ${what}: needs a code change (Muse Code template ${c.template ?? "none"}); make it in the cockpit and take its patch.`));
		} else {
			const at = c.file && c.line ? ` at ${c.file} line ${c.line}${c.expr ? ` (${c.expr})` : ""}` : "";
			notes.push(oneLine(`# ${what}: edit by hand${at}.`));
		}
	}
	lines.push("", ...notes);
	if (exact.length) {
		lines.push("", ...HELPERS, "", "# Check every edit first; nothing changes unless all of them still hold.");
		for (const c of exact) lines.push(`check ${(editArgs(c) as string[]).slice(0, 3).join(" ")}`);
		lines.push("", "# Then make them.");
		for (const c of exact) lines.push(`apply ${(editArgs(c) as string[]).join(" ")}`);
	}
	return `${lines.join("\n")}\n`;
}

/** Slurm's time format, D-HH:MM:SS from one day up and HH:MM:SS below, in whole minutes. */
export function slurmTime(seconds: number, round: "up" | "down" = "up"): string {
	const step = round === "up" ? 300 : 60; // asks round up to 5 minutes; a stated limit only drops its seconds
	const total = round === "up" ? Math.ceil(seconds / step) * step : Math.floor(seconds / step) * step;
	const d = Math.floor(total / 86400);
	const h = Math.floor((total % 86400) / 3600);
	const m = Math.floor((total % 3600) / 60);
	const hms = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:00`;
	return d > 0 ? `${d}-${hms}` : hms;
}

function slug(s: string): string {
	return (
		s
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, "-")
			.replace(/^-|-$/g, "")
			.slice(0, 40)
			.replace(/-$/, "") || "preflight-job"
	);
}

/** Installs Preflight's command-line tool from the public repo (shown once the repo is public). */
export const CLI_INSTALL = "pip install git+https://github.com/uchuuronin/preflight";

/**
 * Whether the CLI can measure this job again on the visitor's machine: not an API run (the CLI
 * measures nothing for those), and not a sample or a notebook (no project folder of theirs holds
 * the script the record ran).
 */
export function canRecheck(record: LoadedRecord): boolean {
	const kind = record.job.source?.kind;
	return !isApi(record) && kind !== "sample" && kind !== "notebook";
}

/**
 * Install Preflight's CLI and measure the job again on the visitor's own machine, after the
 * edits; null when that can't work (canRecheck). Checked Oct 31: the plain install plus the
 * user's own torch runs `preflight run` (Python 3.11 or 3.12; torch is not installed by it).
 * Records go to a dot-folder, which the CLI skips when it reads the project next time.
 */
export function recheckText(input: Pick<ExportInput, "record">): string | null {
	const { record } = input;
	if (!canRecheck(record)) return null;
	const entry = record.job.source?.entrypoint ?? record.job.script_map?.entrypoint ?? "train.py";
	const args = quotedArgs(record);
	const run = ["preflight run .", `--entrypoint=${shq(entry)}`, "--out .preflight/records", ...(args?.length ? ["--", ...args] : [])].join(" ");
	return [
		"# Preflight's command-line tool (Python 3.11 or 3.12, in the environment that has your torch):",
		CLI_INSTALL,
		"# From your project's top folder, after the edits: measures a few steps on this machine,",
		"# on its GPU if it has one, and writes a new flight record to .preflight/records/.",
		...(args ? [] : [ARGS_LEFT_OUT]),
		run,
		"",
	].join("\n");
}

/** Whether the job calls APIs rather than running on a GPU. */
export function isApi(record: LoadedRecord): boolean {
	return record.env?.backend === "api" || record.job.lens === "agent";
}

/** A Slurm batch file: GPUs from the plan, time from the slow end of the predicted range. */
export function sbatchText(input: ExportInput): string {
	const { record, estimates } = input;
	const g = gpu(input);
	const nodes = Math.max(1, Math.round(record.env?.nodes ?? 1));
	const perNode = Math.ceil(g.count / nodes); // --gres counts GPUs per node
	const time = estimates["time.total"];
	const lines = ["#!/bin/bash", `#SBATCH --job-name=${slug(record.job.title ?? record.job.lens)}`, `#SBATCH --nodes=${nodes}`];
	if (!isApi(record)) {
		lines.push(`#SBATCH --gres=gpu:${perNode}`);
		if (g.name) lines.push(oneLine(`# Planned on ${g.count} x ${g.name}. If your cluster names GPU types, add it: --gres=gpu:<type>:${perNode}`));
	}
	const asked = askedTime(input);
	if (asked && time) {
		const range = finite(time.hi) ? (finite(time.level) ? `${Math.round(time.level * 100)}% range` : "range") : "estimate";
		if (asked.capped) {
			lines.push(`#SBATCH --time=${slurmTime(asked.ask, "down")}`);
			lines.push(`# Your limit. The plan needs about ${slurmTime(asked.need)} (the slow end of Preflight's ${range} plus 15 minutes to set up): save checkpoints and resubmit to resume.`);
		} else {
			lines.push(`#SBATCH --time=${slurmTime(asked.ask)}`);
			lines.push(`# Time: the slow end of Preflight's ${range} plus 15 minutes to set up.`);
		}
	} else {
		lines.push("# Preflight has no time estimate at these settings: set --time yourself.");
	}
	lines.push("#SBATCH --output=%x-%j.out", "", "set -euo pipefail");
	const cs = changes(input).filter((c) => c.how !== "launch");
	const scripted = cs.filter((c) => c.how === "exact").length;
	const rest = cs.length - scripted;
	if (scripted) lines.push("# Run 'sh preflight-edits.sh' once before submitting: it makes the setting changes it can.");
	if (rest) lines.push(`# ${rest} more change${rest === 1 ? "" : "s"} to make by hand or with Muse Code: see preflight-edits.sh.`);
	if (input.illustrative) lines.push("# Sample record: sized from illustrative numbers, not a measurement.");
	lines.push(commandText(input));
	return `${lines.join("\n")}\n`;
}
