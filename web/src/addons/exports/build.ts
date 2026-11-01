/**
 * Exports (moved to D, Oct 31): what a visitor takes away from the cockpit. Everything comes from
 * C's ExportPlan (web/src/workspace/export/plan.ts), which holds the settings on screen and the
 * cockpit's own estimates, so an export prints no number the cockpit wouldn't; this file only
 * formats. Pure: no React, no network, no clock.
 *
 * - config: the plan as JSON (settings, what changed, the trusted readouts with their ranges)
 * - edits: each changed setting at its line in the script; where the record names the exact
 *   expression, a shell script that checks every line first and then makes the edits
 * - sbatch: a Slurm batch file asking for the upper end of the predicted time
 * - command: the command that runs the job
 * - link: the record's read-only link
 * - recheck: install Preflight's CLI and measure again (behind d.exports_cli)
 *
 * A true unified diff needs the script's text, which the browser never has; Muse Code's patches
 * are listed by title and verify status until their diffs can be fetched (requests in from-d.md).
 */
import type { ExportPlan } from "../../registry";
import type { KnobValue, LoadedRecord } from "../../workspace/record/types";

export interface ExportInput {
	/** C's plan for the settings on screen. */
	plan: ExportPlan;
	/** The record, for what the plan doesn't carry: nodes, the backend, the time limit. */
	record: LoadedRecord;
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

/** Settings about where the job runs (C's plan reads gpu_type, gpu and gpus); launch-only when the script doesn't set them. */
const LAUNCH_KNOBS = new Set(["gpu_type", "gpu", "gpus", "gpu_count"]);

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

export function changes(input: Pick<ExportInput, "plan">): Change[] {
	return input.plan.changes.map((c) => {
		const loc = c.location;
		const base = {
			id: c.knobId,
			from: c.from,
			to: c.to,
			file: loc?.file ?? null,
			line: loc?.line ?? null,
			expr: loc?.expr ?? null,
			template: c.patchTemplate,
		};
		// Where it runs, unless the script itself sets it (e.g. torchrun --nproc_per_node on the launch line).
		if (LAUNCH_KNOBS.has(c.knobId) && !loc) return { ...base, newExpr: null, how: "launch" as const };
		// Only what C counts as a code change, so "not verified" covers every one listed as such.
		if (c.needsCode) return { ...base, newExpr: null, how: "code" as const };
		const newExpr = loc?.expr && loc.file && loc.line ? newExpression(loc.expr, c.from, c.to) : null;
		return { ...base, newExpr, how: newExpr ? ("exact" as const) : ("by_hand" as const) };
	});
}

export function gpu(input: Pick<ExportInput, "plan">): { name: string | null; count: number } {
	const { name, count } = input.plan.gpu;
	return { name, count: Math.max(1, Math.round(count)) };
}

function finite(x: number | null | undefined): x is number {
	return typeof x === "number" && Number.isFinite(x);
}

/**
 * What the batch file asks for: the upper end of the predicted time (C's wallTimeHiS), and the
 * record's time limit when that is shorter (Slurm won't start a job asking for more than the
 * limit). Null when the cockpit doesn't trust a time.
 */
export function askedTime(input: Pick<ExportInput, "plan" | "record">): { need: number; ask: number; capped: boolean } | null {
	const need = input.plan.wallTimeHiS;
	if (!finite(need) || need <= 0) return null;
	const cap = input.record.job.constraints?.wall_time_cap_s;
	const rounded = Math.ceil(need / 300) * 300; // whole 5 minutes
	if (finite(cap) && cap > 0 && rounded > cap) return { need, ask: Math.floor(cap / 60) * 60, capped: need > cap }; // never past the limit
	return { need, ask: rounded, capped: false };
}

/** Whether every code change has a verified Muse Code patch (C's plan); true with none. */
export function verified(input: Pick<ExportInput, "plan">): boolean {
	return input.plan.codeChangesVerified;
}

/** The plan as JSON. Readouts the cockpit doesn't trust are named, never given a number. */
export function configText(input: ExportInput): string {
	const { plan } = input;
	const changed: Record<string, { from: KnobValue | null; to: KnobValue }> = {};
	for (const c of plan.changes) changed[c.knobId] = { from: c.from, to: c.to };
	const predicted: Record<string, unknown> = {};
	const unpredicted: string[] = [];
	for (const r of plan.readouts) {
		if (!finite(r.point)) {
			unpredicted.push(r.targetId);
			continue;
		}
		predicted[r.targetId] = {
			point: r.point,
			...(finite(r.lo) && finite(r.hi) ? { lo: r.lo, hi: r.hi } : {}),
			unit: r.unit,
			status: r.status,
		};
	}
	const g = gpu(input);
	const out = {
		preflight: "plan",
		...(plan.shareUrl ? { link: plan.shareUrl } : {}),
		...(input.illustrative ? { note: "Sample record: the numbers are illustrative, not measured." } : {}),
		job: { title: plan.title, lens: plan.lens, entrypoint: plan.source.entrypoint ?? null, command: commandText(input) },
		gpu: { type: g.name, count: g.count },
		settings: plan.config,
		changed,
		...(plan.changes.some((c) => c.needsCode) ? { code_changes_verified: plan.codeChangesVerified } : {}),
		...(plan.codeChangesVerified ? {} : { not_verified: "A code change here has no Muse Code patch that passed its verify run yet." }),
		predicted,
		...(unpredicted.length ? { not_predicted: unpredicted } : {}),
	};
	return `${JSON.stringify(out, null, "\t")}\n`;
}

/** POSIX shell single quotes. */
export function shq(s: string): string {
	return /^[A-Za-z0-9_./:=@%+,-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`;
}

/**
 * The plan's script arguments, shell-quoted; null when any holds a control character (a
 * carriage return or escape sequence can hide what a pasted command really says).
 */
function quotedArgs(plan: ExportPlan): string[] | null {
	const args = plan.source.args ?? [];
	return args.every((a) => oneLine(a) === a) ? args.map(shq) : null;
}

const ARGS_LEFT_OUT = "# The record's script arguments hold control characters, so they are left out: add them yourself.";

/** The command line that runs the job, from the plan. */
export function commandText(input: Pick<ExportInput, "plan">): string {
	const { plan } = input;
	const entry = plan.source.entrypoint ?? "train.py";
	const launch = oneLine(plan.launch ?? "").trim();
	// A's reader writes "python <entrypoint>" unquoted; quote the path so spaces survive.
	const head = !launch || launch === `python ${entry}` ? `python ${shq(entry)}` : launch;
	const args = quotedArgs(plan);
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
	if (!verified(input)) lines.push("# Not verified: a code change here has no Muse Code patch that passed its verify run yet.");
	if (exact.length) {
		lines.push("", ...HELPERS, "", "# Check every edit first; nothing changes unless all of them still hold.");
		for (const c of exact) lines.push(`check ${(editArgs(c) as string[]).slice(0, 3).join(" ")}`);
		lines.push("", "# Then make them.");
		for (const c of exact) lines.push(`apply ${(editArgs(c) as string[]).join(" ")}`);
	}
	return `${lines.join("\n")}\n`;
}

/** Slurm's time format, D-HH:MM:SS from one day up and HH:MM:SS below, in whole minutes. */
export function slurmTime(seconds: number, round: "up" | "down" | "minute" = "up"): string {
	// "up": an ask, up to whole 5 minutes; "down": a limit, seconds dropped; "minute": a figure, up to the minute
	const step = round === "up" ? 300 : 60;
	const total = round === "down" ? Math.floor(seconds / step) * step : Math.ceil(seconds / step) * step;
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

/** Whether the job calls APIs rather than running on a GPU. */
export function isApi(input: Pick<ExportInput, "plan" | "record">): boolean {
	return input.record.env?.backend === "api" || input.plan.lens === "agent";
}

/**
 * Whether the CLI can measure this job again on the visitor's machine: not an API run (the CLI
 * measures nothing for those), and not a sample or a notebook (no project folder of theirs holds
 * the script the record ran).
 */
export function canRecheck(input: Pick<ExportInput, "plan" | "record">): boolean {
	const kind = input.plan.source.kind;
	return !isApi(input) && kind !== "sample" && kind !== "notebook";
}

/**
 * Install Preflight's CLI and measure the job again on the visitor's own machine, after the
 * edits; null when that can't work (canRecheck). Checked Oct 31: the plain install plus the
 * user's own torch runs `preflight run` (Python 3.11 or 3.12; torch is not installed by it).
 * Records go to a dot-folder, which the CLI skips when it reads the project next time.
 */
export function recheckText(input: Pick<ExportInput, "plan" | "record">): string | null {
	if (!canRecheck(input)) return null;
	const { plan } = input;
	const entry = plan.source.entrypoint ?? "train.py";
	const args = quotedArgs(plan);
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

/** A Slurm batch file: GPUs from the plan, time from the upper end of the predicted time. */
export function sbatchText(input: ExportInput): string {
	const { plan, record } = input;
	const g = gpu(input);
	const nodes = Math.max(1, Math.round(record.env?.nodes ?? 1));
	const lines = ["#!/bin/bash", `#SBATCH --job-name=${slug(plan.title ?? plan.lens)}`, `#SBATCH --nodes=${nodes}`];
	if (!isApi(input)) {
		// --gpus is the job's total, the plan's own count, however Slurm spreads it over the nodes.
		lines.push(`#SBATCH --gpus=${g.count}`);
		if (g.name) lines.push(oneLine(`# Planned on ${g.count} x ${g.name}. If your cluster names GPU types, add it: --gpus=<type>:${g.count}`));
	}
	const asked = askedTime(input);
	if (asked?.capped) {
		lines.push(`#SBATCH --time=${slurmTime(asked.ask, "down")}`);
		lines.push(`# Your limit. The plan may need up to ${slurmTime(asked.need, "minute")} (the upper end of Preflight's predicted time): save checkpoints and resubmit to resume.`);
	} else if (asked && asked.ask < Math.ceil(asked.need / 300) * 300) {
		lines.push(`#SBATCH --time=${slurmTime(asked.ask, "down")}`);
		lines.push("# Time: the upper end of Preflight's predicted time, held to your limit. Add your cluster's start-up time if it counts.");
	} else if (asked) {
		lines.push(`#SBATCH --time=${slurmTime(asked.ask)}`);
		lines.push("# Time: the upper end of Preflight's predicted time, rounded up to 5 minutes. Add your cluster's start-up time if it counts.");
	} else {
		lines.push("# Preflight has no trusted time at these settings: set --time yourself.");
	}
	lines.push("#SBATCH --output=%x-%j.out", "", "set -euo pipefail");
	const cs = changes(input).filter((c) => c.how !== "launch");
	const scripted = cs.filter((c) => c.how === "exact").length;
	const rest = cs.length - scripted;
	if (scripted) lines.push("# Run 'sh preflight-edits.sh' once before submitting: it makes the setting changes it can.");
	if (rest) lines.push(`# ${rest} more change${rest === 1 ? "" : "s"} to make by hand or with Muse Code: see preflight-edits.sh.`);
	if (!verified(input)) lines.push("# Not verified: a code change here has no Muse Code patch that passed its verify run yet.");
	if (input.illustrative) lines.push("# Sample record: sized from illustrative numbers, not a measurement.");
	lines.push(commandText(input));
	return `${lines.join("\n")}\n`;
}
