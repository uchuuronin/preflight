/**
 * D5 · Notebook intake: what a visitor's .ipynb becomes, before anything runs.
 *
 * A converts notebooks on the server (docs/decisions/upload-format.md, runner/sources.py); D
 * sends the notebook as kind "notebook" and shows a preview of exactly what A will run. The
 * conversion below is a line-for-line port of A's `_notebook_to_script` and
 * `notebook_requirements`, held to A's output by fixtures that A's own code produced
 * (convert.test.ts here, tests/test_d_notebook_parity.py on A's side).
 *
 * D adds two things A doesn't need but a visitor does: saved outputs are left out of what is
 * sent (A never reads them; they can be megabytes and can hold data), and the file name is
 * cleaned to what A accepts (a browser's "train (1).ipynb" otherwise fails A's path rule).
 */
import { MAX_TOTAL_BYTES as CHECKIN_FILE_BYTES, checkinBytes } from "../../app/upload";

/** A's per-file cap, after conversion [cited: upload-format.md, probes/local.py MAX_FILE_BYTES]. */
export const MAX_FILE_BYTES = 8 * 1024 * 1024;
/** A's cap on all files together [cited: upload-format.md, probes/local.py MAX_TOTAL_BYTES]. */
export const MAX_TOTAL_BYTES = 25 * 1024 * 1024;
/**
 * Most bytes a check-in carries as files, counted as JSON: C's limit at the desk, inside the API's
 * 6 MiB body cap [cited: app/upload.ts MAX_TOTAL_BYTES, api routes/jobs.py MAX_BODY_BYTES].
 */
export const CHECKIN_BYTES = CHECKIN_FILE_BYTES;
/** Longest stem D sends [derived: A accepts a converted name of up to 200 characters]. */
const MAX_STEM = 120;

export type NotebookError = "not_ipynb" | "not_json" | "no_cells" | "cell_not_text" | "no_code" | "too_big" | "bad_text";

export interface NotebookPreview {
	/** Exactly the files A will run: {"<stem>.py": script, "requirements.txt"?: pip lines}. */
	files: Record<string, string>;
	entrypoint: string;
	requirements: string[];
	codeCells: number;
	/** Markdown and raw cells, which don't run. */
	otherCells: number;
	/** Shell and magic lines (!pip, %time) that become comments. */
	commentedLines: number;
	/** Bytes of saved outputs and metadata left out of what is sent. */
	removedBytes: number;
	/** The original file name, when D had to clean it. */
	renamedFrom: string | null;
}

// A type alias, not an interface: C's IntakePanelProps hands back JobRequest["source"], an index
// signature object, and only an alias is assignable to it (PM, merge of ws-c and ws-d, Oct 28).
export type NotebookSource = {
	kind: "notebook";
	files: Record<string, string>;
};

export type Prepared = { ok: true; source: NotebookSource; preview: NotebookPreview } | { ok: false; error: NotebookError };

class ConvertError extends Error {
	constructor(readonly code: NotebookError) {
		super(code);
	}
}

// Python's str.isspace() set, so lstrip() and \s match A's code exactly.
const PY_WS = " \\t\\n\\r\\v\\f\\x1c-\\x1f\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
// Python's str.splitlines() boundaries.
const PY_BREAK = "\\n\\r\\v\\f\\x1c\\x1d\\x1e\\x85\\u2028\\u2029";
const LINES_KEEPENDS = new RegExp(`[^${PY_BREAK}]*(?:\\r\\n|[${PY_BREAK}])|[^${PY_BREAK}]+$`, "g");
const LSTRIP = new RegExp(`^[${PY_WS}]+`);

function splitlinesKeepends(text: string): string[] {
	return text.match(LINES_KEEPENDS) ?? [];
}

function splitlines(text: string): string[] {
	return splitlinesKeepends(text).map((l) => l.replace(new RegExp(`(?:\\r\\n|[${PY_BREAK}])$`), ""));
}

interface Notebook {
	nbformat?: unknown;
	nbformat_minor?: unknown;
	metadata?: Record<string, unknown>;
	cells: unknown[];
}

/**
 * Python's json reads bare NaN, Infinity and -Infinity, which Jupyter writes in outputs and widget
 * state; JSON.parse doesn't. Outside strings they become null (D drops outputs anyway).
 */
// A JSON string (unrolled so long strings scan in linear time), or a bare NaN or [-]Infinity.
const PY_TOKENS = /"[^"\\]*(?:\\[\s\S][^"\\]*)*"|-?Infinity|NaN/g;

function pythonJson(text: string): string {
	if (!/NaN|Infinity/.test(text)) return text;
	return text.replace(PY_TOKENS, (m) => (m.charCodeAt(0) === 34 ? m : "null"));
}

/** Parse once; everything else works on the parsed notebook. */
function parseNotebook(text: string): Notebook {
	let nb: unknown;
	try {
		nb = JSON.parse(pythonJson(text));
	} catch {
		throw new ConvertError("not_json");
	}
	const cells = nb && typeof nb === "object" && !Array.isArray(nb) ? (nb as { cells?: unknown }).cells : undefined;
	if (!Array.isArray(cells)) throw new ConvertError("no_cells");
	return nb as Notebook;
}

function isCode(cell: unknown): cell is { cell_type: "code"; source?: unknown } {
	return !!cell && typeof cell === "object" && !Array.isArray(cell) && (cell as { cell_type?: unknown }).cell_type === "code";
}

function cellLines(cell: { source?: unknown }): string[] {
	const src = "source" in cell ? cell.source : "";
	if (typeof src === "string") return splitlinesKeepends(src);
	if (Array.isArray(src) && src.every((l) => typeof l === "string")) return src as string[];
	throw new ConvertError("cell_not_text");
}

function isMagic(line: string): boolean {
	const s = line.replace(LSTRIP, "");
	return s.startsWith("!") || s.startsWith("%");
}

function scriptFrom(cells: unknown[]): string {
	const out: string[] = [];
	for (const cell of cells) {
		if (!isCode(cell)) continue;
		out.push(cellLines(cell).map((ln) => (isMagic(ln) ? `# ${ln}` : ln)).join(""));
	}
	return `${out.join("\n\n")}\n`;
}

/** A's `_notebook_to_script`: code cells in order; shell and magic lines become comments. */
export function notebookToScript(text: string): string {
	return scriptFrom(parseNotebook(text).cells);
}

const PIP = new RegExp(`^[${PY_WS}]*[!%][${PY_WS}]*(?:python3?[${PY_WS}]+-m[${PY_WS}]+)?pip3?[${PY_WS}]+install[${PY_WS}]+(.*)$`, "s");
const PIP_TAKES_VALUE = new Set([
	"-r",
	"--requirement",
	"-e",
	"--editable",
	"-c",
	"--constraint",
	"-i",
	"--index-url",
	"--extra-index-url",
	"-f",
	"--find-links",
	"--target",
	"-t",
	"--prefix",
	"--root",
]);

/** A's `notebook_requirements`: the packages `!pip install` / `%pip install` lines ask for. */
export function notebookRequirements(script: string): string[] {
	const reqs: string[] = [];
	for (const line of splitlines(script)) {
		const m = PIP.exec(line.startsWith("# ") ? line.slice(2) : line);
		if (!m) continue;
		const words = m[1]
			.split(/[#;|&`$]/, 1)[0]
			.split(new RegExp(`[${PY_WS}]+`))
			.filter(Boolean);
		let skip = false;
		for (let w of words) {
			if (skip) {
				skip = false;
				continue;
			}
			if (PIP_TAKES_VALUE.has(w)) {
				skip = true;
				continue;
			}
			w = w.replace(/^['"]+|['"]+$/g, "");
			if (!w || w.startsWith("-")) continue;
			if (!reqs.includes(w)) reqs.push(w);
		}
	}
	return reqs;
}

/** A file name A accepts: last path part, accents folded, other characters made "_". */
export function safeNotebookName(name: string): string {
	const base = name.split(/[\\/]/).pop() ?? "";
	const stem = base.replace(/\.ipynb$/i, "");
	const clean = stem
		.normalize("NFKD")
		.replace(/\p{M}+/gu, "")
		.replace(/[^A-Za-z0-9._ -]+/g, "_")
		.replace(/ *_[ _]*/g, "_")
		.replace(/^[._ -]+/, "")
		.replace(/[_ ]+$/, "")
		.slice(0, MAX_STEM);
	return `${clean || "notebook"}.ipynb`;
}

const utf8 = (s: string) => new TextEncoder().encode(s).length;

const isObject = (x: unknown): x is Record<string, unknown> => !!x && typeof x === "object" && !Array.isArray(x);

/** Text with an unpaired surrogate, which A can't encode as UTF-8. */
const BROKEN = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

/**
 * The notebook as sent: code cells' sources unchanged; outputs, attachments, cell metadata and
 * other cells' text left out (A reads only cell type and code). Notebook metadata keeps only
 * kernelspec and language_info, when they are objects.
 */
function stripped(nb: Notebook): string {
	const meta: Record<string, unknown> = {};
	if (isObject(nb.metadata)) for (const key of ["kernelspec", "language_info"]) if (isObject(nb.metadata[key])) meta[key] = nb.metadata[key];
	const cells = nb.cells.map((cell) => {
		if (!cell || typeof cell !== "object" || Array.isArray(cell)) return cell;
		const c = cell as { cell_type?: unknown; source?: unknown };
		if (c.cell_type !== "code") return { cell_type: c.cell_type, metadata: {}, source: "" };
		const out: Record<string, unknown> = { cell_type: "code", metadata: {}, execution_count: null, outputs: [] };
		if ("source" in c) out.source = c.source;
		return out;
	});
	return JSON.stringify({ nbformat: nb.nbformat ?? 4, nbformat_minor: nb.nbformat_minor ?? 5, metadata: meta, cells });
}

/** Everything the intake needs: what to send, and a preview of what A will run. */
export function prepareNotebook(name: string, text: string): Prepared {
	if (!/\.ipynb$/i.test(name)) return { ok: false, error: "not_ipynb" };
	try {
		const nb = parseNotebook(text);
		const cells = nb.cells;
		const script = scriptFrom(cells);
		if (!script.trim()) return { ok: false, error: "no_code" };
		const codeCells = cells.filter(isCode).length;
		const commentedLines = cells.filter(isCode).reduce<number>((n, c) => n + cellLines(c).filter(isMagic).length, 0);
		const requirements = notebookRequirements(script);
		const safe = safeNotebookName(name);
		const entrypoint = safe.replace(/\.ipynb$/, ".py");
		const files: Record<string, string> = { [entrypoint]: script };
		if (requirements.length) files["requirements.txt"] = `${requirements.join("\n")}\n`;
		if (BROKEN.test(script)) return { ok: false, error: "bad_text" };
		const sizes = Object.values(files).map(utf8);
		if (sizes.some((s) => s > MAX_FILE_BYTES) || sizes.reduce((a, b) => a + b, 0) > MAX_TOTAL_BYTES) {
			return { ok: false, error: "too_big" };
		}
		const sent = stripped(nb);
		// Measured as C's desk measures a handed source: the whole source as JSON, escapes included.
		const source: NotebookSource = { kind: "notebook", files: { [safe]: sent } };
		if (checkinBytes(source) > CHECKIN_BYTES) return { ok: false, error: "too_big" };
		return {
			ok: true,
			source,
			preview: {
				files,
				entrypoint,
				requirements,
				codeCells,
				otherCells: cells.length - codeCells,
				commentedLines,
				removedBytes: Math.max(0, utf8(text) - utf8(sent)),
				renamedFrom: safe === name ? null : name,
			},
		};
	} catch (error) {
		if (error instanceof ConvertError) return { ok: false, error: error.code };
		throw error;
	}
}
