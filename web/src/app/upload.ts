/**
 * Check-in uploads: the files a visitor picks, as a JobRequest source A's runner accepts
 * (docs/decisions/upload-format.md; A enforces the same rules in runner/sources.py). Checked here
 * first so a visitor learns what to change before anything is sent.
 */

export interface PickedFile {
	name: string;
	text: string;
	/** Bytes on disk, from File.size. */
	size: number;
}

export type UploadSource = { kind: "upload"; files: Record<string, string>; entrypoint: string } | { kind: "notebook"; files: Record<string, string> };

/** `scripts` lists the files that could be the entry point, so the desk can offer a choice. */
export type UploadCheck = { ok: true; source: UploadSource; scripts: string[] } | { ok: false; reason: string; candidates?: string[] };

// Source: derived, the API refuses a check-in body over 6 MiB (api/preflight_api/routes/jobs.py
// MAX_BODY_BYTES); 5 MB of files as JSON leaves room for the rest of the body, as the copy says.
export const MAX_TOTAL_BYTES = 5 * 1000 * 1000;
// Source: derived, A's upload limit (upload-format.md, "At most 2,000 files").
export const MAX_FILES = 2000;
// Source: derived, A's text-file types for repos (upload-format.md) plus notebooks and the small
// text data A allows inside an upload ("from text files inside the upload").
const TEXT_TYPES = [".py", ".ipynb", ".sh", ".sbatch", ".slurm", ".yaml", ".yml", ".json", ".toml", ".cfg", ".txt", ".md", ".csv", ".tsv", ".jsonl"];
// Source: derived, A's rule for picking the entry point (upload-format.md, "Entry point").
const ENTRY_ORDER = ["train.py", "main.py", "run.py", "finetune.py", "pretrain.py"];
// Source: derived, A's path rule (letters, digits, ". _ - /" and spaces) for a file picked on
// its own, which has no directory part.
const NAME = /^[A-Za-z0-9._ -]+$/;

/** The file picker's filter: the same types the check accepts. */
export const ACCEPT = TEXT_TYPES.join(",");

const lower = (n: string) => n.toLowerCase();

/** What can be said from names and sizes alone, before reading anything into memory. */
export function precheck(picked: ReadonlyArray<{ name: string; size: number }>): string | null {
	if (!picked.length) return "upload.empty";
	if (picked.length > MAX_FILES) return "upload.too_many";
	for (const p of picked) {
		if (!NAME.test(p.name) || p.name.startsWith(".")) return "upload.name";
		if (!TEXT_TYPES.some((ext) => lower(p.name).endsWith(ext))) return "upload.type";
		// A's converter knows a notebook only by a lower-case ".ipynb" (runner/sources.py).
		if (lower(p.name).endsWith(".ipynb") && !p.name.endsWith(".ipynb")) return "upload.notebook_case";
	}
	if (picked.reduce((a, p) => a + p.size, 0) > MAX_TOTAL_BYTES) return "upload.too_big";
	return null;
}

export function uploadSource(picked: readonly PickedFile[], entrypoint?: string): UploadCheck {
	const early = precheck(picked);
	if (early) return { ok: false, reason: early };
	const files: Record<string, string> = {};
	for (const p of picked) {
		if (p.text.includes("\u0000")) return { ok: false, reason: "upload.binary" };
		if (p.name in files) return { ok: false, reason: "upload.duplicate" };
		files[p.name] = p.text;
	}
	// What the API counts: the files as JSON, escapes included (notebooks grow when escaped).
	if (new TextEncoder().encode(JSON.stringify(files)).length > MAX_TOTAL_BYTES) return { ok: false, reason: "upload.too_big" };
	const names = Object.keys(files);
	if (names.some((n) => n.endsWith(".ipynb"))) {
		if (names.length > 1) return { ok: false, reason: "upload.notebook_alone" };
		return { ok: true, source: { kind: "notebook", files }, scripts: [] };
	}
	const scripts = names.filter((n) => lower(n).endsWith(".py")).sort();
	if (!scripts.length) return { ok: false, reason: "upload.no_script" };
	const entry = entrypoint && scripts.includes(entrypoint) ? entrypoint : (ENTRY_ORDER.find((n) => scripts.includes(n)) ?? (scripts.length === 1 ? scripts[0] : undefined));
	if (!entry) return { ok: false, reason: "upload.which", candidates: scripts };
	return { ok: true, source: { kind: "upload", files, entrypoint: entry }, scripts };
}
