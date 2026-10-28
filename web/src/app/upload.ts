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

export type UploadCheck = { ok: true; source: UploadSource } | { ok: false; reason: string; candidates?: string[] };

// Source: derived, the API refuses a check-in body over 6 MiB (api/preflight_api/routes/jobs.py
// MAX_BODY_BYTES); 5 MB of files leaves room for JSON escaping, as the check-in copy says.
export const MAX_TOTAL_BYTES = 5 * 1000 * 1000;
// Source: derived, A's text-file types for repos (upload-format.md), plus notebooks and .cfg.
const TEXT_TYPES = [".py", ".ipynb", ".sh", ".sbatch", ".slurm", ".yaml", ".yml", ".json", ".toml", ".cfg", ".txt", ".md"];
// Source: derived, A's rule for picking the entry point (upload-format.md, "Entry point").
const ENTRY_ORDER = ["train.py", "main.py", "run.py", "finetune.py", "pretrain.py"];
const NAME = /^[A-Za-z0-9._ -]+$/;

export const ACCEPT = TEXT_TYPES.join(",");

export function uploadSource(picked: readonly PickedFile[], entrypoint?: string): UploadCheck {
	if (!picked.length) return { ok: false, reason: "upload.empty" };
	const files: Record<string, string> = {};
	let total = 0;
	for (const p of picked) {
		if (!NAME.test(p.name) || p.name.startsWith(".")) return { ok: false, reason: "upload.name" };
		if (!TEXT_TYPES.some((ext) => p.name.toLowerCase().endsWith(ext))) return { ok: false, reason: "upload.type" };
		if (p.text.includes("\u0000")) return { ok: false, reason: "upload.binary" };
		if (p.name in files) return { ok: false, reason: "upload.duplicate" };
		files[p.name] = p.text;
		total += p.size;
	}
	if (total > MAX_TOTAL_BYTES) return { ok: false, reason: "upload.too_big" };
	const names = Object.keys(files);
	const notebooks = names.filter((n) => n.toLowerCase().endsWith(".ipynb"));
	if (notebooks.length) {
		if (names.length > 1) return { ok: false, reason: "upload.notebook_alone" };
		return { ok: true, source: { kind: "notebook", files } };
	}
	const scripts = names.filter((n) => n.endsWith(".py")).sort();
	if (!scripts.length) return { ok: false, reason: "upload.no_script" };
	const entry = entrypoint && scripts.includes(entrypoint) ? entrypoint : (ENTRY_ORDER.find((n) => scripts.includes(n)) ?? (scripts.length === 1 ? scripts[0] : undefined));
	if (!entry) return { ok: false, reason: "upload.which", candidates: scripts };
	return { ok: true, source: { kind: "upload", files, entrypoint: entry } };
}
