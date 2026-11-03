/**
 * Check in by command (contract 0.6.3, `source.kind: "command"`, behind `runner.command_checkin`):
 * the visitor types a console command such as `lm_eval --model hf ...`, and the runner runs it by
 * name, as its words, with no shell. Only the commands on A's list run (`COMMANDS` in
 * python/preflight/runner/sources.py; api/tests/test_command_words.py keeps the two in step).
 */

/** Source: derived, A's `runner.sources.COMMANDS` (Nov 2: lm_eval only; vLLM's bench isn't supported). */
export const COMMANDS = ["lm_eval"] as const;

/** Source: derived, A's limits (`runner.sources`: MAX_ARGS, MAX_ARG_LEN, MAX_COMMAND_CHARS), which the API also checks. */
export const MAX_WORDS = 64;
export const MAX_WORD_CHARS = 4096;
export const MAX_COMMAND_CHARS = 8192;

/** Characters a shell would act on outside quotes (pipes, redirects, a second command, expansions): there's no shell here. */
const SHELL_CHARS = new Set(["|", "&", ";", ">", "<", "$", "`"]);

export type CommandCheck =
	| { ok: true; words: string[] }
	| { ok: false; reason: "empty" | "quote" | "shell" | "unknown" | "too_long"; word?: string };

/** The command as its words; only a command on A's list runs. */
export function commandWords(text: string): CommandCheck {
	const split = splitWords(text);
	if (!split.ok) return split;
	if (!split.words.length) return { ok: false, reason: "empty" };
	if (!(COMMANDS as readonly string[]).includes(split.words[0]!)) return { ok: false, reason: "unknown", word: split.words[0] };
	return split;
}

/**
 * A repo script's arguments (contract `source.args`; PM, Nov 3: the repo link's Arguments field), by
 * the same rules as a command: words, no shell. Empty is fine: the script runs with none.
 */
export function argWords(text: string): CommandCheck {
	return splitWords(text);
}

/**
 * Text as its words, split as a POSIX shell would (spaces separate; single quotes keep text as is;
 * double quotes and a backslash escape), refusing what only a shell could run. No words is ok here.
 */
function splitWords(text: string): CommandCheck {
	if (text.length > MAX_COMMAND_CHARS) return { ok: false, reason: "too_long" };
	const words: string[] = [];
	let cur = "";
	let started = false;
	let i = 0;
	const push = () => {
		if (started) words.push(cur);
		cur = "";
		started = false;
	};
	try {
		while (i < text.length) {
			const c = text[i]!;
			if (/\s/.test(c)) {
				push();
				i++;
			} else if (c === "'") {
				const end = text.indexOf("'", i + 1);
				if (end < 0) return { ok: false, reason: "quote" };
				cur += text.slice(i + 1, end);
				started = true;
				i = end + 1;
			} else if (c === '"') {
				i++;
				started = true;
				for (;;) {
					if (i >= text.length) return { ok: false, reason: "quote" };
					const d = text[i]!;
					if (d === '"') {
						i++;
						break;
					}
					if (d === "\\" && i + 1 < text.length && '"\\$`'.includes(text[i + 1]!)) {
						cur += text[i + 1];
						i += 2;
					} else if (d === "$" || d === "`") {
						throw { reason: "shell", word: d }; // a shell expands these inside double quotes too
					} else {
						cur += d;
						i++;
					}
				}
			} else if (c === "\\") {
				if (i + 1 < text.length) cur += text[i + 1];
				started = true;
				i += 2;
			} else if (SHELL_CHARS.has(c)) {
				throw { reason: "shell", word: c };
			} else {
				cur += c;
				started = true;
				i++;
			}
		}
		push();
	} catch (e) {
		const err = e as { reason: "shell"; word: string };
		return { ok: false, reason: err.reason, word: err.word };
	}
	if (words.length > MAX_WORDS || words.some((w) => w.length > MAX_WORD_CHARS || w.includes("\0"))) return { ok: false, reason: "too_long" };
	return { ok: true, words };
}

/** Source: derived, A's runner (`sources._pick_entrypoint`): a file path among the repository's files. */
export const MAX_ENTRY_CHARS = 512;

export type EntryCheck = { ok: true; path: string | null } | { ok: false; reason: "path" | "not_py" | "too_long" };

/**
 * The repo link's Entry point (contract `source.entrypoint`): empty lets Preflight pick (train.py,
 * main.py, run.py, ... or the only .py file); otherwise a relative path to a .py file in the repo,
 * with no "..", no leading "/" and no hidden folders. The runner checks the file exists.
 */
export function entryPath(text: string): EntryCheck {
	const p = text.trim().replace(/^\.\//, "");
	if (!p) return { ok: true, path: null };
	if (p.length > MAX_ENTRY_CHARS) return { ok: false, reason: "too_long" };
	// A's `probes.local._PATH_OK` (letters, digits, . _ - / and spaces), and only a path the repo reader keeps:
	// no hidden folder or file (sources._wanted), no "." or ".." step, no empty step (review, Nov 3).
	if (!/^[A-Za-z0-9._\-/ ]+$/.test(p) || p.split("/").some((s) => s === "" || s === "." || s === ".." || s.startsWith(".") || s !== s.trim())) return { ok: false, reason: "path" };
	if (!p.endsWith(".py")) return { ok: false, reason: "not_py" };
	return { ok: true, path: p };
}
