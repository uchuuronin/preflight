/**
 * D5 · Notebook intake, a tab on C's check-in desk. The visitor chooses or drops a .ipynb; D shows
 * what will run (the script A converts it to, the packages it will install, what is left out) and
 * hands the desk a `notebook` source. C's desk passes `setSource` when it can check in a source
 * from an intake panel (docs/requests/from-d.md); until then the panel previews and says so.
 */
import { type DragEvent, useEffect, useId, useRef, useState } from "react";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { type NotebookError, type NotebookPreview, type NotebookSource, prepareNotebook } from "./convert";

/**
 * Largest file the browser reads before converting.
 * [measured: JSON.parse of a 100 MB notebook in Chromium took 78 to 88 ms (5 runs, 2-core cloud
 * container, Oct 28); D parses once and reads the file asynchronously, so the page stays inside
 * the 400 ms interaction budget. Choose to preview took 1.3 to 1.5 s end to end, most of it the
 * file read, with "Reading <name>" on screen.]
 */
export const MAX_READ_BYTES = 100 * 1024 * 1024;

/** Lines of the script shown before "and N more lines" [guess: enough to recognise the script without a long page]. */
const PREVIEW_LINES = 200;

type Failure = NotebookError | "read_failed" | "too_big_file" | "unexpected";

type State =
	| { at: "empty" }
	| { at: "reading"; name: string }
	| { at: "ready"; name: string; preview: NotebookPreview }
	| { at: "failed"; error: Failure };

function plural(base: string, n: number): string {
	return t(`${base}.${n === 1 ? "one" : "other"}`, { n });
}

function head(text: string): { shown: string; more: number } {
	const lines = text.split("\n");
	if (lines.at(-1) === "") lines.pop();
	return { shown: lines.slice(0, PREVIEW_LINES).join("\n"), more: Math.max(0, lines.length - PREVIEW_LINES) };
}

export function NotebookIntake({ setSource }: { setSource?: (source: NotebookSource | null) => void }) {
	const [state, setState] = useState<State>({ at: "empty" });
	const [dragging, setDragging] = useState(false);
	const id = useId();
	// Each chosen file gets a number; a read that finishes after a newer choice, or after the tab
	// closed, is dropped, so the desk never holds a notebook other than the one on screen.
	const latest = useRef(0);
	const mounted = useRef(false);
	useEffect(() => {
		mounted.current = true;
		return () => {
			mounted.current = false;
		};
	}, []);

	async function take(file: File | undefined) {
		if (!file) return;
		const mine = ++latest.current;
		const current = () => mounted.current && latest.current === mine;
		setSource?.(null);
		setState({ at: "reading", name: file.name });
		const fail = (error: Failure) => {
			if (!current()) return;
			setState({ at: "failed", error });
			setSource?.(null);
		};
		if (!/\.ipynb$/i.test(file.name)) return fail("not_ipynb");
		if (file.size > MAX_READ_BYTES) return fail("too_big_file");
		let text: string;
		try {
			text = await file.text();
		} catch {
			return fail("read_failed");
		}
		if (!current()) return;
		let out: ReturnType<typeof prepareNotebook>;
		try {
			out = prepareNotebook(file.name, text);
		} catch (error) {
			console.error("[preflight] notebook intake", error);
			return fail("unexpected");
		}
		if (!out.ok) return fail(out.error);
		setState({ at: "ready", name: Object.keys(out.source.files)[0], preview: out.preview });
		setSource?.(out.source);
	}

	function onDrop(e: DragEvent<HTMLElement>) {
		e.preventDefault();
		setDragging(false);
		void take(e.dataTransfer?.files?.[0]);
	}

	const status =
		state.at === "reading"
			? t("addon.notebook.reading", { name: state.name })
			: state.at === "ready"
				? t("addon.notebook.runs_as", { name: state.name, script: state.preview.entrypoint })
				: "";

	return (
		<div className="pf-addon-notebook">
			<label
				className={`pf-drop pf-addon-notebook__drop${dragging ? " is-dragging" : ""}`}
				htmlFor={`${id}-file`}
				data-testid="notebook-drop"
				onDragOver={(e) => {
					e.preventDefault();
					setDragging(true);
				}}
				onDragLeave={() => setDragging(false)}
				onDrop={onDrop}
			>
				{state.at === "ready" || state.at === "failed" ? t("addon.notebook.choose_again") : t("addon.notebook.choose")}
				<small>{t("addon.notebook.drop")}</small>
				<input
					id={`${id}-file`}
					type="file"
					accept=".ipynb"
					aria-describedby={`${id}-privacy`}
					onChange={(e) => {
						void take(e.currentTarget.files?.[0]);
						e.currentTarget.value = "";
					}}
				/>
			</label>
			<p className="pf-hint" id={`${id}-privacy`}>
				{t("addon.notebook.privacy")}
			</p>
			<p className="pf-addon-notebook__status" role="status">
				{status}
			</p>
			{state.at === "failed" ? (
				<p className="pf-error" role="alert">
					{t(`addon.notebook.error.${state.error}`, { size: fmt(MAX_READ_BYTES, "bytes") })}
				</p>
			) : null}
			{state.at === "ready" ? <Preview preview={state.preview} canCheckIn={!!setSource} /> : null}
		</div>
	);
}

function Preview({ preview, canCheckIn }: { preview: NotebookPreview; canCheckIn: boolean }) {
	const script = head(preview.files[preview.entrypoint] ?? "");
	const facts = [
		plural("addon.notebook.code_cells", preview.codeCells),
		preview.otherCells ? plural("addon.notebook.other_cells", preview.otherCells) : null,
		preview.commentedLines ? plural("addon.notebook.magic", preview.commentedLines) : null,
		preview.requirements.length ? plural("addon.notebook.packages", preview.requirements.length) : null,
		preview.removedBytes ? t("addon.notebook.removed", { size: fmt(preview.removedBytes, "bytes") }) : null,
	].filter(Boolean);
	return (
		<div className="pf-addon-notebook__preview">
			{preview.renamedFrom ? <p className="pf-hint">{t("addon.notebook.renamed", { name: preview.renamedFrom })}</p> : null}
			<ul className="pf-addon-notebook__facts">
				{facts.map((f) => (
					<li key={f}>{f}</li>
				))}
			</ul>
			<details className="pf-addon-notebook__files">
				<summary>{t("addon.notebook.what_runs")}</summary>
				<figure>
					<figcaption>{preview.entrypoint}</figcaption>
					<pre tabIndex={0} aria-label={preview.entrypoint}>
						{script.shown}
						{script.more ? `\n# ${t("addon.notebook.more_lines", { n: script.more })}` : ""}
					</pre>
				</figure>
				{preview.requirements.length ? (
					<figure>
						<figcaption>requirements.txt</figcaption>
						<pre tabIndex={0} aria-label="requirements.txt">
							{preview.files["requirements.txt"]}
						</pre>
					</figure>
				) : null}
			</details>
			{canCheckIn ? null : <p className="pf-hint">{t("addon.notebook.not_open")}</p>}
		</div>
	);
}
