/**
 * New flight: a check-in desk and a departures board. The desk submits a JobRequest to
 * POST /v1/jobs (validated by C, executed by A) with an Idempotency-Key, so a double click
 * never pays twice. Samples on the departures board open straight in the cockpit for free.
 */
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { flag } from "../../lib/flags";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useAppPanels } from "../../registry";
import { SAMPLES } from "../../workspace/record/load";
import type { JobRequest, ObjectiveKind } from "../../workspace/record/types";
import { go } from "../route";
import { AddonBoundary } from "../../components/AddonBoundary";
import { DATA_USE_OK_ID, DataUseNotice, dataUseAccepted } from "../DataUse";
import { ACCEPT, ACCEPT_SCRIPTS, checkinBytes, MAX_TOTAL_BYTES, type PickedFile, precheck, uploadSource } from "../upload";

/** A public GitHub repository link: A's runner fetches from GitHub only, at a pinned commit. */
const GITHUB_REPO = /^(https?:\/\/)?(www\.)?github\.com\/[^/\s]+\/[^/\s]+/;

/** Source: guess, a typing pause long enough that a half-typed link isn't called wrong (UX convention: 0.5 to 1 s). */
const REPO_PAUSE_MS = 800;

/** D's notebook intake panel (addons/notebook); while it is on, notebooks check in there. */
const NOTEBOOK_TAB = "notebook.intake";

/** Same origin: /v1 is served by the web app's own Pages Function (functions/v1). */
const API_BASE = "";

const GOALS: Array<{ kind: ObjectiveKind; key: string }> = [
	{ kind: "min_cost", key: "goal.min_cost" },
	{ kind: "min_time", key: "goal.min_time" },
	{ kind: "fit_memory", key: "goal.fit_memory" },
];

/** Canonical names from B's GPU table (engine/gpus.py, generated into gpus.json). */
const GPUS = ["A100-SXM-80GB", "H100-SXM-80GB", "L40S", "L4", "MI300X"];

type Intake = "repo" | "upload" | string;

function newKey(): string {
	return typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
}

export function NewFlight({ onSample }: { onSample: (id: string) => void }) {
	const extraIntake = useAppPanels("intake");
	const [intake, setIntake] = useState<Intake>("repo");
	const [repo, setRepo] = useState("");
	const [goal, setGoal] = useState<ObjectiveKind>("min_cost");
	const [gpu, setGpu] = useState<string | null>("A100-SXM-80GB");
	const [description, setDescription] = useState("");
	const [error, setError] = useState<string | null>(null);
	const [busy, setBusy] = useState(false);
	const [accepted, setAccepted] = useState(dataUseAccepted);
	const [picked, setPicked] = useState<PickedFile[]>([]);
	const [entry, setEntry] = useState<string | undefined>();
	const [reading, setReading] = useState(false);
	const idem = useRef<{ key: string; body: string | null }>({ key: newKey(), body: null });
	const ids = { repo: useId(), describe: useId(), upload: useId(), entry: useId(), repoHint: useId(), repoError: useId() };
	// Uploads open with live probes (the same flag the API checks before taking a job).
	const uploadsOpen = flag("runner.queue");
	// A notebook tab (D's intake panel) previews what will run, so notebooks go through it there.
	const notebookTab = extraIntake.some((p) => p.id === NOTEBOOK_TAB);
	const check = uploadSource(picked, entry, { notebooks: !notebookTab });
	// Sources handed over by intake panels, by panel id. A panel unmounts when its tab is left and
	// forgets what it showed, so leaving a tab drops its source too (switchTab).
	const [handed, setHanded] = useState<Record<string, JobRequest["source"] | null>>({});
	const handedSource = intake !== "repo" && intake !== "upload" ? (handed[intake] ?? null) : null;
	// Measured once per source, not on every keystroke elsewhere on the desk.
	const handedTooBig = useMemo(() => handedSource !== null && checkinBytes(handedSource) > MAX_TOTAL_BYTES, [handedSource]);
	// One stable setSource per panel, so a panel may call it from an effect without a render loop.
	const setters = useRef(new Map<string, (src: JobRequest["source"] | null) => void>());
	const setterFor = (id: string) => {
		let fn = setters.current.get(id);
		if (!fn) {
			fn = (src) => setHanded((m) => (m[id] === src ? m : { ...m, [id]: src }));
			setters.current.set(id, fn);
		}
		return fn;
	};
	const smooth = (): ScrollBehavior => (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");
	// From the hint to the box and, once it's ticked, back to Check in (two panels up on a phone).
	const fromHint = useRef(false);
	const checkInBtn = useRef<HTMLButtonElement>(null);
	const focusDataUse = () => {
		const box = document.getElementById(DATA_USE_OK_ID);
		fromHint.current = true;
		box?.scrollIntoView({ block: "center", behavior: smooth() });
		box?.focus({ preventScroll: true });
	};
	const onDataUse = (on: boolean) => {
		setAccepted(on);
		if (on && fromHint.current) {
			fromHint.current = false;
			// After the re-render that enables it, so focus lands on a live button.
			requestAnimationFrame(() => {
				checkInBtn.current?.scrollIntoView({ block: "center", behavior: smooth() });
				checkInBtn.current?.focus({ preventScroll: true });
			});
		}
	};
	const switchTab = (next: Intake) => {
		if (next === intake) return;
		setHanded((m) => (m[intake] == null ? m : { ...m, [intake]: null }));
		setError(null);
		setIntake(next);
	};

	// GitHub only: A's runner fetches from GitHub, and the API pins the link to a commit there.
	const repoOk = GITHUB_REPO.test(repo.trim());
	// Say why Check in is grey once the visitor has finished with the field (a paste, leaving it,
	// or a pause in typing), not from the first character of a link still being typed.
	const [repoSettled, setRepoSettled] = useState(false);
	useEffect(() => {
		setRepoSettled(false);
		const id = setTimeout(() => setRepoSettled(true), REPO_PAUSE_MS);
		return () => clearTimeout(id);
	}, [repo]);
	const repoBad = intake === "repo" && repoSettled && repo.trim() !== "" && !repoOk;
	const canSubmit =
		intake === "repo" ? repoOk : intake === "upload" ? uploadsOpen && check.ok : uploadsOpen && handedSource !== null && !handedTooBig;

	const pick = async (list: FileList | null) => {
		if (!list?.length) return;
		setError(null);
		const chosen = Array.from(list);
		// Names and sizes first: a 4 GB checkpoint is refused without being read into the tab.
		const early = precheck(chosen, { notebooks: !notebookTab });
		if (early) {
			setPicked([]);
			setError(t(`newflight.${early}`));
			return;
		}
		setReading(true);
		try {
			setPicked(await Promise.all(chosen.map(async (f) => ({ name: f.name, size: f.size, text: await f.text() }))));
			setEntry(undefined);
		} catch {
			setPicked([]);
			setError(t("newflight.upload.unreadable"));
		} finally {
			setReading(false);
		}
	};

	const submit = async () => {
		setError(null);
		if (intake === "upload" && !check.ok) return;
		if (handedTooBig) return;
		setBusy(true);
		const body: JobRequest = {
			source:
				intake === "upload" && check.ok
					? check.source
					: (handedSource ?? { kind: "repo", uri: repo.trim().startsWith("http") ? repo.trim() : `https://${repo.trim()}` }),
			objective: { kind: goal, description: description.trim() || null },
			gpu,
			description: description.trim() || null,
		};
		// One key per distinct check-in: a retry of the same body reuses it (the API returns the job
		// it already made), and any change to the body is a new check-in with a new key.
		const sent = JSON.stringify(body);
		if (idem.current.body !== sent) idem.current = { key: newKey(), body: sent };
		try {
			const res = await fetch(`${API_BASE}/v1/jobs`, {
				method: "POST",
				credentials: "same-origin",
				headers: { "Content-Type": "application/json", "Idempotency-Key": idem.current.key },
				body: sent,
			});
			const json = (await res.json().catch(() => null)) as { job_id?: string; error?: { code?: string; message?: string } } | null;
			if (!res.ok || !json?.job_id) {
				if (res.status === 409) idem.current = { key: newKey(), body: null }; // a key clash: start afresh
				const retry = res.headers.get("Retry-After");
				setError(
					json?.error?.message ??
						(res.status === 429 && retry ? t("newflight.error.busy", { s: retry }) : res.status === 404 ? t("newflight.error.not_open") : t("newflight.error.generic")),
				);
				return;
			}
			idem.current = { key: newKey(), body: null };
			go("boarding", json.job_id);
		} catch {
			setError(t("newflight.error.offline"));
		} finally {
			setBusy(false);
		}
	};

	return (
		<div className="pf-scroll">
			<div className="pf-inner pf-checkin">
				<header className="pf-checkin__head">
					<span className="pf-eyebrow">{t("newflight.eyebrow")}</span>
					<h1 className="pf-title">{t("newflight.title")}</h1>
					<p className="pf-lede pf-lede--sm">{t("newflight.lede")}</p>
				</header>
				<div className="pf-checkin__grid">
					<section className="glass pf-desk" aria-label={t("newflight.desk")}>
						<div className="pf-seg" role="tablist" aria-label={t("newflight.where")}>
							{[
								{ id: "repo", key: "newflight.tab.repo" },
								{ id: "upload", key: "newflight.tab.upload" },
								...extraIntake.map((p) => ({ id: p.id, key: p.titleKey })),
							].map((tab) => (
								<button key={tab.id} type="button" role="tab" aria-selected={intake === tab.id} onClick={() => switchTab(tab.id)}>
									{t(tab.key)}
								</button>
							))}
						</div>
						<div className="pf-desk__body">
							{intake === "repo" ? (
								<div className="pf-pane">
									<div className="pf-field">
										<label htmlFor={ids.repo}>{t("newflight.repo.label")}</label>
										<input
											id={ids.repo}
											type="url"
											inputMode="url"
											placeholder={t("newflight.repo.placeholder")}
											value={repo}
											aria-invalid={repoBad || undefined}
											aria-describedby={repoBad ? `${ids.repoError} ${ids.repoHint}` : ids.repoHint}
											onChange={(e) => setRepo(e.currentTarget.value)}
											onPaste={() => setTimeout(() => setRepoSettled(true), 0)}
											onBlur={() => setRepoSettled(true)}
										/>
									</div>
									{/* Always mounted, so a screen reader speaks the message when it appears. */}
									<div role="status" aria-live="polite">
										{repoBad ? (
											<p id={ids.repoError} className="pf-error pf-field__error">
												{t("newflight.repo.invalid")}
											</p>
										) : null}
									</div>
									<p id={ids.repoHint} className="pf-hint">
										{t("newflight.repo.hint")}
									</p>
								</div>
							) : intake === "upload" ? (
								<div className="pf-pane">
									<label
										className="pf-drop"
										htmlFor={ids.upload}
										onDragOver={(e) => uploadsOpen && e.preventDefault()}
										onDrop={(e) => {
											if (!uploadsOpen) return;
											e.preventDefault();
											void pick(e.dataTransfer.files);
										}}
									>
										{t("newflight.upload.label")}
										<small>{t(notebookTab ? "newflight.upload.limits_scripts" : "newflight.upload.limits")}</small>
										<input
											id={ids.upload}
											type="file"
											multiple
											accept={notebookTab ? ACCEPT_SCRIPTS : ACCEPT}
											onChange={(e) => {
												const input = e.currentTarget;
												if (!uploadsOpen) setError(t("newflight.upload.not_open"));
												else void pick(input.files).finally(() => (input.value = "")); // the same files can be chosen again
											}}
										/>
									</label>
									{reading ? <p className="pf-hint">{t("newflight.upload.reading")}</p> : null}
									{picked.length ? (
										<div className="pf-upload">
											<ul className="pf-upload__files">
												{picked.map((p, i) => (
													<li key={`${i}:${p.name}`} className="pf-tag">
														{p.name}
													</li>
												))}
											</ul>
											<span className="pf-dim">{t("newflight.upload.files", { n: fmt(picked.length, "count"), size: fmt(picked.reduce((a, p) => a + p.size, 0), "bytes") })}</span>
											<button
												type="button"
												className="pf-btn pf-btn--small"
												onClick={() => {
													setPicked([]);
													setEntry(undefined);
												}}
											>
												{t("newflight.upload.clear")}
											</button>
										</div>
									) : null}
									{(check.ok ? check.scripts : (check.candidates ?? [])).length > 1 ? (
										<div className="pf-field">
											<label htmlFor={ids.entry}>{t("newflight.upload.which_label")}</label>
											<select
												id={ids.entry}
												value={check.ok && check.source.kind === "upload" ? check.source.entrypoint : (entry ?? "")}
												onChange={(e) => setEntry(e.currentTarget.value || undefined)}
											>
												{check.ok ? null : <option value="">{t("newflight.upload.which")}</option>}
												{(check.ok ? check.scripts : (check.candidates ?? [])).map((c) => (
													<option key={c} value={c}>
														{c}
													</option>
												))}
											</select>
										</div>
									) : picked.length && !check.ok ? (
										<p className="pf-error" role="alert">
											{t(`newflight.${check.reason}`)}
										</p>
									) : null}
								</div>
							) : (
								extraIntake
									.filter((p) => p.id === intake)
									.map((p) => (
										<div key={p.id} className="pf-pane">
											<AddonBoundary id={p.id}>
												<p.component setSource={uploadsOpen ? setterFor(p.id) : undefined} />
											</AddonBoundary>
											{handedTooBig ? (
												<p className="pf-error" role="alert">
													{t("newflight.handed.too_big")}
												</p>
											) : null}
										</div>
									))
							)}
						</div>
						<div className="pf-desk__row">
							<span className="pf-desk__label" id={`${ids.describe}-goal`}>
								{t("newflight.goal")}
							</span>
							<div className="pf-choice" role="group" aria-labelledby={`${ids.describe}-goal`}>
								{GOALS.map((g) => (
									<button key={g.kind} type="button" aria-pressed={goal === g.kind} onClick={() => setGoal(g.kind)}>
										{t(g.key)}
									</button>
								))}
							</div>
						</div>
						<div className="pf-desk__row">
							<span className="pf-desk__label" id={`${ids.describe}-gpu`}>
								{t("newflight.gpu")}
							</span>
							<div className="pf-choice" role="group" aria-labelledby={`${ids.describe}-gpu`}>
								{GPUS.map((g) => (
									<button key={g} type="button" aria-pressed={gpu === g} onClick={() => setGpu(g)}>
										{g}
									</button>
								))}
								<button type="button" aria-pressed={gpu === null} onClick={() => setGpu(null)}>
									{t("newflight.gpu.unsure")}
								</button>
							</div>
						</div>
						<div className="pf-field">
							<label htmlFor={ids.describe}>{t("newflight.describe.label")}</label>
							<textarea id={ids.describe} rows={2} placeholder={t("newflight.describe.placeholder")} value={description} onChange={(e) => setDescription(e.currentTarget.value)} />
						</div>
						{error ? (
							<p className="pf-error" role="alert">
								{error}
							</p>
						) : null}
						<div className="pf-desk__foot">
							{accepted ? (
								// Source: measured, A's run a-smoke-l4-main-4 (Oct 28): the LoRA zoo script from
								// upload to record in 6 min 45 s on an L4 (docs/status/a.md). No dollar figure until
								// the sandbox's CPU and memory cost is read off the bill.
								<span className="pf-hint">{t("newflight.cost_hint")}</span>
							) : (
								// On a phone the box is two panels down: the hint takes the visitor there.
								<button type="button" className="pf-link pf-hint" onClick={focusDataUse}>
									{t("datause.needed")}
								</button>
							)}
							<button ref={checkInBtn} type="button" className="pf-btn pf-btn--primary" disabled={!canSubmit || busy || !accepted} onClick={() => void submit()}>
								{busy ? t("newflight.checking_in") : t("newflight.check_in")}
							</button>
						</div>
					</section>
					<section className="glass pf-departures" aria-label={t("newflight.samples")}>
						<div className="pf-departures__head">
							<b>{t("newflight.departures")}</b>
							<span>{t("newflight.departures.sub")}</span>
						</div>
						<div className="pf-departures__cols" aria-hidden="true">
							<span>{t("newflight.col.flight")}</span>
							<span>{t("newflight.col.job")}</span>
							<span>{t("newflight.col.gpu")}</span>
							<span>{t("newflight.col.status")}</span>
						</div>
						{SAMPLES.map((s) => (
							<button key={s.id} type="button" className="pf-departures__row" onClick={() => onSample(s.id)} aria-label={t("newflight.open_sample", { job: t(s.titleKey) })}>
								<span className="pf-departures__code">{s.code}</span>
								<span className="pf-departures__job">{t(s.titleKey)}</span>
								<span className="pf-departures__gpu">{s.gpu}</span>
								<span className="pf-departures__status">{t("newflight.ready")}</span>
							</button>
						))}
						<p className="pf-departures__foot">{t("newflight.departures.foot")}</p>
					</section>
				</div>
				<DataUseNotice onChange={onDataUse} />
				<p className="pf-notice">{t("newflight.notice")}</p>
			</div>
		</div>
	);
}
