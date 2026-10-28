/**
 * New flight: a check-in desk and a departures board. The desk submits a JobRequest to
 * POST /v1/jobs (validated by C, executed by A) with an Idempotency-Key, so a double click
 * never pays twice. Samples on the departures board open straight in the cockpit for free.
 */
import { useId, useRef, useState } from "react";
import { t } from "../../lib/i18n";
import { useAppPanels } from "../../registry";
import { SAMPLES } from "../../workspace/record/load";
import type { JobRequest, ObjectiveKind } from "../../workspace/record/types";
import { go } from "../route";
import { AddonBoundary } from "../../components/AddonBoundary";

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
	const idem = useRef(newKey());
	const ids = { repo: useId(), describe: useId(), upload: useId() };

	const canSubmit = intake === "repo" ? /^(https?:\/\/)?(www\.)?(github\.com|huggingface\.co|colab\.research\.google\.com)\/.+/.test(repo.trim()) : false;

	const submit = async () => {
		setError(null);
		setBusy(true);
		const body: JobRequest = {
			source: { kind: "repo", uri: repo.trim().startsWith("http") ? repo.trim() : `https://${repo.trim()}` },
			objective: { kind: goal, description: description.trim() || null },
			gpu,
			description: description.trim() || null,
		};
		try {
			const res = await fetch(`${API_BASE}/v1/jobs`, {
				method: "POST",
				credentials: "same-origin",
				headers: { "Content-Type": "application/json", "Idempotency-Key": idem.current },
				body: JSON.stringify(body),
			});
			const json = (await res.json().catch(() => null)) as { job_id?: string; error?: { code?: string; message?: string } } | null;
			if (!res.ok || !json?.job_id) {
				const retry = res.headers.get("Retry-After");
				setError(
					json?.error?.message ??
						(res.status === 429 && retry ? t("newflight.error.busy", { s: retry }) : res.status === 404 ? t("newflight.error.not_open") : t("newflight.error.generic")),
				);
				return;
			}
			idem.current = newKey();
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
								<button key={tab.id} type="button" role="tab" aria-selected={intake === tab.id} onClick={() => setIntake(tab.id)}>
									{t(tab.key)}
								</button>
							))}
						</div>
						<div className="pf-desk__body">
							{intake === "repo" ? (
								<div className="pf-pane">
									<div className="pf-field">
										<label htmlFor={ids.repo}>{t("newflight.repo.label")}</label>
										<input id={ids.repo} type="url" inputMode="url" placeholder={t("newflight.repo.placeholder")} value={repo} onChange={(e) => setRepo(e.currentTarget.value)} />
									</div>
									<p className="pf-hint">{t("newflight.repo.hint")}</p>
								</div>
							) : intake === "upload" ? (
								<div className="pf-pane">
									<label className="pf-drop" htmlFor={ids.upload}>
										{t("newflight.upload.label")}
										<small>{t("newflight.upload.limits")}</small>
										<input id={ids.upload} type="file" accept=".py,.ipynb" onChange={() => setError(t("newflight.upload.not_open"))} />
									</label>
								</div>
							) : (
								extraIntake
									.filter((p) => p.id === intake)
									.map((p) => (
										<div key={p.id} className="pf-pane">
											<AddonBoundary id={p.id}>
												<p.component />
											</AddonBoundary>
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
							<span className="pf-hint">{t("newflight.cost_hint")}</span>
							<button type="button" className="pf-btn pf-btn--primary" disabled={!canSubmit || busy} onClick={() => void submit()}>
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
				<p className="pf-notice">{t("newflight.notice")}</p>
			</div>
		</div>
	);
}
