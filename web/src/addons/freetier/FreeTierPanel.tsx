/**
 * Free GPUs (D4): a band under the cockpit's whole run, one row per free tier, saying whether
 * this plan finishes inside one session there. Reads only what C's registry passes in
 * (record, knobs, estimates at the current settings), so it moves with every knob.
 */
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import type { WorkspaceContextValue } from "../../registry";
import type { Estimate } from "../../workspace/record/types";
import { provenanceKey } from "../../workspace/components/StatusMark";
import { isFixture } from "../../workspace/record/model";
import { canonicalGpu, checkPreset, gpuMemoryBytes, type Verdict } from "./check";
import { PRESETS, type Preset, type Source } from "./presets";

const SHORT: Record<string, string> = { "P100-16GB": "P100", "H100-SXM-80GB": "H100", "H200-SXM-141GB": "H200" };

function short(gpu: string): string {
	return SHORT[gpu] ?? gpu;
}

/** "T4", "T4 or P100", "L40S, H100 or H200". */
function gpuList(p: Preset): string {
	const names = p.gpus.map(short);
	if (names.length < 2) return names.join("");
	return `${names.slice(0, -1).join(", ")}${t("addon.freetier.or")}${names[names.length - 1]}`;
}

const CHIP: Record<Verdict["kind"], string> = {
	one_session: "pf-status pf-status--fits",
	sessions: "pf-status pf-status--tight",
	no_memory: "pf-status pf-status--fails",
	other_gpu: "pf-addon-freetier__chip",
	abstain: "pf-addon-freetier__chip",
};

function chip(v: Verdict, p: Preset): string {
	if (v.kind === "sessions") return t("addon.freetier.chip.sessions", { n: v.sessions });
	if (v.kind === "other_gpu") return t("addon.freetier.chip.other_gpu", { gpus: gpuList(p) });
	return t(`addon.freetier.chip.${v.kind}`);
}

/** The memory need says where its number comes from, as the readouts do (a sample's is illustrative). */
function say(v: Verdict, p: Preset, current: string | null, gpu: string | null, memStatus: string, mem: Estimate | undefined): string {
	switch (v.kind) {
		case "sessions":
			return t("addon.freetier.say.sessions", { every: fmt(v.saveEveryS, "s") });
		case "no_memory": {
			const on = gpu && p.gpus.includes(gpu) ? gpu : (p.gpus.find((g) => gpuMemoryBytes(g) === v.haveBytes) ?? p.gpus[0] ?? "");
			// Judged on the top of the range on purpose; the estimate itself is what the cockpit shows (B, Nov 3).
			const pct = Math.round((mem?.level ?? 0.9) * 100);
			const point = typeof mem?.point === "number" ? fmt(mem.point, "bytes") : fmt(v.needBytes, "bytes");
			return t("addon.freetier.say.no_memory", { need: fmt(v.needBytes, "bytes"), pct, point, status: memStatus, gpu: short(on), have: fmt(v.haveBytes, "bytes") });
		}
		case "other_gpu":
			return t("addon.freetier.say.other_gpu", { current: current ?? "", gpus: gpuList(p) });
		default:
			return t(`addon.freetier.say.${v.kind}`);
	}
}

function readDate(iso: string): string {
	return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(`${iso}T00:00:00Z`));
}

function uniqueSources(): Source[] {
	const seen = new Map<string, Source>();
	for (const p of PRESETS) for (const s of p.sources) seen.set(s.url, s);
	return [...seen.values()];
}

export function FreeTierPanel({ record, knobs, estimates }: WorkspaceContextValue & { nodeId?: string }) {
	const raw = typeof knobs.gpu_type === "string" ? knobs.gpu_type : (record.env?.gpu?.name ?? null);
	const gpu = canonicalGpu(raw);
	const input = { gpu, memory: estimates["memory.peak"], time: estimates["time.total"], cost: estimates["cost.total"] };
	const mem = estimates["memory.peak"];
	const memStatus = mem ? t(provenanceKey(mem.status, isFixture(record))).toLowerCase() : "";
	const usesBf16 = typeof knobs.precision === "string" && knobs.precision.toLowerCase().includes("bf16");
	return (
		<section className="pf-addon-freetier" aria-labelledby="pf-addon-freetier-title">
			<h3 id="pf-addon-freetier-title" className="pf-addon-freetier__title">
				{t("addon.freetier.title")}
			</h3>
			<p className="pf-addon-freetier__lede">{t("addon.freetier.lede")}</p>
			<ul className="pf-addon-freetier__rows">
				{PRESETS.map((p) => {
					const v = checkPreset(p, input);
					const name = t(`addon.freetier.name.${p.id}`);
					return (
						<li key={p.id} className="pf-addon-freetier__row" aria-label={name}>
							<div className="pf-addon-freetier__head">
								<b>{name}</b>
								<span className="pf-addon-freetier__meta">{gpuList(p)}</span>
								<span className="pf-addon-freetier__meta">{t("addon.freetier.cap", { h: p.sessionCapS / 3600 })}</span>
								<span className={CHIP[v.kind]}>{chip(v, p)}</span>
							</div>
							<p className="pf-addon-freetier__say">{say(v, p, raw ? short(gpu ?? raw) : null, gpu, memStatus, mem)}</p>
							{"overCredit" in v && v.overCredit && p.monthlyCreditUsd !== undefined ? (
								<p className="pf-addon-freetier__say">{t("addon.freetier.over_credit", { credit: fmt(p.monthlyCreditUsd, "usd") })}</p>
							) : null}
							{p.fp16Only ? (
								<p className="pf-addon-freetier__fp16">
									{usesBf16
										? t("addon.freetier.fp16_switch", { gpu: short(gpu && p.gpus.includes(gpu) ? gpu : (p.gpus[0] ?? "")) })
										: t("addon.freetier.fp16")}
								</p>
							) : null}
							<p className="pf-addon-freetier__note">{t(`addon.freetier.note.${p.id}`)}</p>
						</li>
					);
				})}
			</ul>
			<p className="pf-addon-freetier__sources">
				{t("addon.freetier.sources")}:{" "}
				{uniqueSources().map((s, i) => (
					<span key={s.url}>
						{i > 0 ? "; " : null}
						<a href={s.url} target="_blank" rel="noopener noreferrer">
							{s.label}
						</a>{" "}
						({t("addon.freetier.read", { date: readDate(s.read) })})
					</span>
				))}
			</p>
		</section>
	);
}
