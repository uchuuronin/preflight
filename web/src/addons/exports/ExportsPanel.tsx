/**
 * Exports: a band under the cockpit's whole run with the plan, the edits, a Slurm batch file,
 * the command and the record's link, each to copy or download. Everything is built by build.ts
 * from what C's registry passes in (record, knobs, estimates at the current settings).
 */
import { useEffect, useId, useRef, useState } from "react";
import { href, useRoute } from "../../app/route";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import type { WorkspaceContextValue } from "../../registry";
import { isFixture } from "../../workspace/record/model";
import { askedTime, changes, commandText, configText, editsText, type ExportInput, gpu, isApi, sbatchText } from "./build";

interface Item {
	id: string;
	name: string;
	say: string;
	text: string;
	file: string | null;
	type: string;
}

/** Hand the visitor a file; false when this browser can't. */
function save(text: string, file: string, type: string): boolean {
	if (typeof URL.createObjectURL !== "function") return false;
	const url = URL.createObjectURL(new Blob([text], { type }));
	const a = document.createElement("a");
	a.href = url;
	a.download = file;
	document.body.append(a);
	a.click();
	a.remove();
	setTimeout(() => URL.revokeObjectURL(url), 1000); // some browsers start the download late
	return true;
}

export function ExportsPanel({ record, knobs, estimates }: WorkspaceContextValue) {
	const route = useRoute();
	const headingId = useId();
	const [said, setSaid] = useState("");
	const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
	useEffect(() => () => {
		if (timer.current) clearTimeout(timer.current);
	}, []);
	// Clear, then set, so the same message twice is announced twice.
	function say(text: string): void {
		setSaid("");
		if (timer.current) clearTimeout(timer.current);
		timer.current = setTimeout(() => setSaid(text), 60);
	}
	const ref = (route.screen === "cockpit" && route.param) || record.share_token;
	const link = `${window.location.origin}${window.location.pathname}${href("cockpit", ref)}`;
	const illustrative = isFixture(record);
	const input: ExportInput = { record, knobs, estimates, link, illustrative };

	const cs = changes(input);
	const tally = (how: string) => cs.filter((c) => c.how === how).length;
	const g = gpu(input);
	const gpus = `${g.count} x ${g.name ?? "GPU"}`;
	const asked = askedTime(input);
	const api = isApi(record);

	const items: Item[] = [
		{
			id: "plan",
			name: t("addon.exports.plan.name"),
			say: t("addon.exports.plan.say"),
			text: configText(input),
			file: "preflight-plan.json",
			type: "application/json",
		},
		{
			id: "edits",
			name: t("addon.exports.edits.name"),
			say: cs.length
				? `${t(cs.length === 1 ? "addon.exports.edits.count_one" : "addon.exports.edits.count_other", { n: cs.length })} ${(
						["exact", "by_hand", "code", "launch"] as const
					)
						.filter((h) => tally(h))
						.map((h) => t(`addon.exports.edits.${h}${tally(h) === 1 ? "_one" : ""}`, { n: tally(h) }))
						.join(", ")}.${tally("exact") ? ` ${t("addon.exports.edits.run")}` : ""}`
				: t("addon.exports.edits.none"),
			text: editsText(input),
			file: "preflight-edits.sh",
			type: "text/x-sh",
		},
		{
			id: "sbatch",
			name: t("addon.exports.sbatch.name"),
			say:
				asked === null
					? t(api ? "addon.exports.sbatch.say_api_untimed" : "addon.exports.sbatch.say_untimed", { gpus })
					: asked.capped
						? t(api ? "addon.exports.sbatch.say_api_capped" : "addon.exports.sbatch.say_capped", { gpus, time: fmt(asked.ask, "s"), need: fmt(asked.need, "s") })
						: t(api ? "addon.exports.sbatch.say_api" : "addon.exports.sbatch.say", { gpus, time: fmt(asked.ask, "s") }),
			text: sbatchText(input),
			file: "preflight.sbatch",
			type: "text/x-sh",
		},
		{
			id: "command",
			name: t("addon.exports.command.name"),
			say: t("addon.exports.command.say"),
			text: commandText(input),
			file: null,
			type: "text/plain",
		},
		{
			id: "link",
			name: t("addon.exports.link.name"),
			say: t("addon.exports.link.say"),
			text: link,
			file: null,
			type: "text/plain",
		},
	];

	async function copy(item: Item): Promise<void> {
		try {
			await navigator.clipboard.writeText(item.text);
			say(t("addon.exports.copied", { what: item.name }));
		} catch {
			say(t("addon.exports.copy_failed"));
		}
	}

	const patches = record.patches ?? [];

	return (
		<section className="pf-addon-exports" aria-labelledby={headingId}>
			<h2 className="pf-addon-exports__title" id={headingId}>
				{t("addon.exports.title")}
			</h2>
			<p className="pf-addon-exports__lede">
				{t("addon.exports.lede")}
				{illustrative ? ` ${t("addon.exports.illustrative")}` : ""}
			</p>
			<ul className="pf-addon-exports__items">
				{items.map((item) => (
					<li key={item.id} className="pf-addon-exports__item">
						<details>
							<summary>
								<span className="pf-addon-exports__name">{item.name}</span>
								<span className="pf-addon-exports__say">{item.say}</span>
							</summary>
							<pre className="pf-addon-exports__text" tabIndex={0} aria-label={item.name}>
								{item.text}
							</pre>
							{item.id === "edits" && patches.length ? (
								<div className="pf-addon-exports__patches">
									<h3>{t("addon.exports.patches")}</h3>
									<ul>
										{patches.map((p) => (
											<li key={p.id}>
												{p.title} · {t(`addon.exports.patch.${p.verify?.status ?? "pending"}`)}
											</li>
										))}
									</ul>
								</div>
							) : null}
						</details>
						<div className="pf-addon-exports__actions">
							<button type="button" className="pf-btn pf-btn--small" onClick={() => void copy(item)} aria-label={`${t("addon.exports.copy")}: ${item.name}`}>
								{t("addon.exports.copy")}
							</button>
							{item.file ? (
								<button
									type="button"
									className="pf-btn pf-btn--small pf-btn--quiet"
									onClick={() => {
										if (!save(item.text, item.file as string, item.type)) say(t("addon.exports.download_failed"));
									}}
									aria-label={`${t("addon.exports.download")}: ${item.file}`}
								>
									{t("addon.exports.download")}
								</button>
							) : null}
						</div>
					</li>
				))}
			</ul>
			<p className="pf-visually-hidden" role="status" aria-live="polite">
				{said}
			</p>
		</section>
	);
}
