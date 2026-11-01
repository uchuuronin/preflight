/**
 * The Weights & Biases file in "Take it with you", as one more row beside the exports panel's
 * (D's addons/exports: the same markup and classes, so the section reads as one list): open it to
 * read the file, copy it, or download it. See wandb.ts.
 */
import { useState } from "react";
import { t } from "../../lib/i18n";
import type { ExportPlan } from "./plan";
import { WANDB_FILE, wandbFile } from "./wandb";

export function WandbExport({ plan, illustrative = false }: { plan: ExportPlan; illustrative?: boolean }) {
	const [said, setSaid] = useState("");
	const text = wandbFile(plan, illustrative);
	const name = t("exports.wandb.title");
	// Cleared first, so the same message twice is announced twice.
	const say = (m: string) => {
		setSaid("");
		setTimeout(() => setSaid(m), 60);
	};
	const download = () => {
		if (typeof URL.createObjectURL !== "function") return say(t("exports.wandb.download_failed"));
		const url = URL.createObjectURL(new Blob([text], { type: "text/x-python" }));
		const a = document.createElement("a");
		a.href = url;
		a.download = WANDB_FILE;
		document.body.append(a);
		a.click();
		a.remove();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	};
	const copy = async () => {
		try {
			await navigator.clipboard.writeText(text);
			say(t("exports.wandb.copied"));
		} catch {
			say(t("exports.wandb.copy_failed"));
		}
	};
	return (
		<>
			<ul className="pf-addon-exports__items pf-exports__wandb">
				<li className="pf-addon-exports__item">
					<details>
						<summary>
							<span className="pf-addon-exports__name">{name}</span>
							<span className="pf-addon-exports__say">{t(plan.shareUrl ? "exports.wandb.body_link" : "exports.wandb.body", { file: WANDB_FILE })}</span>
						</summary>
						<pre className="pf-addon-exports__text" tabIndex={0} aria-label={name}>
							{text}
						</pre>
					</details>
					<div className="pf-addon-exports__actions">
						<button type="button" className="pf-btn pf-btn--small" onClick={() => void copy()} aria-label={`${t("exports.wandb.copy")}: ${name}`}>
							{t("exports.wandb.copy")}
						</button>
						<button type="button" className="pf-btn pf-btn--small pf-btn--quiet" onClick={download} aria-label={t("exports.wandb.download", { file: WANDB_FILE })}>
							{t("exports.wandb.download_short")}
						</button>
					</div>
				</li>
			</ul>
			<p className="pf-visually-hidden" role="status" aria-live="polite">
				{said}
			</p>
		</>
	);
}
