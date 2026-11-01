/** The Weights & Biases file in "Take it with you": download it, or copy it. See wandb.ts. */
import { useState } from "react";
import { t } from "../../lib/i18n";
import type { ExportPlan } from "./plan";
import { WANDB_FILE, wandbFile } from "./wandb";

export function WandbExport({ plan }: { plan: ExportPlan }) {
	const [said, setSaid] = useState<string | null>(null);
	const text = wandbFile(plan);
	const download = () => {
		const url = URL.createObjectURL(new Blob([text], { type: "text/x-python" }));
		const a = document.createElement("a");
		a.href = url;
		a.download = WANDB_FILE;
		a.click();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	};
	const copy = async () => {
		try {
			await navigator.clipboard.writeText(text);
			setSaid(t("exports.wandb.copied"));
		} catch {
			setSaid(t("exports.wandb.copy_failed"));
		}
	};
	return (
		<div className="glass pf-exports__wandb">
			<b>{t("exports.wandb.title")}</b>
			<p className="pf-dim">{t("exports.wandb.body", { file: WANDB_FILE })}</p>
			<div className="pf-guide__actions">
				<button type="button" className="pf-btn pf-btn--small" onClick={download}>
					{t("exports.wandb.download", { file: WANDB_FILE })}
				</button>
				<button type="button" className="pf-btn pf-btn--small" onClick={copy}>
					{t("exports.wandb.copy")}
				</button>
			</div>
			{said ? (
				<p className="pf-dim" role="status">
					{said}
				</p>
			) : null}
		</div>
	);
}
